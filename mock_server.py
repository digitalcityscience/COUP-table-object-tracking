"""Run the real tracking server against a simulated table, with a keyboard CLI to move the blocks.

    python mock_server.py            # ws://0.0.0.0:8053, same protocol as the rig
    python mock_server.py --help     # ports, motion modes, sandboxing

Why this exists
---------------
The frontend's whole tracking path -- virtual map-calibration markers, the `map_calibration`
handshake, the building GeoJSON feed, `markers_on_table`, scan progress, registration and
per-building calibration -- can only be exercised next to the physical table. That makes every
frontend change a trip to the rig, and makes bugs that only appear when a block *moves*
essentially undebuggable.

What it is
----------
Not a reimplementation. This module imports `server` and runs it unmodified; it replaces exactly
one thing -- the detection thread that turns camera frames into marker snapshots -- with
`mock_table.MockTable`. Every byte on the websocket is produced by the same code the rig runs, so
a frontend that works against this works against the table, and a protocol change in `server.py`
reaches the mock for free.

The seam is `server.tracking_queue`. `server._detection_worker` captures, stitches and detects,
then pushes `{marker_id: [x, y, rotation, camera_id]}` onto it every 200 ms; `_feed_worker` below
pushes the same shape on the same cadence from a synthetic table. Nothing downstream can tell the
difference, because nothing downstream ever sees a camera.

Sandboxing
----------
Registration and building calibration *write catalogs*, and refusing to run them would cut out
half of what there is to develop against. So by default the mock copies both catalogs and the
calibration record into `mock_state/` and points `server` at the copies, leaving the real files
untouched. `--no-sandbox` writes to the real ones (for when you actually mean to), and `--reset`
throws the sandbox away and starts from a fresh copy.
"""

from __future__ import annotations

import argparse
import asyncio
import builtins
import os
import queue as queue_module
import shutil
import threading
from pathlib import Path

import server
from mock_table import DEFAULT_LAYOUT, MockTable
from physical_building_catalog import load_catalog, marker_index
from session_store import SessionStore

#: How often a snapshot is published, matching `server._detection_worker`'s default 200 ms. The
#: frontend's stability gates are written against this cadence (`REFERENCE_ROTATION_BUFFER` holds
#: about six seconds at it), so it is a contract, not a tuning knob -- `--interval` exists to make
#: a slow rig reproducible, not to run faster.
SNAPSHOT_INTERVAL_SECONDS = 0.2

SANDBOX_DIR = Path(server._SCRIPT_DIR) / "mock_state"

_stop_feed = threading.Event()


# -- sandbox ---------------------------------------------------------------------------------


def _install_sandbox(reset: bool) -> Path:
    """Point `server`'s catalog and session-record paths at throwaway copies under `mock_state/`.

    Patched on the `server` module rather than passed in, because `_register_building` and
    `_record_building_calibration` read these as module globals at call time -- which is what makes
    the redirection total: there is no path left by which a mock session can touch the real files.
    """
    SANDBOX_DIR.mkdir(exist_ok=True)
    runtime_copy = SANDBOX_DIR / "runtime-physical-building-catalog.json"
    working_copy = SANDBOX_DIR / "working-physical-building-catalog.json"
    sessions_copy = SANDBOX_DIR / "calibration_sessions.sqlite3"

    if reset:
        for path in (runtime_copy, working_copy, sessions_copy):
            path.unlink(missing_ok=True)

    if not runtime_copy.exists():
        shutil.copyfile(server.PHYSICAL_BUILDING_CATALOG_PATH, runtime_copy)
    if not working_copy.exists():
        shutil.copyfile(server.WORKING_BUILDING_CATALOG_PATH, working_copy)

    server.PHYSICAL_BUILDING_CATALOG_PATH = str(runtime_copy)
    server.WORKING_BUILDING_CATALOG_PATH = str(working_copy)
    server.SESSION_STORE_PATH = str(sessions_copy)
    server.session_store = SessionStore(str(sessions_copy))

    # Re-read the catalog the server booted from, so the in-memory copy is the sandbox's and not
    # the real file's -- otherwise a registration would merge onto one catalog and save to another.
    server.physical_building_catalog = load_catalog(runtime_copy)
    server.physical_buildings_by_marker = marker_index(server.physical_building_catalog)
    return SANDBOX_DIR


# -- the synthetic detection thread ----------------------------------------------------------


def _feed_worker(table: MockTable, interval: float) -> None:
    """`server._detection_worker`'s replacement: publish a synthetic snapshot every `interval`.

    Same queue, same shape, same "drop the stale one rather than let it pile up" rule -- that rule
    is not incidental, it is what keeps a paused consumer from making the mock's table appear to
    lag behind the CLI by however long the pause was.
    """
    while not _stop_feed.is_set():
        snapshot = table.snapshot()
        if server.tracking_queue.full():
            try:
                server.tracking_queue.get_nowait()
            except queue_module.Empty:
                pass
        try:
            server.tracking_queue.put_nowait(snapshot)
        except queue_module.Full:
            pass
        _stop_feed.wait(interval)


# -- console ---------------------------------------------------------------------------------


#: Whether `server`'s per-snapshot dump is echoed. A dict rather than a bare global so the CLI
#: thread's `v` command and the filter running on the event-loop thread share one cell.
_console_state = {"verbose": False}


def _install_quiet_console(verbose: bool) -> None:
    """Keep `server`'s 200 ms GeoJSON dump from burying the CLI.

    Assigning `print` as a module global on `server` shadows the builtin for that module only, so
    the server's own logging decisions are left alone and every other module still prints normally.
    Registration banners and calibration verdicts -- the lines a developer is actually waiting for
    -- pass through; only the per-snapshot feed dumps are dropped, and `--verbose` (or `v` at the
    prompt) puts them back.
    """
    noisy_prefixes = ("Sending to web client:", "Sending to Unity client:")

    def filtered_print(*args, **kwargs):
        if not _console_state["verbose"] and args:
            first = str(args[0])
            if first.startswith(noisy_prefixes):
                return
        builtins.print(*args, **kwargs)

    _console_state["verbose"] = verbose
    server.print = filtered_print


HELP = """
commands (a block is a building id like g11, or a marker id like 18)

  <enter> / l         show the table
  sel <block>         choose the block the one-key commands act on
  4 6 8 2             nudge the chosen block west / east / north / south
  + -                 turn the chosen block by +/- 15 degrees
  0                   put the chosen block back at its catalog heading

  m <block> <du> <dv> move a block by a fraction of the table (e.g. m g11 0.1 -0.05)
  p <block> <u> <v>   place a block at an absolute table fraction (0..1)
  r <block> <deg>     turn a block by <deg>
  a <block>           align a block to its catalog reference heading
  off <block>         take a block off the table (it stops being detected)
  on <block>          put it back

  s                   scatter every block to a random place and heading
  x                   reset the table to its starting layout
  still jitter drift  motion mode: frozen / detection noise / blocks wander and spin
  u <id> [u v]        add an UNCLAIMED marker (what registration is about)
  nu <id>             remove an unclaimed marker
  c                   toggle the projected map-calibration markers on/off
  v                   toggle the server's per-snapshot GeoJSON dump
  ?                   this help
  q                   quit
"""


def _run_cli(table: MockTable, loop: asyncio.AbstractEventLoop) -> None:
    """The operator's hands. Runs on its own thread; every mutator it calls takes the table lock."""
    chosen = DEFAULT_LAYOUT[1][0]  # G11: the one building with a calibration block in the catalog
    nudge = 0.05
    turn = 15.0

    builtins.print(HELP)
    builtins.print(table.describe())

    while True:
        try:
            raw = input(f"\n[{chosen}] table> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not raw:
            builtins.print(table.describe())
            continue

        parts = raw.split()
        command, arguments = parts[0].lower(), parts[1:]
        try:
            if command in ("q", "quit", "exit"):
                break
            elif command in ("?", "h", "help"):
                builtins.print(HELP)
            elif command in ("l", "ls", "list"):
                builtins.print(table.describe())
            elif command == "sel":
                chosen = table.block(arguments[0]).building_id
            elif command == "4":
                _report(table.move(chosen, -nudge, 0.0))
            elif command == "6":
                _report(table.move(chosen, nudge, 0.0))
            elif command == "8":
                _report(table.move(chosen, 0.0, -nudge))
            elif command == "2":
                _report(table.move(chosen, 0.0, nudge))
            elif command == "+":
                _report(table.turn(chosen, turn))
            elif command == "-":
                _report(table.turn(chosen, -turn))
            elif command == "0":
                _report(table.align(chosen))
            elif command == "m":
                _report(table.move(arguments[0], float(arguments[1]), float(arguments[2])))
            elif command == "p":
                _report(table.place(arguments[0], float(arguments[1]), float(arguments[2])))
            elif command == "r":
                _report(table.turn(arguments[0], float(arguments[1])))
            elif command == "a":
                _report(table.align(arguments[0]))
            elif command == "off":
                _report(table.set_on_table(arguments[0], False))
            elif command == "on":
                _report(table.set_on_table(arguments[0], True))
            elif command == "s":
                table.scatter()
                builtins.print(table.describe())
            elif command == "x":
                table.reset()
                builtins.print(table.describe())
            elif command in ("still", "jitter", "drift"):
                table.set_motion(command)
                builtins.print(f"motion: {command}")
            elif command == "u":
                u = float(arguments[1]) if len(arguments) > 2 else 0.5
                v = float(arguments[2]) if len(arguments) > 2 else 0.5
                _report(table.add_unclaimed(int(arguments[0]), u, v))
            elif command == "nu":
                table.remove_unclaimed(int(arguments[0]))
                builtins.print(f"removed unclaimed marker {arguments[0]}")
            elif command == "c":
                visible = not table.calibration_markers_visible
                table.set_calibration_markers_visible(visible)
                builtins.print(f"map-calibration markers: {'on' if visible else 'OFF'}")
            elif command == "v":
                _console_state["verbose"] = not _console_state["verbose"]
                builtins.print(f"snapshot dump: {'on' if _console_state['verbose'] else 'off'}")
            else:
                builtins.print(f"unknown command {command!r} -- '?' for help")
        except (IndexError, ValueError) as exc:
            builtins.print(f"bad arguments for {command!r}: {exc} -- '?' for help")
        except KeyError as exc:
            builtins.print(exc.args[0] if exc.args else str(exc))

    builtins.print("\nstopping mock server...")
    _stop_feed.set()
    loop.call_soon_threadsafe(loop.stop)


def _report(block) -> None:
    builtins.print(
        f"{block.building_id} (marker {block.marker_id}) -> u={block.u:.3f} v={block.v:.3f} "
        f"heading={block.heading:.1f} ({block.heading_from_aligned:+.1f} from its reference)"
    )


# -- entry point -----------------------------------------------------------------------------


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the COUP tracking server against a simulated table (no cameras).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=HELP,
    )
    parser.add_argument("--host", default=server.WEB_WS_HOST, help="websocket bind host")
    parser.add_argument("--port", type=int, default=server.WEB_WS_PORT, help="websocket port")
    parser.add_argument(
        "--motion",
        choices=["still", "jitter", "drift"],
        default="jitter",
        help="still: frozen, exactly repeatable. jitter: detection noise only (default). "
        "drift: blocks wander and spin on their own.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=SNAPSHOT_INTERVAL_SECONDS,
        help=f"seconds between snapshots (default {SNAPSHOT_INTERVAL_SECONDS}, the rig's cadence)",
    )
    parser.add_argument("--seed", type=int, default=7, help="noise seed, for repeatable runs")
    parser.add_argument(
        "--no-sandbox",
        action="store_true",
        help="write registrations and calibrations to the REAL catalogs instead of mock_state/",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="throw away mock_state/ and start from a fresh copy of the real catalogs",
    )
    parser.add_argument("--no-cli", action="store_true", help="run headless (for scripts and CI)")
    parser.add_argument(
        "--verbose", action="store_true", help="print every snapshot the server sends"
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)

    # `server.main` does this for the real launch; the mock never calls it, and the catalog helpers
    # still resolve some paths relatively.
    os.chdir(server._SCRIPT_DIR)

    if args.no_sandbox:
        builtins.print("!! --no-sandbox: registrations will overwrite the REAL catalogs")
    else:
        sandbox = _install_sandbox(args.reset)
        builtins.print(f"sandboxed catalogs and calibration record in {sandbox}")

    server.WEB_WS_HOST, server.WEB_WS_PORT = args.host, args.port
    _install_quiet_console(args.verbose)

    table = MockTable(server.physical_building_catalog, motion=args.motion, seed=args.seed)

    builtins.print(
        f"mock table: {len(table.buildings)} blocks "
        f"({', '.join(b.building_id for b in table.buildings.values())}), "
        f"{len(server.physical_building_catalog['buildings'])} catalog entries, "
        f"motion={args.motion}"
    )

    threading.Thread(
        target=_feed_worker, args=(table, args.interval), name="mock-feed", daemon=True
    ).start()

    loop = server.loop
    asyncio.set_event_loop(loop)

    if not args.no_cli:
        threading.Thread(target=_run_cli, args=(table, loop), name="mock-cli", daemon=True).start()

    try:
        loop.run_until_complete(server.run_web_server())
    except KeyboardInterrupt:
        pass
    except RuntimeError as exc:
        # `run_web_server` awaits a Future that never resolves, so the only way out is the CLI's
        # `loop.stop()`. That makes `run_until_complete` raise; a quit typed at the prompt is not
        # an error, and anything else still surfaces.
        if "Event loop stopped before Future completed" not in str(exc):
            raise
    finally:
        _stop_feed.set()
        builtins.print("mock server stopped")


if __name__ == "__main__":
    main()
