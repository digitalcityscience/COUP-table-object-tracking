# Mock table server

Develop the frontend against the tracking backend on your laptop, with no cameras, no projector
and no tangible table.

```powershell
.\run-mock-server.ps1          # Windows
```
```bash
./run-mock-server.sh           # macOS / Linux
```

Both wrap `uv run` exactly as `run-server.ps1` does, so they need no virtualenv of your own. If you
already have the dependencies installed, `python mock_server.py` is the same thing without the
wrapper — the launchers only translate flags.

| flag | |
|---|---|
| `-Motion` / `--motion` `still\|jitter\|drift` | how much the table moves on its own |
| `-Port` / `--port` | websocket port (default 8053, the rig's own) |
| `-Reset` / `--reset` | start from a fresh copy of the real catalogs |
| `-NoSandbox` / `--no-sandbox` | write registrations to the REAL catalogs |
| `-ShowFeed` / `--verbose` | echo every snapshot the server sends |
| `-NoCli` / `--no-cli` | headless, for scripts and CI |

It listens on `ws://0.0.0.0:8053` — the same host and port `server.py` uses — and speaks the same
protocol. TOSCA-2 needs no changes and no flags to talk to it.

Stop it with `quit` at the prompt, or Ctrl+C. Both leave immediately: Ctrl+C is handled explicitly
(`_install_signal_handlers`) rather than left to unwind as a `KeyboardInterrupt`, because the CLI
thread sits inside a blocking console read that the interrupt does not cancel, and interpreter
finalisation then waits on it. Leaving abruptly is safe — the session store commits per write and
the catalogs are written synchronously, so nothing is buffered.

---

## The one idea

**It is not a reimplementation of the server.** `mock_server.py` imports `server` and runs it
unmodified. It replaces exactly one thing: the detection thread that turns camera frames into
marker snapshots.

```
      REAL RIG                                MOCK

  RealSense cameras                      mock_table.MockTable
        |                                        |
  stitch + ArUco detect                   synthetic snapshot
        |                                        |
        +----------> server.tracking_queue <-----+
                             |
                     server.handle_web_client     <- identical from here down
                     homography / catalog / GeoJSON
                             |
                        websocket :8053
```

Everything below `tracking_queue` — the `map_calibration` handshake, `session_state`, the building
GeoJSON, `markers_on_table`, `scan_progress`, `register_building`, `building_calibration`, the
session store, the calibration log — is the production code path. So:

- a frontend that works against the mock works against the rig;
- a protocol change in `server.py` reaches the mock for free, with nothing to keep in sync;
- the only thing that can be fiction is *"a camera saw marker 12 at pixel (x, y), turned r
  degrees"* — which is exactly where the real hardware sits.

That boundary is the reason to prefer this over a hand-written fake websocket server. A fake would
drift from `server.py` the first time a field was added, and a frontend developed against it would
be developed against fiction.

---

## What is on the mock table

The three registered blocks from the runtime catalog, laid out in different regions of the table so
the admin panel's table diagram is meaningful:

| block | marker | starts at (u, v) |
|-------|--------|------------------|
| G07   | 12     | 0.30, 0.35       |
| G11   | 18     | 0.55, 0.62       |
| G17   | 24     | 0.78, 0.33       |

plus the ten map-calibration markers (200–203, 206, 207, 209–212) the Table window projects, which
is what lets the frontend run its **real** calibration handshake against the mock rather than a
short-circuited one.

Every block starts at the heading its catalog `marker_reference_rotations` entry was measured at,
so **the mock's rest state is a correctly aligned table**: any rotation you see on the map is one
you asked for. That makes a frontend rotation bug unambiguous, which a simulator that started
blocks at arbitrary angles could never be.

### Pixel space

`server` never learns the stitched image's size — it only ever sees marker pixels — so the mock
needs a *plausible* pixel frame, not a correct one. It uses the frame the reference rig actually
reported on 2026-08-31: a 1600×800 image in which the four calibration corners read
`(288,658) (1347,663) (298,153) (1350,150)`. Two properties of that reading are reproduced
deliberately:

- the calibration quad spans only the middle two-thirds of the image, so a block placed outside it
  exercises the same extrapolation the real table does; and
- **pixel `y` runs opposite to north**, because the cameras look down at the table. A mirrored mock
  is the worst failure this module could have: `cv2.findHomography` still fits its ten points
  exactly, nothing reports an error, and every building is simply reflected across the table.
  `test/test_mock_table.py::test_the_calibration_quad_is_not_mirrored` pins it.

---

## The CLI

While the server runs, a prompt lets you stand in for the hands that would be moving blocks on the
real table. **Four verbs. Typed on their own, each one picks a block at random** — that is the
normal way to use it, because the usual request at the table is not "nudge G11 five centimetres
west", it is "make something change so I can watch the frontend react".

```
  move        move a random block somewhere new
  turn        turn a random block by a random angle
  add         put a block back on the table
  remove      take a random block off the table
```

Say which block, and how much, when you need to be exact — same four verbs:

```
  move g11            move G11 somewhere random
  move g11 0.6 0.4    put G11 exactly there  (u, v run 0..1 across the table)
  move all            move every block at once
  turn g11 90         turn G11 by 90 degrees
  add 42              put marker 42 on the table, claimed by no building
  remove g11          take G11 off
```

The rest:

```
  list                show the table   (a blank line does the same)
  align g11           put G11 back at its catalog heading
  reset               back to the starting layout
  motion still|jitter|drift
  calib               hide or show the projected map-calibration markers
  verbose             show or hide the server's per-snapshot dump
  help / quit
```

A block is named by building id (`g11`) or marker id (`18`).

Two behaviours worth knowing:

- **`add` always does something.** With every block already on the table there is nothing to
  restore, so it invents an unclaimed marker instead and says so. "Nothing to add" would be
  indistinguishable, at the prompt, from the command silently failing — and an unclaimed marker is
  a useful thing to be handed anyway (see registration, below).
- **`remove` on a catalogued block only takes it off the table**, leaving the catalog alone —
  which is what happens physically. An unclaimed marker is deleted outright, since nothing but the
  mock knows it ever existed.

### Motion modes

| mode | what it does | what it is for |
|------|--------------|----------------|
| `still` | frozen, no noise at all | exactly repeatable runs; bisecting a frontend bug |
| `jitter` *(default)* | positions fixed, ±0.8 px / ±0.4° of detection noise per frame | the realistic resting table — the frontend's stability gates have something to ride out |
| `drift` | every block wanders its own slow ellipse and spins | continuous movement, without a hand on the table |

`drift` gives each block its *own* ellipse and spin rate on purpose. Moving the whole table as one
rigid body would be indistinguishable from a homography change, and so would test nothing.

---

## Things worth reproducing with it

- **Calibration.** `calib` hides the projected markers: the frontend should report an incomplete
  calibration, not calibrate from a stale reading.
- **A block leaving the table.** `remove g11` — the building must disappear from the feed, and
  `building_calibration` against it must be refused once the reading goes stale
  (`TABLE_POSITION_MAX_AGE_SECONDS`, 3 s).
- **Registration.** `add` on a full table hands you an unclaimed marker — the block registration is
  actually about, and the one id the building feed never carries. `move 30 <u> <v>` walks it onto
  the projected outline, and `scan_progress` should unlock Register only once it is there.
- **Re-registration.** `add 18` re-offers an already-claimed marker, the re-glued-block flow.
- **Rotation.** `turn g11 30`, repeatedly. Because the block starts aligned, the map heading and
  the number the CLI prints must track each other exactly.

---

## Sandboxing

Registration and building calibration **write catalogs**. Refusing to run them would cut out half
of what there is to develop against, so by default the mock copies both catalogs and the
calibration record into `mock_state/` (gitignored) and points `server` at the copies. The real
files are never touched.

- `--reset` throws the sandbox away and starts from a fresh copy.
- `--no-sandbox` writes to the real catalogs, for when you actually mean to.

The redirection is done by patching module globals on `server`, because
`_register_building` and `_record_building_calibration` read those paths at call time — which is
what makes it total: there is no path left by which a mock session can reach the real files.

---

## Two ordering traps in the protocol

Both are `server.py`'s, not the mock's, but the mock is where you will meet them:

- `map_calibration`'s `lat_lon_position` is **`[lat, lng]`** (`pixel_to_utm.latlon_to_utm`).
- `register_building`'s and `scan_target`'s `target` is **`[lng, lat]`**.

Swapping either produces a homography that fits its own points perfectly and puts every building
in the wrong place, with no error anywhere.

Also: **the calibration is process-global and survives a disconnect.** A freshly connected client
that expects raw pre-calibration snapshots has to send `clear_calibration` first, or it will
receive GeoJSON from the previous client's session. Real behaviour, and easy to lose an hour to.

---

## Files

| file | what it holds |
|------|---------------|
| `mock_server.py` | wiring: sandbox, feed thread, CLI, launch |
| `mock_table.py` | the simulated table — the *only* fiction in the stack |
| `test/test_mock_table.py` | the invariants that make the mock worth trusting |
| `run-mock-server.ps1` | launcher, alongside `run-server.ps1` |
| `run-mock-server.sh` | the same launcher for macOS and Linux |

`mock_camera.py` is a different, older thing: it replays MP4 files through the *real* detection
pipeline. Use it to test detection itself; use this to test everything downstream of it.
