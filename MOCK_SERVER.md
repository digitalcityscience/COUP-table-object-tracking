# Mock table server

Develop the frontend against the tracking backend on your laptop, with no cameras, no projector
and no tangible table.

```powershell
.\run-mock-server.ps1          # or: python mock_server.py
```

It listens on `ws://0.0.0.0:8053` — the same host and port `server.py` uses — and speaks the same
protocol. TOSCA-2 needs no changes and no flags to talk to it.

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

While the server runs, a prompt lets you be the operator's hands. `?` prints this list.

```
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
  still jitter drift  motion mode
  u <id> [u v]        add an UNCLAIMED marker (what registration is about)
  nu <id>             remove an unclaimed marker
  c                   toggle the projected map-calibration markers on/off
  v                   toggle the server's per-snapshot GeoJSON dump
  q                   quit
```

A block is named by building id (`g11`) or marker id (`18`).

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

- **Calibration.** `c` hides the projected markers: the frontend should report an incomplete
  calibration, not calibrate from a stale reading.
- **A block leaving the table.** `off g11` — the building must disappear from the feed, and
  `building_calibration` against it must be refused once the reading goes stale
  (`TABLE_POSITION_MAX_AGE_SECONDS`, 3 s).
- **Registration.** `u 42` puts an unclaimed marker on the table — the block registration is
  actually about, and the one id the building feed never carries. `p 42 <u> <v>` walks it onto the
  projected outline, and `scan_progress` should unlock Register only once it is there.
- **Re-registration.** `u 18` re-offers an already-claimed marker, the re-glued-block flow.
- **Rotation.** `sel g11`, then `+` repeatedly. Because the block starts aligned, the map heading
  and the number the CLI prints must track each other exactly.

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

`mock_camera.py` is a different, older thing: it replays MP4 files through the *real* detection
pipeline. Use it to test detection itself; use this to test everything downstream of it.
