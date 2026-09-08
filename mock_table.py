"""A synthetic tangible table: the marker snapshots the cameras would produce, without cameras.

This module owns *only* the thing the real rig owns and the mock cannot borrow -- the contents of
one `Markers.toDict()` snapshot. Everything downstream of that snapshot (homography, catalog
geometry, GeoJSON, registration, the websocket protocol itself) is the real `server.py` running
unmodified; see `mock_server.py`.

That split is the whole design. A mock that reimplemented the message shapes would drift from the
server the first time a field was added, and a frontend developed against it would be developed
against fiction. Here the only fiction is "a camera saw marker 12 at pixel (x, y) turned r
degrees", which is exactly the boundary the real hardware sits on.

Pixel space
-----------
`server` never learns the stitched image's size -- it only ever sees marker pixels -- so this
module needs a plausible pixel frame rather than a correct one. It uses the frame the reference
rig actually reported on 2026-08-31: a 1600x800 stitched image in which the four map-calibration
corners 200-203 read `(288,658) (1347,663) (298,153) (1350,150)` (`collabTracking.ts`). Two
properties of that reading matter and both are reproduced here:

* the calibration quad spans only the middle two thirds of the image, so a mock building placed
  outside it exercises the same extrapolation the real table does; and
* pixel `y` runs *opposite* to the AOI's north, because the cameras look at the table from above
  and the frontend's `row: "min"` band is its northern edge. A mock that got this backwards would
  produce a mirrored homography that still fits its four points perfectly -- the worst kind of
  wrong, since nothing would report an error and every building would simply be in the wrong place.

Angles
------
`rotation` is `detection.normalizeCorners`'s raw table-frame angle in degrees, the same number the
real detector emits. A building whose heading equals its catalog
`marker_reference_rotations` entry draws at its true real-world heading (`building_feature`
subtracts the two), so that is what `MockTable` starts every building at: the mock's rest state is
a correctly aligned table, and any rotation seen on the map is one this module was told to make.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple

from calibration_contract import MAP_CALIBRATION_MARKER_CORNERS

#: The stitched image the reference rig produces, in pixels. Only the *ratio* of marker positions
#: to this frame is observable downstream, but keeping the real number makes the mock's debug
#: output directly comparable with a real session's log.
TABLE_WIDTH_PX = 1600
TABLE_HEIGHT_PX = 800

#: The camera id every mock marker is attributed to. The real server stitches several streams into
#: one frame and `map_detected_markers` stamps them all with the same id ("000"); nothing
#: downstream reads it, but the snapshot's shape is `[x, y, rotation, camera_id]` and a mock that
#: dropped the fourth element would be a different shape from the real one.
CAMERA_ID = "000"

#: `collabCalibration.calibrationMarkerInsetFractions` for the reference 160x80 cm table: the
#: calibration markers sit 5% of the width in from the left/right edges and the same *physical*
#: distance -- 10% of the height -- in from the top/bottom.
INSET_U = 0.05
INSET_V = 0.10

#: `collabCalibration.seamClearanceRatio`: how far either side of the table's horizontal centre a
#: `midLeft`/`midRight` marker sits, as a fraction of the inset rectangle's width. The centre line
#: itself is the camera-stitch seam and cannot be decoded on the two-desk rig.
SEAM_CLEARANCE = 0.10

#: The ten map-calibration markers, as `(id, column_band, row_band)`, mirroring
#: `collabTracking.ts::MAP_CALIBRATION_MARKERS`. The frontend projects these onto the table and
#: reads them back out of the raw pre-calibration snapshot to build its `map_calibration` message,
#: so the mock has to emit all ten or the frontend can never calibrate against it.
#:
#: Duplicated from the frontend rather than imported because there is nothing to import from: this
#: is a Python process and the list lives in TypeScript. `calibration_contract.MAP_CALIBRATION_
#: MARKER_IDS` is the half Python owns, and the assertion below keeps the two from diverging.
CALIBRATION_MARKER_BANDS: Tuple[Tuple[int, str, str], ...] = (
    (200, "min", "min"),
    (201, "max", "min"),
    (202, "min", "max"),
    (203, "max", "max"),
    (206, "min", "mid"),
    (207, "max", "mid"),
    (209, "midLeft", "min"),
    (210, "midRight", "min"),
    (211, "midLeft", "max"),
    (212, "midRight", "max"),
)


def _band_fraction(band: str, low: float, high: float) -> float:
    """`collabTracking.ts::mapCalibrationMarkerUv`'s band resolution, one axis at a time."""
    mid = (low + high) / 2.0
    if band == "min":
        return low
    if band == "max":
        return high
    if band == "mid":
        return mid
    if band == "midLeft":
        return mid - SEAM_CLEARANCE * (high - low)
    if band == "midRight":
        return mid + SEAM_CLEARANCE * (high - low)
    raise ValueError(f"unknown calibration marker band {band!r}")


def uv_to_pixel(u: float, v: float) -> Tuple[float, float]:
    """Where `(u, v)` in the AOI lands in stitched-image pixels on the reference rig.

    `u` runs west->east and `v` north->south as fractions of the table, matching the frontend's
    bands. The affine below is fitted to the four corner readings quoted in this module's
    docstring, which is what gives the mock the real rig's inset quad and its inverted `y`.
    """
    # x: u = INSET_U -> 293 px, u = 1 - INSET_U -> 1348.5 px (the mean of each measured pair).
    x_at_min_u, x_at_max_u = 293.0, 1348.5
    # y: v = INSET_V (the AOI's north edge) -> 660.5 px, v = 1 - INSET_V -> 151.5 px. Decreasing,
    # deliberately: see this module's docstring.
    y_at_min_v, y_at_max_v = 660.5, 151.5

    u_span = (1.0 - INSET_U) - INSET_U
    v_span = (1.0 - INSET_V) - INSET_V
    x = x_at_min_u + (u - INSET_U) / u_span * (x_at_max_u - x_at_min_u)
    y = y_at_min_v + (v - INSET_V) / v_span * (y_at_max_v - y_at_min_v)
    return x, y


def calibration_marker_pixels() -> Dict[int, Tuple[float, float]]:
    """The ten projected calibration markers, where the cameras would read them."""
    min_u, max_u = INSET_U, 1.0 - INSET_U
    min_v, max_v = INSET_V, 1.0 - INSET_V
    return {
        marker_id: uv_to_pixel(
            _band_fraction(column, min_u, max_u), _band_fraction(row, min_v, max_v)
        )
        for marker_id, column, row in CALIBRATION_MARKER_BANDS
    }


@dataclass
class MockBuilding:
    """One physical block on the mock table.

    `u`/`v` are its position as a fraction of the table (the same frame the calibration markers
    use), not pixels, so a block can be placed and reasoned about without knowing the rig's image
    size. `heading` is the raw table-frame angle the detector would report.

    `reference_heading` is the block's catalog `marker_reference_rotations` value -- the heading at
    which this building draws at its true real-world orientation. Kept on the block so the CLI can
    say "turn G11 by 30 degrees *from aligned*", which is the question an operator actually has,
    rather than making them do the subtraction against a number like -110.5.
    """

    building_id: str
    marker_id: int
    u: float
    v: float
    reference_heading: float
    heading: float
    on_table: bool = True

    #: Per-block wander, in the `drift` motion mode: a phase and radius so each block traces its
    #: own slow ellipse instead of the whole table moving as one rigid body (which would be
    #: indistinguishable from a homography change on the frontend, and so would test nothing).
    drift_phase: float = 0.0
    drift_radius_u: float = 0.06
    drift_radius_v: float = 0.06
    spin_rate_deg_s: float = 4.0

    #: Where `drift` orbits around. Set from `u`/`v` whenever the block is placed, so a drifting
    #: block that is then moved by hand orbits its *new* home rather than snapping back.
    home_u: float = 0.0
    home_v: float = 0.0

    def place(self, u: float, v: float) -> None:
        self.u = _clamp01(u)
        self.v = _clamp01(v)
        self.home_u, self.home_v = self.u, self.v

    def align(self) -> None:
        """Put the block back at the heading its catalog reference was measured at."""
        self.heading = self.reference_heading

    @property
    def heading_from_aligned(self) -> float:
        """How far this block is turned from its catalog reference, in (-180, 180] degrees."""
        return (self.heading - self.reference_heading + 180.0) % 360.0 - 180.0


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


#: The three registered blocks, laid out across the table so no two overlap and each sits in a
#: different region of the calibration quad -- the frontend's admin panel draws a table diagram of
#: where measurements were taken, and three blocks stacked in one corner would make it useless.
#:
#: The reference headings are read from the runtime catalog at construction time rather than
#: hardcoded here, so re-registering a building on the real rig cannot leave the mock quietly
#: starting its blocks at a stale angle.
DEFAULT_LAYOUT: Tuple[Tuple[str, int, float, float], ...] = (
    ("G07", 12, 0.30, 0.35),
    ("G11", 18, 0.55, 0.62),
    ("G17", 24, 0.78, 0.33),
)


class MockTable:
    """The mock rig's whole observable state, and the snapshots it produces.

    Not thread-safe by accident: `snapshot()` runs on the feed thread and every mutator runs on the
    CLI thread, so all of them take `_lock`. The consequence of skipping it is not a crash but a
    half-moved block appearing in one snapshot, which is exactly the kind of thing a developer
    would spend an afternoon blaming the frontend for.
    """

    def __init__(
        self,
        catalog: dict,
        *,
        motion: str = "jitter",
        seed: int | None = 7,
    ) -> None:
        import threading

        self._lock = threading.Lock()
        self._random = random.Random(seed)
        self._started_at = time.monotonic()
        self.motion = motion

        #: Detection noise, in pixels and degrees, applied to every reading in every mode except
        #: `still`. Roughly what one frame of ArUco corner detection carries on the real rig; the
        #: point is that the frontend's own stability checks (and Python's confidence gate) see a
        #: signal that moves, as they will in production.
        self.pixel_noise = 0.8
        self.angle_noise = 0.4

        references = _reference_headings(catalog)
        self.buildings: Dict[int, MockBuilding] = {}
        for index, (building_id, marker_id, u, v) in enumerate(DEFAULT_LAYOUT):
            reference = references.get((building_id, marker_id), 0.0)
            block = MockBuilding(
                building_id=building_id,
                marker_id=marker_id,
                u=u,
                v=v,
                reference_heading=reference,
                heading=reference,
                drift_phase=index * 2.0 * math.pi / len(DEFAULT_LAYOUT),
                spin_rate_deg_s=3.0 + 2.0 * index,
            )
            block.place(u, v)
            self.buildings[marker_id] = block

        #: Markers with no catalog entry, keyed by id. The registration flow's entire subject is a
        #: block *no building claims* -- it is the one marker the building feed never carries -- so
        #: a mock that could not produce one could not exercise registration at all.
        self.unclaimed: Dict[int, MockBuilding] = {}

        self.calibration_markers_visible = True

    # -- snapshot ---------------------------------------------------------------------------

    def snapshot(self) -> Dict[int, List]:
        """One `Markers.toDict()`-shaped reading: `{marker_id: [x_px, y_px, rotation, camera_id]}`.

        Int keys and a trailing camera id, both matching `marker.printJSON`. `json.dumps` turns the
        keys into strings on the wire exactly as it does for the real server, so the frontend sees
        no difference.
        """
        with self._lock:
            now = time.monotonic() - self._started_at
            reading: Dict[int, List] = {}

            if self.calibration_markers_visible:
                for marker_id, (x, y) in calibration_marker_pixels().items():
                    # Calibration markers are projected, not physical: they do not move, but the
                    # camera reading them still carries noise, and the frontend's stability gate
                    # exists to ride that out. Give it something to ride out.
                    reading[marker_id] = [
                        *self._noisy_pixel(x, y),
                        0.0,
                        CAMERA_ID,
                    ]

            for block in list(self.buildings.values()) + list(self.unclaimed.values()):
                if not block.on_table:
                    continue
                u, v, heading = self._pose_at(block, now)
                x, y = uv_to_pixel(u, v)
                reading[block.marker_id] = [
                    *self._noisy_pixel(x, y),
                    self._noisy_angle(heading),
                    CAMERA_ID,
                ]
            return reading

    def _pose_at(self, block: MockBuilding, elapsed: float) -> Tuple[float, float, float]:
        """Where a block is *right now*, including whatever the motion mode is doing to it."""
        if self.motion != "drift":
            return block.u, block.v, block.heading
        # A slow ellipse per block, at a period long enough (about 40 s) that a developer watching
        # the map sees movement rather than vibration.
        angle = block.drift_phase + elapsed * (2.0 * math.pi / 40.0)
        u = _clamp01(block.home_u + block.drift_radius_u * math.cos(angle))
        v = _clamp01(block.home_v + block.drift_radius_v * math.sin(angle))
        heading = block.heading + block.spin_rate_deg_s * elapsed
        return u, v, (heading + 180.0) % 360.0 - 180.0

    def _noisy_pixel(self, x: float, y: float) -> Tuple[float, float]:
        if self.motion == "still":
            return x, y
        return (
            x + self._random.gauss(0.0, self.pixel_noise),
            y + self._random.gauss(0.0, self.pixel_noise),
        )

    def _noisy_angle(self, heading: float) -> float:
        if self.motion == "still":
            return heading
        return heading + self._random.gauss(0.0, self.angle_noise)

    # -- mutators (the CLI's whole vocabulary) ----------------------------------------------

    def block(self, key: str) -> MockBuilding:
        """The block named by a building id (`g11`) or a marker id (`18`), or raise."""
        lookup = key.strip().upper()
        with self._lock:
            everything = {**self.buildings, **self.unclaimed}
            if lookup.isdigit() and int(lookup) in everything:
                return everything[int(lookup)]
            for candidate in everything.values():
                if candidate.building_id.upper() == lookup:
                    return candidate
        known = ", ".join(sorted(b.building_id for b in self.buildings.values()))
        raise KeyError(f"no block called {key!r}; known blocks: {known}")

    def move(self, key: str, du: float, dv: float) -> MockBuilding:
        block = self.block(key)
        with self._lock:
            block.place(block.u + du, block.v + dv)
        return block

    def place(self, key: str, u: float, v: float) -> MockBuilding:
        block = self.block(key)
        with self._lock:
            block.place(u, v)
        return block

    def turn(self, key: str, degrees: float) -> MockBuilding:
        block = self.block(key)
        with self._lock:
            block.heading = (block.heading + degrees + 180.0) % 360.0 - 180.0
        return block

    def align(self, key: str) -> MockBuilding:
        block = self.block(key)
        with self._lock:
            block.align()
        return block

    def set_on_table(self, key: str, on_table: bool) -> MockBuilding:
        block = self.block(key)
        with self._lock:
            block.on_table = on_table
        return block

    def scatter(self) -> None:
        """Throw every block somewhere new inside the calibration quad, at a random heading.

        The single most useful button on the mock: it changes every observable at once, so a
        frontend that has cached a position, a heading or a feature id anywhere it should not have
        shows it immediately.
        """
        with self._lock:
            for block in self.buildings.values():
                block.place(
                    self._random.uniform(INSET_U + 0.05, 1.0 - INSET_U - 0.05),
                    self._random.uniform(INSET_V + 0.05, 1.0 - INSET_V - 0.05),
                )
                block.heading = self._random.uniform(-180.0, 180.0)

    def reset(self) -> None:
        with self._lock:
            for building_id, marker_id, u, v in DEFAULT_LAYOUT:
                block = self.buildings.get(marker_id)
                if block is not None:
                    block.place(u, v)
                    block.align()
                    block.on_table = True
            self.unclaimed.clear()
            self.calibration_markers_visible = True

    def add_unclaimed(self, marker_id: int, u: float = 0.5, v: float = 0.5) -> MockBuilding:
        """Put a block on the table that no catalog entry claims -- registration's subject.

        Deliberately allowed to name an id the catalog *does* know: re-registering a building whose
        marker was re-glued is a real flow, and refusing it here would make the mock unable to
        reproduce the bug that flow exists to fix.
        """
        with self._lock:
            block = MockBuilding(
                building_id=f"<unclaimed {marker_id}>",
                marker_id=marker_id,
                u=u,
                v=v,
                reference_heading=0.0,
                heading=0.0,
            )
            block.place(u, v)
            self.unclaimed[marker_id] = block
            return block

    def remove_unclaimed(self, marker_id: int) -> None:
        with self._lock:
            self.unclaimed.pop(marker_id, None)

    def set_motion(self, motion: str) -> None:
        if motion not in ("still", "jitter", "drift"):
            raise ValueError(f"motion must be still, jitter or drift (got {motion!r})")
        with self._lock:
            if motion == "drift" and self.motion != "drift":
                # Restart the clock so a block does not jump to wherever its ellipse would have
                # carried it while drift was off.
                self._started_at = time.monotonic()
            self.motion = motion

    def set_calibration_markers_visible(self, visible: bool) -> None:
        with self._lock:
            self.calibration_markers_visible = visible

    # -- reporting --------------------------------------------------------------------------

    def describe(self) -> str:
        with self._lock:
            now = time.monotonic() - self._started_at
            lines = [
                f"motion={self.motion}  calibration_markers="
                f"{'on' if self.calibration_markers_visible else 'OFF'}",
                "",
                f"{'block':<18}{'marker':>7}{'u':>8}{'v':>8}{'px':>16}"
                f"{'heading':>10}{'vs ref':>9}   state",
            ]
            for block in sorted(
                list(self.buildings.values()) + list(self.unclaimed.values()),
                key=lambda b: b.marker_id,
            ):
                u, v, heading = self._pose_at(block, now)
                x, y = uv_to_pixel(u, v)
                offset = (heading - block.reference_heading + 180.0) % 360.0 - 180.0
                state = "on table" if block.on_table else "OFF TABLE"
                lines.append(
                    f"{block.building_id:<18}{block.marker_id:>7}{u:>8.3f}{v:>8.3f}"
                    f"{f'({x:.0f},{y:.0f})':>16}{heading:>10.1f}{offset:>9.1f}   {state}"
                )
            return "\n".join(lines)


def _reference_headings(catalog: dict) -> Dict[Tuple[str, int], float]:
    """Every `(building_id, marker_id) -> reference rotation` the runtime catalog holds."""
    references: Dict[Tuple[str, int], float] = {}
    for entry in catalog.get("buildings", []):
        building_id = entry["building_id"]
        stored = entry.get("marker_reference_rotations") or {}
        for marker_id in entry.get("marker_ids", []):
            references[(building_id, int(marker_id))] = float(
                stored.get(str(marker_id), 0.0)
            )
    return references


def _assert_calibration_contract() -> None:
    """The mock's marker list and Python's own contract must name the same ids.

    A mismatch is silent and expensive: `marker.reduceToCalibrationMarkers` only waves ids in
    `calibration_contract.MAP_CALIBRATION_MARKER_IDS` past the confidence gate, so an id this
    module invented would simply never reach the frontend, and the frontend would report an
    un-decodable marker on a rig with no cameras.
    """
    from calibration_contract import MAP_CALIBRATION_MARKER_IDS

    mine = {marker_id for marker_id, _, _ in CALIBRATION_MARKER_BANDS}
    if mine != set(MAP_CALIBRATION_MARKER_IDS):
        raise AssertionError(
            "mock_table's calibration marker list has drifted from calibration_contract: "
            f"only in mock={sorted(mine - set(MAP_CALIBRATION_MARKER_IDS))}, "
            f"only in contract={sorted(set(MAP_CALIBRATION_MARKER_IDS) - mine)}"
        )
    corners = {marker_id for marker_id, _, _ in CALIBRATION_MARKER_BANDS[:4]}
    if corners != set(MAP_CALIBRATION_MARKER_CORNERS):
        raise AssertionError("mock_table's corner ids have drifted from calibration_contract")


_assert_calibration_contract()
