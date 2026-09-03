"""The calibration record for the tangible table: one SQLite file, two tables.

Written only here, only by Python. The browser cannot write SQLite and the frontend stores no
calibration value of its own (see `BUILDING-GEOMETRY-CALIBRATION-WORKFLOW.md`, "Değişmeyecek
kararlar"), so this module is the single writer of everything an operator measures at the rig.

Two tables, because the error model has two layers and they must not be mixed:

- `sessions` holds what is *global* to one calibration: the AOI, the homography solved from it,
  and the ground scale and `global_k` derived from that homography. Everything a building row
  means is relative to its session's row.
- `session_buildings` holds the *residual* left over per building once that global mapping is
  applied -- and, crucially, `table_x_px`/`table_y_px`: where on the table the measurement was
  taken. Without that column pair a position-dependent homography error is indistinguishable
  from a per-building sticking error, which is exactly the question step 6.3 exists to settle.
  Fitting per-building constants first would let each building's offset silently absorb the
  homography error in its corner of the table, and the constant would then be wrong the moment
  the block is moved.

A building's identity within a session is `(building_id, marker_id, table_x_px, table_y_px)`.
Re-saving a building the operator is still nudging replaces its row; the same block measured at
a different place on the table is a new measurement, which is what makes step 6.3's five-position
sweep recordable at all.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence
import hashlib
import json
import sqlite3

#: Decimal places the AOI corners are rounded to before hashing. ~1e-8 degrees is about a
#: millimetre on the ground, far below any real AOI change, so this absorbs the float formatting
#: noise a re-serialised handshake carries without ever merging two AOIs an operator meant to
#: keep apart.
_AOI_HASH_PRECISION = 8

#: The grain a measurement's table position is filed at, in table pixels. One pixel is one
#: millimetre at 10 px/cm, which is already the finest movement the pipeline can represent, so
#: rounding to it loses nothing real.
#:
#: Without it the primary key below never actually collapses a repeat measurement: `table_x_px`
#: arrives from an ArUco centre that jitters by fractions of a pixel between frames, so two saves
#: of a block the operator never moved land on two different REALs. An operator dialling in one
#: offset over four saves would leave four rows -- three of them superseded, all four weighted
#: equally by whatever fits the global correction.
_TABLE_POSITION_GRAIN_PX = 1.0

#: How much of the AOI digest goes into the session id. Eight hex characters is plenty to keep
#: a day's AOIs apart and short enough to read out of a log line or a filename.
_AOI_HASH_LENGTH = 8

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id           TEXT PRIMARY KEY,
    created_at   TEXT NOT NULL,
    aoi_corners  TEXT NOT NULL,
    ground_scale REAL NOT NULL,
    homography   TEXT NOT NULL,
    global_k     REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS session_buildings (
    session_id          TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    building_id         TEXT NOT NULL,
    marker_id           INTEGER NOT NULL,
    rotation_offset_deg REAL NOT NULL,
    offset_east_m       REAL NOT NULL,
    offset_north_m      REAL NOT NULL,
    scale_residual      REAL NOT NULL,
    table_x_px          REAL NOT NULL,
    table_y_px          REAL NOT NULL,
    recorded_at         TEXT NOT NULL,
    PRIMARY KEY (session_id, building_id, marker_id, table_x_px, table_y_px)
);
"""


def _quantised_table_position(value: float) -> float:
    """A table pixel position rounded to `_TABLE_POSITION_GRAIN_PX` — see that constant."""
    return round(float(value) / _TABLE_POSITION_GRAIN_PX) * _TABLE_POSITION_GRAIN_PX


def _aoi_digest(aoi_corners: Sequence[Sequence[float]]) -> str:
    """A stable digest of the *area* a set of geographic points covers.

    Deliberately the rounded bounding box rather than the points themselves. The points handed in
    are the calibration handshake's own `lat_lon_position`s, and there are now between four and
    nine of them depending on which grid markers the cameras managed to decode this round -- so
    hashing the list would make the digest a signature of one detection round rather than of the
    AOI, and the same operator on the same AOI would fork a new session every time a marker
    blinked. The extremes are the four corner markers either way, so the bounding box is the same
    whether four points arrive or nine.

    Rounded to `_AOI_HASH_PRECISION` (~1 mm on the ground, far below any real AOI change) so
    float formatting noise in a re-serialised handshake cannot fork a session either.
    """
    points = [[float(value) for value in corner] for corner in aoi_corners]
    if not points:
        raise ValueError("cannot identify an AOI from no points")
    extent = [
        round(min(point[0] for point in points), _AOI_HASH_PRECISION),
        round(min(point[1] for point in points), _AOI_HASH_PRECISION),
        round(max(point[0] for point in points), _AOI_HASH_PRECISION),
        round(max(point[1] for point in points), _AOI_HASH_PRECISION),
    ]
    serialized = json.dumps(extent, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:_AOI_HASH_LENGTH]


def session_id_for(aoi_corners: Sequence[Sequence[float]], at: str | datetime | None = None) -> str:
    """`<date>-<aoi digest>`, e.g. `20260903-1f3a9c02`.

    Generated here and never sent by the frontend, which has no session id at all.

    The stamp is the *date*, not the instant. A calibration sitting is a day at the rig, and the
    property that matters is the one the workflow states outright: re-entering calibration on the
    same AOI lands on the same session rather than forking one. `server.py` calls this on every
    accepted `map_calibration`, and an operator recalibrates repeatedly while getting the markers
    read -- with second resolution each of those would open a new session and strand the previous
    one's building rows against a homography that had already been superseded. A different day on
    the same AOI is a genuinely different sitting (the projector has been moved, the room relit)
    and does get its own session.

    `at` is a wall-clock timestamp (ISO 8601, or a `datetime`); it defaults to now.
    """
    moment = datetime.now() if at is None else (at if isinstance(at, datetime) else datetime.fromisoformat(at))
    return f"{moment:%Y%m%d}-{_aoi_digest(aoi_corners)}"


@dataclass(frozen=True)
class BuildingCalibration:
    """One building's residual pose, measured at one place on the table.

    Every field is the *leftover* after the session's global mapping has been applied, which is
    why the offsets are small: `offset_east_m`/`offset_north_m` are stored in the building's own
    local frame (they turn with the block, not with the compass), `rotation_offset_deg` is added
    on top of `detected_rotation - marker_reference_rotation`, and `scale_residual` multiplies
    the session's `global_k` for this one building.
    """

    building_id: str
    marker_id: int
    rotation_offset_deg: float
    offset_east_m: float
    offset_north_m: float
    scale_residual: float
    table_x_px: float
    table_y_px: float


class SessionStore:
    """The SQLite calibration record. Creates its file and schema on first write."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self._schema_ready = False

    def _connect(self) -> sqlite3.Connection:
        if not self._schema_ready:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        if not self._schema_ready:
            with connection:
                connection.executescript(_SCHEMA)
            self._schema_ready = True
        return connection

    def begin_session(
        self,
        *,
        aoi_corners: Sequence[Sequence[float]],
        ground_scale: float,
        homography: Sequence[Sequence[float]],
        global_k: float,
        at: str | datetime | None = None,
    ) -> str:
        """Record the accepted calibration and return its session id.

        Called the moment a `map_calibration` message is accepted, so the derived `ground_scale`
        and `global_k` are on disk before any building row can reference them. Re-accepting a
        calibration for the same session id overwrites those derived values rather than adding a
        row: the building rows in that session were measured against the newest homography, not
        the one it replaced.
        """
        session_id = session_id_for(aoi_corners, at=at)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO sessions (id, created_at, aoi_corners, ground_scale, homography, global_k)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    aoi_corners  = excluded.aoi_corners,
                    ground_scale = excluded.ground_scale,
                    homography   = excluded.homography,
                    global_k     = excluded.global_k
                """,
                (
                    session_id,
                    datetime.now().isoformat(timespec="seconds"),
                    json.dumps([[float(value) for value in corner] for corner in aoi_corners]),
                    float(ground_scale),
                    json.dumps([[float(value) for value in row] for row in homography]),
                    float(global_k),
                ),
            )
        return session_id

    def record_building(self, session_id: str, calibration: BuildingCalibration) -> None:
        """Store one building's residual pose. Refuses a session that was never begun.

        A building row with no session row behind it cannot be interpreted at all -- there is no
        homography and no `global_k` to read it relative to -- so this fails loudly rather than
        keeping an orphan.
        """
        with self._connect() as connection:
            if connection.execute("SELECT 1 FROM sessions WHERE id = ?", (session_id,)).fetchone() is None:
                raise ValueError(f"unknown session {session_id!r}; call begin_session first")
            connection.execute(
                """
                INSERT INTO session_buildings (
                    session_id, building_id, marker_id, rotation_offset_deg,
                    offset_east_m, offset_north_m, scale_residual,
                    table_x_px, table_y_px, recorded_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id, building_id, marker_id, table_x_px, table_y_px)
                DO UPDATE SET
                    rotation_offset_deg = excluded.rotation_offset_deg,
                    offset_east_m       = excluded.offset_east_m,
                    offset_north_m      = excluded.offset_north_m,
                    scale_residual      = excluded.scale_residual,
                    recorded_at         = excluded.recorded_at
                """,
                (
                    session_id,
                    calibration.building_id,
                    int(calibration.marker_id),
                    float(calibration.rotation_offset_deg),
                    float(calibration.offset_east_m),
                    float(calibration.offset_north_m),
                    float(calibration.scale_residual),
                    _quantised_table_position(calibration.table_x_px),
                    _quantised_table_position(calibration.table_y_px),
                    datetime.now().isoformat(timespec="seconds"),
                ),
            )

    def session(self, session_id: str) -> dict[str, Any] | None:
        """One session row, with `aoi_corners`/`homography` decoded back out of JSON."""
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        if row is None:
            return None
        record = dict(row)
        record["aoi_corners"] = json.loads(record["aoi_corners"])
        record["homography"] = json.loads(record["homography"])
        return record

    def session_count(self) -> int:
        with self._connect() as connection:
            (count,) = connection.execute("SELECT COUNT(*) FROM sessions").fetchone()
        return int(count)

    def session_buildings(self, session_id: str) -> list[dict[str, Any]]:
        """Every building measurement in a session, ordered so the list is diffable and readable."""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM session_buildings
                WHERE session_id = ?
                ORDER BY building_id, marker_id, table_x_px, table_y_px
                """,
                (session_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def all_building_measurements(self) -> list[dict[str, Any]]:
        """Every building row across every session, each carrying its session's global values.

        The input to the "sonrası" step: fit the homography correction from these first, then
        derive the per-building constants from what is left over. Joined here rather than in the
        caller so nothing can read a residual without the global mapping it is a residual of.
        """
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT b.*, s.ground_scale, s.global_k, s.aoi_corners
                FROM session_buildings b
                JOIN sessions s ON s.id = b.session_id
                ORDER BY b.session_id, b.building_id, b.marker_id
                """
            ).fetchall()
        measurements = []
        for row in rows:
            record = dict(row)
            record["aoi_corners"] = json.loads(record["aoi_corners"])
            measurements.append(record)
        return measurements


def calibration_rows_as_dicts(calibrations: Iterable[BuildingCalibration]) -> list[dict[str, Any]]:
    """`BuildingCalibration`s as plain dicts, for logging and for the catalog writer."""
    return [asdict(calibration) for calibration in calibrations]
