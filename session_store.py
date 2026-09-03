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


def _aoi_digest(aoi_corners: Sequence[Sequence[float]]) -> str:
    """A stable digest of an AOI's four corners, insensitive to float formatting noise."""
    rounded = [
        [round(float(value), _AOI_HASH_PRECISION) for value in corner] for corner in aoi_corners
    ]
    serialized = json.dumps(rounded, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:_AOI_HASH_LENGTH]


def session_id_for(aoi_corners: Sequence[Sequence[float]], at: str | datetime | None = None) -> str:
    """`<timestamp>-<aoi digest>`, e.g. `20260903T140500-1f3a9c02`.

    Generated here and never sent by the frontend, which has no session id at all. Two properties
    matter: re-entering calibration on the same AOI at the same moment lands on the same session
    rather than forking one, and the id stays legible to a human reading a log line.

    `at` is a wall-clock timestamp (ISO 8601, or a `datetime`); it defaults to now. Passing it
    explicitly is what lets a caller resume a session it already knows the timestamp of.
    """
    moment = datetime.now() if at is None else (at if isinstance(at, datetime) else datetime.fromisoformat(at))
    return f"{moment:%Y%m%dT%H%M%S}-{_aoi_digest(aoi_corners)}"


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
                    float(calibration.table_x_px),
                    float(calibration.table_y_px),
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
