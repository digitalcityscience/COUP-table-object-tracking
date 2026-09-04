"""The calibration record: one SQLite file, two tables, written only by Python.

What the schema has to make possible is the thing step 6.3 cannot yet answer -- whether the
1-1.5 cm centre drift is homography residual or per-building sticking error. That question is
only answerable if every per-building measurement records *where on the table* it was taken,
which is why `table_x_px`/`table_y_px` are not optional extras here.
"""

from pathlib import Path
import sqlite3
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from session_store import (
    BuildingCalibration,
    SessionStore,
    session_id_for,
)

_AOI_CORNERS = [
    [10.027891448941773, 53.57073154051318],
    [10.033103979838955, 53.57073154051318],
    [10.033103979838955, 53.569183833647685],
    [10.027891448941773, 53.569183833647685],
]

_OTHER_AOI_CORNERS = [
    [9.9, 53.5],
    [9.91, 53.5],
    [9.91, 53.49],
    [9.9, 53.49],
]

_HOMOGRAPHY = [
    [0.33290904193780435, -0.0006958672719895796, -271.5773577346676],
    [0.004500496353812643, 0.3444114543974342, -143.20838797045926],
    [1.5132620012213879e-05, 1.3725096255251326e-05, 1.0],
]


def _store(tmp_path) -> SessionStore:
    return SessionStore(tmp_path / "calibration.sqlite3")


def _begin(store: SessionStore, aoi_corners=None) -> str:
    return store.begin_session(
        aoi_corners=aoi_corners if aoi_corners is not None else _AOI_CORNERS,
        ground_scale=327.0299,
        homography=_HOMOGRAPHY,
        global_k=0.654060,
    )


# --- session ids -------------------------------------------------------------------------


def test_session_id_is_stable_for_the_same_day_and_aoi():
    """Re-entering calibration on the same AOI must land on the same session, not a new one."""
    assert session_id_for(_AOI_CORNERS, at="2026-09-03T14:05:00") == session_id_for(
        _AOI_CORNERS, at="2026-09-03T14:05:00"
    )


def test_session_id_separates_two_different_aois():
    first = session_id_for(_AOI_CORNERS, at="2026-09-03T14:05:00")
    second = session_id_for(_OTHER_AOI_CORNERS, at="2026-09-03T14:05:00")
    assert first != second


def test_re_entering_the_same_aoi_later_the_same_day_is_the_same_session():
    """The workflow's own words: "Aynı AOI ile tekrar başlanınca aynı id çıkar."

    `server.py` mints an id on every accepted `map_calibration`, and an operator recalibrates
    repeatedly while coaxing the markers into being read. At second resolution each of those would
    open a fresh session and strand the previous one's building rows against a homography that had
    already been replaced.
    """
    assert session_id_for(_AOI_CORNERS, at="2026-09-03T09:00:00") == session_id_for(
        _AOI_CORNERS, at="2026-09-03T17:40:00"
    )


def test_a_different_day_on_the_same_aoi_is_a_different_session():
    """The projector has been moved and the room relit; it is not the same measurement set."""
    first = session_id_for(_AOI_CORNERS, at="2026-09-03T14:05:00")
    second = session_id_for(_AOI_CORNERS, at="2026-09-04T09:00:00")
    assert first != second


def test_the_session_id_does_not_depend_on_how_many_markers_were_decoded():
    """The handshake now carries 4-9 points depending on which grid markers the cameras read.

    Hashing the point list would make the id a signature of one detection round rather than of the
    AOI, so a marker blinking out would silently fork the session mid-sitting.
    """
    four_corners = _AOI_CORNERS
    with_extras = _AOI_CORNERS + [
        [10.030497714390364, 53.57073154051318],  # top edge midpoint
        [10.030497714390364, 53.569183833647685],  # bottom edge midpoint
        [10.027891448941773, 53.5699576870804],  # left edge midpoint
        [10.033103979838955, 53.5699576870804],  # right edge midpoint
        [10.030497714390364, 53.5699576870804],  # centre
    ]

    assert session_id_for(with_extras, at="2026-09-03T14:05:00") == session_id_for(
        four_corners, at="2026-09-03T14:05:00"
    )


def test_a_genuinely_different_area_still_gets_its_own_session():
    """The digest must not be so forgiving that two real AOIs collapse into one."""
    assert session_id_for(_AOI_CORNERS, at="2026-09-03T14:05:00") != session_id_for(
        _OTHER_AOI_CORNERS, at="2026-09-03T14:05:00"
    )


def test_session_id_survives_floating_point_formatting_noise():
    """The frontend re-serialises the AOI on every handshake; that must not fork the session."""
    noisy = [[value + 1e-12 for value in corner] for corner in _AOI_CORNERS]
    assert session_id_for(noisy, at="2026-09-03T14:05:00") == session_id_for(
        _AOI_CORNERS, at="2026-09-03T14:05:00"
    )


def test_session_id_carries_a_readable_date_and_an_aoi_hash():
    """Filenames and log lines are read by humans; an opaque digest would not be."""
    session_id = session_id_for(_AOI_CORNERS, at="2026-09-03T14:05:00")
    stamp, _, digest = session_id.partition("-")
    assert stamp == "20260903"
    assert len(digest) == 8 and all(character in "0123456789abcdef" for character in digest)


# --- sessions table ----------------------------------------------------------------------


def test_beginning_a_session_writes_one_row_with_the_derived_ground_scale(tmp_path):
    store = _store(tmp_path)

    session_id = _begin(store)

    row = store.session(session_id)
    assert row["id"] == session_id
    assert row["ground_scale"] == pytest.approx(327.0299)
    assert row["global_k"] == pytest.approx(0.654060)
    assert row["aoi_corners"] == _AOI_CORNERS
    assert row["homography"] == _HOMOGRAPHY
    assert row["created_at"]


def test_the_store_creates_its_file_and_schema_on_first_use(tmp_path):
    path = tmp_path / "nested" / "calibration.sqlite3"
    store = SessionStore(path)

    _begin(store)

    assert path.exists()
    with sqlite3.connect(path) as connection:
        tables = {
            name for (name,) in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    assert {"sessions", "session_buildings"} <= tables


def test_re_entering_the_same_session_refreshes_it_rather_than_duplicating_it(tmp_path):
    store = _store(tmp_path)

    first = store.begin_session(
        aoi_corners=_AOI_CORNERS,
        ground_scale=327.0299,
        homography=_HOMOGRAPHY,
        global_k=0.654060,
        at="2026-09-03T14:05:00",
    )
    second = store.begin_session(
        aoi_corners=_AOI_CORNERS,
        ground_scale=331.5,
        homography=_HOMOGRAPHY,
        global_k=0.663,
        at="2026-09-03T14:05:00",
    )

    assert first == second
    assert store.session_count() == 1
    # A recalibration of the same session overwrites the derived values; the old ones are not
    # the ones the buildings in that session were measured against any more.
    assert store.session(second)["ground_scale"] == pytest.approx(331.5)


def test_a_session_row_is_readable_after_reopening_the_file(tmp_path):
    path = tmp_path / "calibration.sqlite3"
    session_id = _begin(SessionStore(path))

    assert SessionStore(path).session(session_id)["global_k"] == pytest.approx(0.654060)


# --- session_buildings table -------------------------------------------------------------


def _calibration(**overrides) -> BuildingCalibration:
    values = {
        "building_id": "G17",
        "marker_id": 24,
        "rotation_offset_deg": -2.5,
        "offset_east_m": 0.35,
        "offset_north_m": -0.12,
        "scale_residual": 1.01,
        "table_x_px": 812.0,
        "table_y_px": 405.0,
    }
    values.update(overrides)
    return BuildingCalibration(**values)


def test_recording_a_building_calibration_keeps_where_on_the_table_it_was_measured(tmp_path):
    """The whole reason `table_x_px`/`table_y_px` exist (step 5's and the global fit's input)."""
    store = _store(tmp_path)
    session_id = _begin(store)

    store.record_building(session_id, _calibration())

    (row,) = store.session_buildings(session_id)
    assert row["building_id"] == "G17"
    assert row["marker_id"] == 24
    assert row["rotation_offset_deg"] == pytest.approx(-2.5)
    assert row["offset_east_m"] == pytest.approx(0.35)
    assert row["offset_north_m"] == pytest.approx(-0.12)
    assert row["scale_residual"] == pytest.approx(1.01)
    assert row["table_x_px"] == pytest.approx(812.0)
    assert row["table_y_px"] == pytest.approx(405.0)


def test_a_repeat_save_of_an_unmoved_block_replaces_its_row_despite_marker_jitter(tmp_path):
    """The reason the table position is quantised before it becomes part of the row's identity.

    `table_x_px` comes from an ArUco centre that wobbles by fractions of a pixel between frames,
    so two saves of a block nobody touched arrive as two different REALs. Unrounded, an operator
    dialling in one offset over four saves would leave four rows -- three superseded, all four
    weighted equally by whatever fits the global correction.
    """
    store = _store(tmp_path)
    session_id = _begin(store)

    store.record_building(session_id, _calibration(table_x_px=812.13, table_y_px=404.91))
    store.record_building(session_id, _calibration(table_x_px=811.87, table_y_px=405.22, rotation_offset_deg=-1.0))

    (row,) = store.session_buildings(session_id)
    assert row["rotation_offset_deg"] == pytest.approx(-1.0)
    assert row["table_x_px"] == pytest.approx(812.0)


def test_re_recording_the_same_building_at_the_same_spot_replaces_its_row(tmp_path):
    """An operator nudges a building repeatedly; only the pose they settled on is the answer.

    Identity is `(session, building, marker, table position)` — see
    `test_the_same_building_at_two_table_positions_is_two_measurements` for the other half of
    that rule, which is what step 6.3's five-position sweep depends on.
    """
    store = _store(tmp_path)
    session_id = _begin(store)

    store.record_building(session_id, _calibration(rotation_offset_deg=-2.5))
    store.record_building(session_id, _calibration(rotation_offset_deg=-1.0))

    (row,) = store.session_buildings(session_id)
    assert row["rotation_offset_deg"] == pytest.approx(-1.0)


def test_the_same_building_measured_in_two_sessions_keeps_both_rows(tmp_path):
    """Two AOIs are two homographies; collapsing them would destroy the position-dependent signal."""
    store = _store(tmp_path)
    first = _begin(store)
    second = _begin(store, aoi_corners=_OTHER_AOI_CORNERS)

    store.record_building(first, _calibration(offset_east_m=0.35))
    store.record_building(second, _calibration(offset_east_m=0.02))

    assert len(store.session_buildings(first)) == 1
    assert len(store.session_buildings(second)) == 1
    assert store.session_buildings(first)[0]["offset_east_m"] == pytest.approx(0.35)


def test_the_same_building_at_two_table_positions_is_two_measurements(tmp_path):
    """Step 6.3 puts one block in five places; each reading has to survive on its own."""
    store = _store(tmp_path)
    session_id = _begin(store)

    store.record_building(session_id, _calibration(table_x_px=100.0, table_y_px=100.0))
    store.record_building(session_id, _calibration(table_x_px=1500.0, table_y_px=700.0))

    positions = {
        (row["table_x_px"], row["table_y_px"]) for row in store.session_buildings(session_id)
    }
    assert positions == {(100.0, 100.0), (1500.0, 700.0)}


def test_recording_against_an_unknown_session_is_refused(tmp_path):
    """A building row with no session has no homography behind it, so it cannot be interpreted."""
    store = _store(tmp_path)

    with pytest.raises(ValueError, match="unknown session"):
        store.record_building("20260101T000000-deadbeef", _calibration())


def test_building_rows_come_back_in_a_stable_order(tmp_path):
    store = _store(tmp_path)
    session_id = _begin(store)

    store.record_building(session_id, _calibration(building_id="G17", marker_id=24))
    store.record_building(session_id, _calibration(building_id="G07", marker_id=12))
    store.record_building(session_id, _calibration(building_id="G11", marker_id=18))

    assert [row["building_id"] for row in store.session_buildings(session_id)] == [
        "G07",
        "G11",
        "G17",
    ]


# --- an unmeasured heading is a row too -------------------------------------------------


def test_a_measurement_can_be_filed_for_a_building_whose_heading_was_never_verified(tmp_path):
    """The normal case, not an edge one: nudging the east offset says nothing about the heading.

    `rotation_offset_deg` was `NOT NULL`, so the only way to file this row at all was to invent a
    measured zero for a heading nobody had looked at -- and this table is the input to the global
    fit, which would then weight that invention equally with a real measurement.
    """
    store = _store(tmp_path)
    session_id = _begin(store)

    store.record_building(session_id, _calibration(rotation_offset_deg=None, offset_east_m=0.35))

    (row,) = store.session_buildings(session_id)
    assert row["rotation_offset_deg"] is None
    assert row["offset_east_m"] == pytest.approx(0.35)


def test_a_store_written_before_the_heading_could_be_unmeasured_is_migrated(tmp_path):
    """An existing rig file was written with `rotation_offset_deg REAL NOT NULL`.

    SQLite cannot drop a NOT NULL in place, so the table is rebuilt on open. The rows already in
    it have to survive that rebuild in the right columns -- a reordered column list in a table
    copy silently shuffles every stored measurement one field sideways.
    """
    path = tmp_path / "calibration_sessions.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE sessions (
                id           TEXT PRIMARY KEY,
                created_at   TEXT NOT NULL,
                aoi_corners  TEXT NOT NULL,
                ground_scale REAL NOT NULL,
                homography   TEXT NOT NULL,
                global_k     REAL NOT NULL
            );
            CREATE TABLE session_buildings (
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
            INSERT INTO sessions VALUES
                ('20260903-abcd1234', '2026-09-03T14:05:00', '[[10.0, 53.0]]', 327.0, '[[1.0]]', 0.654);
            INSERT INTO session_buildings VALUES
                ('20260903-abcd1234', 'G07', 12, -2.5, 0.35, -0.12, 1.01, 811.0, 405.0,
                 '2026-09-03T14:06:00');
            """
        )

    store = SessionStore(path)
    (existing,) = store.session_buildings("20260903-abcd1234")

    assert existing["building_id"] == "G07"
    assert existing["marker_id"] == 12
    assert existing["rotation_offset_deg"] == pytest.approx(-2.5)
    assert existing["offset_east_m"] == pytest.approx(0.35)
    assert existing["offset_north_m"] == pytest.approx(-0.12)
    assert existing["scale_residual"] == pytest.approx(1.01)
    assert existing["table_x_px"] == pytest.approx(811.0)
    assert existing["table_y_px"] == pytest.approx(405.0)
    assert existing["recorded_at"] == "2026-09-03T14:06:00"

    store.record_building(
        "20260903-abcd1234",
        _calibration(building_id="G17", marker_id=24, rotation_offset_deg=None),
    )
    assert [row["rotation_offset_deg"] for row in store.session_buildings("20260903-abcd1234")] == [
        pytest.approx(-2.5),
        None,
    ]


def test_migrating_an_already_current_store_is_a_no_op(tmp_path):
    """It runs on every open, so it has to be safe to run on a file that has already had it."""
    store = _store(tmp_path)
    session_id = _begin(store)
    store.record_building(session_id, _calibration(rotation_offset_deg=-2.5))

    reopened = SessionStore(store.path)

    assert [row["rotation_offset_deg"] for row in reopened.session_buildings(session_id)] == [
        pytest.approx(-2.5)
    ]
