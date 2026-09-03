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


def test_session_id_separates_two_runs_over_the_same_aoi():
    first = session_id_for(_AOI_CORNERS, at="2026-09-03T14:05:00")
    second = session_id_for(_AOI_CORNERS, at="2026-09-04T09:00:00")
    assert first != second


def test_session_id_survives_floating_point_formatting_noise():
    """The frontend re-serialises the AOI on every handshake; that must not fork the session."""
    noisy = [[value + 1e-12 for value in corner] for corner in _AOI_CORNERS]
    assert session_id_for(noisy, at="2026-09-03T14:05:00") == session_id_for(
        _AOI_CORNERS, at="2026-09-03T14:05:00"
    )


def test_session_id_carries_a_readable_timestamp_and_an_aoi_hash():
    """Filenames and log lines are read by humans; an opaque digest would not be."""
    session_id = session_id_for(_AOI_CORNERS, at="2026-09-03T14:05:00")
    stamp, _, digest = session_id.partition("-")
    assert stamp == "20260903T140500"
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
