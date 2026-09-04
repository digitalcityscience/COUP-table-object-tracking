"""The `building_calibration` message, end to end through the real websocket handler.

Two destinations on accept, and the split matters: the *working* catalog
(`building_catalog/physical-building-catalog.json`) plus a SQLite row. The runtime catalog next
to `server.py` is promoted separately by `publish-to-runtime.ps1`, so an operator calibrating on
a live table cannot break the running session's own catalog file by saving a value.
"""

import copy
import json
from pathlib import Path

import pytest

import server
from marker import Marker, Markers
from physical_building_catalog import (
    alignment_is_verified,
    building_calibration_of,
    load_catalog,
    save_catalog,
    table_millimetres_to_local_metres,
)
from session_store import SessionStore

websockets = pytest.importorskip("websockets")

FIXTURE_DIR = Path(__file__).parent / "fixtures"
CONTRACT = json.loads((FIXTURE_DIR / "frontend_contract.json").read_text(encoding="utf-8"))

#: Marker 12 is G07 in the shipped catalog, and the one building marker the rig snapshot carries.
_MARKER_ID = 12
_BUILDING_ID = "G07"


def _rig_snapshot() -> dict:
    holder = Markers()
    holder.clear()
    points = CONTRACT["frontend_to_backend"]["map_calibration"]["points"]
    from calibration_contract import MAP_CALIBRATION_MARKER_CORNERS

    for marker_id, point in zip(sorted(MAP_CALIBRATION_MARKER_CORNERS), points):
        x, y = point["pixel_position"]
        holder.addMarker(Marker(marker_id, (x, y, 0.0), 12121, "000"))
    holder.addMarker(Marker(_MARKER_ID, (700, 400, 12.5), 12121, "000"))
    holder.addMarker(Marker(_MARKER_ID, (700, 400, 12.5), 12121, "000"))
    return holder.toDict()


@pytest.fixture
def rig(tmp_path, monkeypatch):
    """A server whose catalog file and calibration record both live in `tmp_path`.

    The working catalog is seeded from the real one, so the test exercises the shipped buildings
    without ever writing to the repository's copy.
    """
    working_catalog_path = tmp_path / "working" / "physical-building-catalog.json"
    working_catalog_path.parent.mkdir(parents=True)
    real_catalog = load_catalog(Path(server.PHYSICAL_BUILDING_CATALOG_PATH))
    save_catalog(working_catalog_path, copy.deepcopy(real_catalog))

    monkeypatch.setattr(server, "WORKING_BUILDING_CATALOG_PATH", str(working_catalog_path))
    monkeypatch.setattr(server, "session_store", SessionStore(tmp_path / "calibration.sqlite3"))
    monkeypatch.setattr(server, "physical_building_catalog", copy.deepcopy(real_catalog))
    monkeypatch.setattr(
        server, "physical_buildings_by_marker", server.marker_index(server.physical_building_catalog)
    )
    return type(
        "Rig",
        (),
        {"working_catalog_path": working_catalog_path, "store": server.session_store},
    )


class _Session:
    """The real handler on a real socket (mirrors `test_frontend_contract._Session`)."""

    async def __aenter__(self):
        server.basemap_homography = None
        server.global_model_scale_factor = None
        server.current_session_id = None
        server.latest_table_pixel_positions.clear()
        while not server.tracking_queue.empty():
            server.tracking_queue.get_nowait()
        import asyncio

        self._handler_done = asyncio.Event()

        async def handler(websocket):
            try:
                await server.handle_web_client(websocket)
            finally:
                self._handler_done.set()

        self._server = await websockets.serve(handler, "127.0.0.1", 0)
        port = self._server.sockets[0].getsockname()[1]
        self._client = await websockets.connect(f"ws://127.0.0.1:{port}")
        return self

    async def __aexit__(self, *exc):
        import asyncio

        await self._client.close()
        server.tracking_queue.put_nowait({})
        try:
            await asyncio.wait_for(self._handler_done.wait(), timeout=5)
        except asyncio.TimeoutError:
            pass
        self._server.close()
        await asyncio.wait_for(self._server.wait_closed(), timeout=5)
        while not server.tracking_queue.empty():
            server.tracking_queue.get_nowait()
        server.basemap_homography = None
        server.global_model_scale_factor = None
        server.current_session_id = None
        server.latest_table_pixel_positions.clear()

    async def push(self, snapshot: dict) -> dict:
        server.tracking_queue.put_nowait(snapshot)
        import asyncio

        raw = await asyncio.wait_for(self._client.recv(), timeout=5)
        return json.loads(raw)

    async def send(self, payload: dict) -> None:
        import asyncio

        await self._client.send(json.dumps(payload))
        await asyncio.sleep(0.1)


def _calibration_message(**overrides) -> dict:
    """The documented message, read from the shared contract fixture rather than restated here."""
    payload = dict(CONTRACT["frontend_to_backend"]["building_calibration"])
    payload = {key: value for key, value in payload.items() if not key.startswith("_")}
    payload.update(overrides)
    return payload


def test_the_documented_message_names_the_catalogued_building_this_test_uses():
    """If the fixture ever pointed at another building, every test below would silently drift."""
    documented = CONTRACT["frontend_to_backend"]["building_calibration"]
    assert documented["building_id"] == _BUILDING_ID
    assert documented["marker_id"] == _MARKER_ID


async def _calibrated_session(session) -> None:
    await session.send(CONTRACT["frontend_to_backend"]["map_calibration"])
    # One snapshot so the server has seen where marker 12 actually sits on the table.
    await session.push(_rig_snapshot())


# --- the feature the panel reads ---------------------------------------------------------


@pytest.mark.asyncio
async def test_a_published_building_carries_where_it_sits_on_the_table(rig):
    """The panel's table diagram needs this, and so does the measurement's own record."""
    async with _Session() as session:
        await session.send(CONTRACT["frontend_to_backend"]["map_calibration"])
        message = await session.push(_rig_snapshot())

    (feature,) = [f for f in message["features"] if f["properties"]["marker_id"] == _MARKER_ID]
    assert feature["properties"]["table_x_px"] == pytest.approx(700)
    assert feature["properties"]["table_y_px"] == pytest.approx(400)


# --- accepting a calibration -------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_accepted_calibration_reaches_the_working_catalog(rig):
    async with _Session() as session:
        await _calibrated_session(session)
        await session.send(_calibration_message())

    catalog = load_catalog(rig.working_catalog_path)
    (building,) = [b for b in catalog["buildings"] if b["building_id"] == _BUILDING_ID]
    documented = _calibration_message()
    assert building_calibration_of(building) == {
        "rotation_offset_deg": pytest.approx(documented["rotation_offset_deg"]),
        # Table millimetres on the wire, catalog metres on disk: 1 mm = 0.5 m at 1:500.
        "offset_east_m": pytest.approx(table_millimetres_to_local_metres(documented["offset_east_mm"])),
        "offset_north_m": pytest.approx(table_millimetres_to_local_metres(documented["offset_north_mm"])),
        "scale_residual": pytest.approx(documented["scale_residual"]),
    }


@pytest.mark.asyncio
async def test_an_accepted_calibration_is_filed_against_the_session_and_table_position(rig):
    """Step 2's `table_x_px`/`table_y_px` are stamped by the server, not echoed by the client."""
    async with _Session() as session:
        await _calibrated_session(session)
        await session.send(_calibration_message())
        session_id = server.current_session_id

    (row,) = rig.store.session_buildings(session_id)
    assert row["building_id"] == _BUILDING_ID
    assert row["marker_id"] == _MARKER_ID
    assert row["rotation_offset_deg"] == pytest.approx(-2.5)
    assert row["table_x_px"] == pytest.approx(700)
    assert row["table_y_px"] == pytest.approx(400)


@pytest.mark.asyncio
async def test_an_accepted_calibration_takes_effect_on_the_live_feed_immediately(rig):
    """The operator is watching the projection; a saved nudge they cannot see is unusable."""
    async with _Session() as session:
        await _calibrated_session(session)
        before = await session.push(_rig_snapshot())
        await session.send(_calibration_message(rotation_offset_deg=-30.0))
        after = await session.push(_rig_snapshot())

    def rotation_of(message):
        return [
            f["properties"]["rotation"]
            for f in message["features"]
            if f["properties"]["marker_id"] == _MARKER_ID
        ][0]

    assert rotation_of(after) == pytest.approx(rotation_of(before) - 30.0)


@pytest.mark.asyncio
async def test_a_saved_calibration_outlives_a_restart(rig):
    """Step 3's done-condition, as the restart actually happens: reload the file from disk."""
    async with _Session() as session:
        await _calibrated_session(session)
        await session.send(_calibration_message())

    reloaded = load_catalog(rig.working_catalog_path)
    (building,) = [b for b in reloaded["buildings"] if b["building_id"] == _BUILDING_ID]
    assert building_calibration_of(building)["rotation_offset_deg"] == pytest.approx(-2.5)


@pytest.mark.asyncio
async def test_a_second_save_merges_onto_the_first(rig):
    """The panel saves the axis the operator touched; the earlier ones must not reset."""
    async with _Session() as session:
        await _calibrated_session(session)
        await session.send(_calibration_message())
        await session.send(
            {
                "type": "building_calibration",
                "version": 2,
                "building_id": _BUILDING_ID,
                "marker_id": _MARKER_ID,
                "offset_north_mm": -0.40,
            }
        )

    documented = _calibration_message()
    catalog = load_catalog(rig.working_catalog_path)
    (building,) = [b for b in catalog["buildings"] if b["building_id"] == _BUILDING_ID]
    stored = building_calibration_of(building)
    assert stored["offset_north_m"] == pytest.approx(table_millimetres_to_local_metres(-0.40))
    assert stored["rotation_offset_deg"] == pytest.approx(documented["rotation_offset_deg"])
    assert stored["offset_east_m"] == pytest.approx(
        table_millimetres_to_local_metres(documented["offset_east_mm"])
    )


# --- refusing a calibration --------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_calibration_sent_before_the_handshake_is_refused(rig):
    """A residual has no meaning without the global mapping it is a residual of."""
    async with _Session() as session:
        await session.send(_calibration_message())

    catalog = load_catalog(rig.working_catalog_path)
    (building,) = [b for b in catalog["buildings"] if b["building_id"] == _BUILDING_ID]
    assert alignment_is_verified(building) is False


@pytest.mark.asyncio
async def test_a_calibration_for_an_uncatalogued_marker_is_refused(rig):
    async with _Session() as session:
        await _calibrated_session(session)
        await session.send(_calibration_message(marker_id=999, building_id="G99"))
        session_id = server.current_session_id

    assert rig.store.session_buildings(session_id) == []


@pytest.mark.asyncio
async def test_a_calibration_whose_building_id_contradicts_its_marker_is_refused(rig):
    """A mismatch means the panel and the catalog disagree about what the operator moved."""
    async with _Session() as session:
        await _calibrated_session(session)
        await session.send(_calibration_message(building_id="G17"))
        session_id = server.current_session_id

    assert rig.store.session_buildings(session_id) == []
    catalog = load_catalog(rig.working_catalog_path)
    for building in catalog["buildings"]:
        assert alignment_is_verified(building) is False


@pytest.mark.asyncio
async def test_a_calibration_with_an_unknown_field_is_refused_whole(rig):
    """Partial application would leave the catalog in a state the operator never chose."""
    async with _Session() as session:
        await _calibrated_session(session)
        await session.send(_calibration_message(offset_up_mm=1.0))

    catalog = load_catalog(rig.working_catalog_path)
    (building,) = [b for b in catalog["buildings"] if b["building_id"] == _BUILDING_ID]
    assert alignment_is_verified(building) is False


@pytest.mark.asyncio
async def test_a_calibration_for_a_building_never_seen_on_the_table_is_refused(rig):
    """With no sighting there is no table position, and the measurement would be unusable."""
    async with _Session() as session:
        # Calibrate, but never publish a snapshot: the server has seen no marker positions.
        await session.send(CONTRACT["frontend_to_backend"]["map_calibration"])
        await session.send(_calibration_message())
        session_id = server.current_session_id

    assert rig.store.session_buildings(session_id) == []


@pytest.mark.asyncio
async def test_a_malformed_calibration_does_not_kill_the_connection(rig):
    """The feed must survive a bad message; a dead socket ends the operator's session."""
    async with _Session() as session:
        await _calibrated_session(session)
        await session.send({"type": "building_calibration"})
        message = await session.push(_rig_snapshot())

    assert message["type"] == "FeatureCollection"


@pytest.mark.asyncio
async def test_a_published_building_carries_the_calibration_it_is_drawn_with(rig):
    """What lets the panel open at the building's real pose instead of at zero."""
    async with _Session() as session:
        await _calibrated_session(session)
        await session.send(_calibration_message())
        message = await session.push(_rig_snapshot())

    (feature,) = [f for f in message["features"] if f["properties"]["marker_id"] == _MARKER_ID]
    documented = _calibration_message()
    assert feature["properties"]["calibration"] == {
        "rotation_offset_deg": pytest.approx(documented["rotation_offset_deg"]),
        "offset_east_mm": pytest.approx(documented["offset_east_mm"]),
        "offset_north_mm": pytest.approx(documented["offset_north_mm"]),
        "scale_residual": pytest.approx(documented["scale_residual"]),
    }
    assert feature["properties"]["model_scale"] == 500


@pytest.mark.asyncio
async def test_a_measurement_is_refused_once_the_block_has_left_the_table(rig, monkeypatch):
    """`table_x_px` exists to separate a position-dependent error from a per-building one.

    Filing a measurement at a position the block is no longer at corrupts exactly that signal, so
    a stale reading is refused rather than silently stamped onto the row.
    """
    async with _Session() as session:
        await _calibrated_session(session)
        # The marker was seen, then the block came off the table and nothing refreshed it.
        stale = {
            marker_id: (x, y, seen_at - server.TABLE_POSITION_MAX_AGE_SECONDS - 5)
            for marker_id, (x, y, seen_at) in server.latest_table_pixel_positions.items()
        }
        server.latest_table_pixel_positions.update(stale)
        await session.send(_calibration_message())
        session_id = server.current_session_id

    assert rig.store.session_buildings(session_id) == []


@pytest.mark.asyncio
async def test_clearing_the_calibration_forgets_where_every_block_was(rig):
    """A pixel read under the old homography says nothing about the next one's table."""
    async with _Session() as session:
        await _calibrated_session(session)
        assert server.latest_table_pixel_positions
        await session.send(CONTRACT["frontend_to_backend"]["clear_calibration"])
        assert server.latest_table_pixel_positions == {}


@pytest.mark.asyncio
async def test_the_live_catalog_and_the_working_catalog_agree_after_a_partial_save(rig):
    """One merge, one result, copied to both -- never two merges onto two different bases.

    The two catalogs are reconciled only by a manual publish, so they can legitimately hold
    different values. Merging a partial message onto each separately would draw one thing on the
    table, persist another, and record a third.
    """
    async with _Session() as session:
        await _calibrated_session(session)
        await session.send(_calibration_message())
        await session.send(
            {
                "type": "building_calibration",
                "version": 2,
                "building_id": _BUILDING_ID,
                "marker_id": _MARKER_ID,
                "offset_north_mm": -0.40,
            }
        )
        live = building_calibration_of(server.physical_buildings_by_marker[_MARKER_ID])

    catalog = load_catalog(rig.working_catalog_path)
    (building,) = [b for b in catalog["buildings"] if b["building_id"] == _BUILDING_ID]
    assert live == building_calibration_of(building)
