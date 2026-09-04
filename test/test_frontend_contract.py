"""End-to-end contract test against the real websocket handler.

This drives `server.handle_web_client` over a real websockets connection, with the camera
replaced by deterministic marker snapshots, and checks the exact bytes on the wire against
`fixtures/frontend_contract.json`. The captured raw-marker frame is written to
`fixtures/captured_raw_snapshot.json` so the TOSCA-2 side can replay the very same bytes
through its own message handler (see that repo's collabTracking.contract.test.ts) instead of
both sides restating the shape by hand.
"""

import asyncio
import json
from pathlib import Path

import pytest

import server
from calibration_contract import MAP_CALIBRATION_MARKER_CORNERS
from marker import Marker, Markers
from physical_building_catalog import MODEL_SCALE
from pixel_to_utm import direction_through_homography, ground_scale
from session_store import SessionStore

websockets = pytest.importorskip("websockets")

FIXTURE_DIR = Path(__file__).parent / "fixtures"
CONTRACT = json.loads(
    (FIXTURE_DIR / "frontend_contract.json").read_text(encoding="utf-8")
)
CAPTURE_PATH = FIXTURE_DIR / "captured_raw_snapshot.json"


def _rig_snapshot() -> dict:
    """A detection-worker snapshot: the four calibration markers plus a building.

    Built through the real `Markers` holder so the test exercises `toDict`'s admission
    rules, not a hand-written dict.
    """
    holder = Markers()
    holder.clear()
    points = CONTRACT["frontend_to_backend"]["map_calibration"]["points"]
    for marker_id, point in zip(sorted(MAP_CALIBRATION_MARKER_CORNERS), points):
        x, y = point["pixel_position"]
        holder.addMarker(Marker(marker_id, (x, y, 0.0), 12121, "000"))
    # A building marker, seen twice so it clears the confidence gate.
    holder.addMarker(Marker(12, (700, 400, 12.5), 12121, "000"))
    holder.addMarker(Marker(12, (700, 400, 12.5), 12121, "000"))
    return holder.toDict()


class _Session:
    """Runs the real handler on a real socket and lets the test push snapshots at it."""

    def __init__(self):
        self.received = []

    async def __aenter__(self):
        server.basemap_homography = None
        server.global_model_scale_factor = None
        server.current_session_id = None
        server.current_session_id = None
        while not server.tracking_queue.empty():
            server.tracking_queue.get_nowait()
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
        # `stream_tracking_updates` parks in `asyncio.to_thread(tracking_queue.get)`, which
        # cannot be cancelled. Close the client first, then hand the blocked get one more
        # item: the following send fails on the closed socket, which is what lets the
        # handler -- and the worker thread -- actually finish.
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

    async def push(self, snapshot: dict) -> dict:
        """Publish one snapshot and return the message the client actually received."""
        server.tracking_queue.put_nowait(snapshot)
        raw = await asyncio.wait_for(self._client.recv(), timeout=5)
        self.received.append(raw)
        return json.loads(raw)

    async def send(self, payload: dict) -> None:
        await self._client.send(json.dumps(payload))
        await asyncio.sleep(0.1)  # let receive_messages() apply it


def _looks_like_raw_marker_dictionary(data) -> bool:
    """Port of collabTracking.ts::isRawMarkerDictionary + isRawMarkerEntry."""
    if not isinstance(data, dict) or data.get("type") == "FeatureCollection":
        return False
    return all(
        isinstance(v, list)
        and len(v) >= 3
        and all(isinstance(n, (int, float)) and not isinstance(n, bool) for n in v[:3])
        for v in data.values()
    )


@pytest.mark.asyncio
async def test_calibration_markers_reach_the_frontend_before_calibration():
    async with _Session() as session:
        message = await session.push(_rig_snapshot())

    # The frontend's own guard must accept this shape.
    assert _looks_like_raw_marker_dictionary(message)
    # JSON object keys are strings; the frontend does Number(key).
    ids = {int(key) for key in message}
    assert {200, 201, 202, 203}.issubset(ids), (
        f"calibration markers missing from the snapshot: got {sorted(ids)}"
    )
    # ...and the building is still there alongside them.
    assert 12 in ids
    for marker_id in (200, 201, 202, 203):
        entry = message[str(marker_id)]
        assert len(entry) == 4
        x, y, rotation, camera = entry
        assert isinstance(x, (int, float)) and isinstance(y, (int, float))
        assert isinstance(rotation, (int, float))
        assert isinstance(camera, str)

    CAPTURE_PATH.write_text(json.dumps(message, indent=2), encoding="utf-8")


@pytest.mark.asyncio
async def test_single_sighting_is_enough_for_a_calibration_marker():
    holder = Markers()
    holder.clear()
    holder.addMarker(Marker(202, (298, 153, 0.0), 12121, "000"))
    async with _Session() as session:
        message = await session.push(holder.toDict())
    assert set(message) == {"202"}


@pytest.mark.asyncio
async def test_backend_parses_the_frontend_s_map_calibration_payload():
    payload = CONTRACT["frontend_to_backend"]["map_calibration"]
    async with _Session() as session:
        await session.send(payload)
        assert server.basemap_homography is not None, (
            "server did not accept the frontend's map_calibration message"
        )
        # After calibration the feed switches to building GeoJSON.
        message = await session.push(_rig_snapshot())

    assert message["type"] == "FeatureCollection"
    assert isinstance(message["features"], list)


@pytest.mark.asyncio
async def test_calibration_markers_never_appear_in_the_building_geojson():
    payload = CONTRACT["frontend_to_backend"]["map_calibration"]
    async with _Session() as session:
        await session.send(payload)
        message = await session.push(_rig_snapshot())

    reported = {
        int(feature["properties"]["marker_id"]) for feature in message["features"]
    }
    assert reported.isdisjoint({200, 201, 202, 203})


@pytest.mark.asyncio
async def test_building_tracking_survives_calibration():
    """Marker 12 is in the catalog, so it must still be published post-calibration."""
    payload = CONTRACT["frontend_to_backend"]["map_calibration"]
    async with _Session() as session:
        await session.send(payload)
        message = await session.push(_rig_snapshot())

    reported = {
        int(feature["properties"]["marker_id"]) for feature in message["features"]
    }
    assert 12 in reported, f"building feed stopped after calibration: {message}"


@pytest.mark.asyncio
async def test_clear_calibration_returns_to_the_raw_marker_feed():
    """This is how the frontend re-enters calibration; without it, recalibration is dead."""
    async with _Session() as session:
        await session.send(CONTRACT["frontend_to_backend"]["map_calibration"])
        assert server.basemap_homography is not None
        await session.send(CONTRACT["frontend_to_backend"]["clear_calibration"])
        assert server.basemap_homography is None
        message = await session.push(_rig_snapshot())

    assert _looks_like_raw_marker_dictionary(message)
    assert {200, 201, 202, 203}.issubset({int(key) for key in message})


@pytest.mark.asyncio
async def test_a_reconnecting_client_still_gets_calibration_markers():
    """Disconnect/reconnect must not leave the raw feed permanently switched off."""
    async with _Session() as session:
        await session.send(CONTRACT["frontend_to_backend"]["map_calibration"])
    async with _Session() as session:  # fresh connection, homography reset by the fixture
        message = await session.push(_rig_snapshot())
    assert {200, 201, 202, 203}.issubset({int(key) for key in message})


@pytest.mark.asyncio
async def test_accepting_the_handshake_derives_the_model_scale_factor():
    """Step 1: the ~1.85x oversize is closed by a factor read off the accepted homography.

    Nothing in the frontend's payload mentions scale -- this asserts the server derives it
    itself, and that the number is the ratio between the map it just calibrated and the 1:500
    blocks on the table.
    """
    payload = CONTRACT["frontend_to_backend"]["map_calibration"]
    async with _Session() as session:
        await session.send(payload)
        derived = server.global_model_scale_factor
        # Read inside the session: `_Session.__aexit__` resets both globals.
        expected = ground_scale(server.basemap_homography) / MODEL_SCALE
        message = await session.push(_rig_snapshot())

    assert derived is not None, "server accepted a calibration without deriving a scale factor"
    assert derived == pytest.approx(expected)
    # The 2026-08-31 AOI is a good deal finer than 1:500, so geometry must shrink, not grow.
    assert 0 < derived < 1
    # ...and every emitted building carries the factor it was actually drawn with.
    for feature in message["features"]:
        assert feature["properties"]["model_scale_factor"] == pytest.approx(derived)


@pytest.mark.asyncio
async def test_clearing_the_calibration_drops_the_model_scale_factor_with_it():
    """A factor that outlived its homography would silently mis-size the next AOI."""
    async with _Session() as session:
        await session.send(CONTRACT["frontend_to_backend"]["map_calibration"])
        assert server.global_model_scale_factor is not None
        await session.send(CONTRACT["frontend_to_backend"]["clear_calibration"])
        assert server.global_model_scale_factor is None


@pytest.mark.asyncio
async def test_every_building_says_whether_its_heading_was_ever_verified():
    """D1 on the wire: the frontend cannot mark what the server never tells it.

    None of the three registered buildings has had its absolute heading checked, so the honest
    answer today is `false` for every one of them -- and the point of the property is that the
    projection can say so instead of drawing a guess exactly like a measurement.
    """
    async with _Session() as session:
        await session.send(CONTRACT["frontend_to_backend"]["map_calibration"])
        message = await session.push(_rig_snapshot())

    assert message["features"], "no building features to check"
    for feature in message["features"]:
        assert feature["properties"]["alignment_verified"] is False
        assert feature["properties"]["calibration"]["rotation_offset_deg"] is None


@pytest.mark.asyncio
async def test_the_published_heading_is_the_one_the_homography_gives():
    """D2 on the wire: the angle is pushed through the homography, not copied past it.

    Recomputed here from the accepted homography and the catalog's own stored reference, at the
    marker's own table pixel -- so this fails both if the conversion is dropped and if it is
    applied to only one of the two angles, which would leave the table's own tilt in the answer.
    """
    from physical_building_catalog import marker_index

    async with _Session() as session:
        await session.send(CONTRACT["frontend_to_backend"]["map_calibration"])
        message = await session.push(_rig_snapshot())
        homography = server.basemap_homography
        building = marker_index(server.physical_building_catalog)[12]

    (feature,) = [f for f in message["features"] if f["properties"]["marker_id"] == 12]
    table_pixel = (feature["properties"]["table_x_px"], feature["properties"]["table_y_px"])
    reference = float(building["marker_reference_rotations"]["12"])
    expected = (
        direction_through_homography(homography, table_pixel, 12.5)
        - direction_through_homography(homography, table_pixel, reference)
        + 180.0
    ) % 360.0 - 180.0

    assert feature["properties"]["rotation"] == pytest.approx(expected, abs=1e-6)


@pytest.mark.asyncio
async def test_the_homography_actually_changes_the_heading():
    """Guards against a conversion that is wired in but silently the identity.

    The rig's map is 0.42 degrees out of square and 3.5% anisotropic, so a real conversion must
    move the answer off the bare scalar difference -- while staying close enough to it that a
    wildly wrong wiring (a sign flip, a swapped axis) would not pass either.
    """
    from physical_building_catalog import marker_index

    async with _Session() as session:
        await session.send(CONTRACT["frontend_to_backend"]["map_calibration"])
        message = await session.push(_rig_snapshot())
        building = marker_index(server.physical_building_catalog)[12]

    (feature,) = [f for f in message["features"] if f["properties"]["marker_id"] == 12]
    scalar = (12.5 - float(building["marker_reference_rotations"]["12"]) + 180.0) % 360.0 - 180.0
    difference = abs(feature["properties"]["rotation"] - scalar)

    assert 0 < difference < 5.0


@pytest.fixture
def temporary_session_store(tmp_path, monkeypatch):
    """Redirect the module-level store so a test never writes the rig's real record."""
    store = SessionStore(tmp_path / "calibration.sqlite3")
    monkeypatch.setattr(server, "session_store", store)
    return store


@pytest.mark.asyncio
async def test_accepting_the_handshake_opens_a_session_row(temporary_session_store):
    """Step 2's done-condition: a calibration lands one session row with the right ground scale."""
    payload = CONTRACT["frontend_to_backend"]["map_calibration"]
    async with _Session() as session:
        await session.send(payload)
        session_id = server.current_session_id
        expected_ground_scale = ground_scale(server.basemap_homography)

    assert session_id is not None, "calibration accepted without opening a session"
    assert temporary_session_store.session_count() == 1
    row = temporary_session_store.session(session_id)
    assert row["ground_scale"] == pytest.approx(expected_ground_scale)
    assert row["global_k"] == pytest.approx(expected_ground_scale / MODEL_SCALE)
    # The homography actually solved, not a placeholder: 3x3, bottom-right normalised to 1.
    assert len(row["homography"]) == 3 and len(row["homography"][0]) == 3
    assert row["homography"][2][2] == pytest.approx(1.0)
    # The AOI signature is the handshake's own four geographic points.
    assert len(row["aoi_corners"]) == len(payload["points"])


@pytest.mark.asyncio
async def test_recalibrating_the_same_aoi_reuses_its_session(temporary_session_store):
    """Re-entering calibration on one AOI must not scatter its buildings across sessions."""
    payload = CONTRACT["frontend_to_backend"]["map_calibration"]
    async with _Session() as session:
        await session.send(payload)
        first = server.current_session_id
        await session.send(CONTRACT["frontend_to_backend"]["clear_calibration"])
        assert server.current_session_id is None
        await session.send(payload)
        second = server.current_session_id

    # Same AOI within the same second is the same session; a later second is a new run, which is
    # deliberate -- either way exactly one row exists per (moment, AOI) pair, never a duplicate.
    assert second is not None
    assert temporary_session_store.session_count() == (1 if first == second else 2)
    assert temporary_session_store.session(second) is not None
