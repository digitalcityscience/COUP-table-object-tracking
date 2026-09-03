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
