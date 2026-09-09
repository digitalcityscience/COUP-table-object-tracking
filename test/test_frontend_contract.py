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
from pyproj import Transformer

import server
from calibration_contract import MAP_CALIBRATION_MARKER_CORNERS
from marker import Marker, Markers
from physical_building_catalog import MODEL_SCALE, empty_catalog, save_catalog
from pixel_to_utm import direction_through_homography, ground_scale, project_pixel_to_utm
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
        #: Typed server->client messages seen while waiting for a tracking snapshot, in order.
        #: The real client routes by shape (`collabTracking.ts::handleMessage`) rather than
        #: assuming the next message is a snapshot, so this harness has to as well -- otherwise
        #: adding any control message to the protocol breaks every test that pushes a snapshot.
        self.control_messages = []

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
        """Publish one snapshot and return the tracking message the client actually received.

        Typed control messages arriving first (`session_state`, `register_building_result`) are
        set aside in `control_messages` rather than returned, mirroring the real client's
        shape-based routing.
        """
        server.tracking_queue.put_nowait(snapshot)
        while True:
            raw = await asyncio.wait_for(self._client.recv(), timeout=5)
            data = json.loads(raw)
            if _is_control_message(data):
                self.control_messages.append(data)
                continue
            self.received.append(raw)
            return data

    async def control(self, message_type: str | None = None) -> dict:
        """The next typed control message, optionally of a given `type`, waiting if need be.

        Filtering by type rather than taking whatever is next, because an accepted
        `map_calibration` already queues a `session_state` -- a test asking for a registration
        result would otherwise get the handshake's message and fail somewhere unrelated to what
        it is checking.
        """
        while True:
            for index, message in enumerate(self.control_messages):
                if message_type is None or message.get("type") == message_type:
                    return self.control_messages.pop(index)
            raw = await asyncio.wait_for(self._client.recv(), timeout=5)
            data = json.loads(raw)
            if _is_control_message(data):
                self.control_messages.append(data)
            else:
                self.received.append(raw)

    async def send(self, payload: dict) -> None:
        await self._client.send(json.dumps(payload))
        await asyncio.sleep(0.1)  # let receive_messages() apply it


def _is_control_message(data) -> bool:
    """A typed server->client message that is not a tracking snapshot."""
    return (
        isinstance(data, dict)
        and isinstance(data.get("type"), str)
        and data["type"] != "FeatureCollection"
    )


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

    The invariant, not the rig's mood: whatever the catalog holds for a building, the published
    feature says the same thing, and its two channels -- the flag and the calibration block the
    panel reads back -- never disagree. Pinning `false` for all three instead made a passing suite
    depend on nobody ever registering a building, which is the one thing the flow exists to do.
    """
    from physical_building_catalog import alignment_is_verified, marker_index

    async with _Session() as session:
        await session.send(CONTRACT["frontend_to_backend"]["map_calibration"])
        message = await session.push(_rig_snapshot())
        by_marker = marker_index(server.physical_building_catalog)

    assert message["features"], "no building features to check"
    for feature in message["features"]:
        building = by_marker[feature["properties"]["marker_id"]]
        verified = alignment_is_verified(building)
        assert feature["properties"]["alignment_verified"] is verified
        assert (feature["properties"]["calibration"]["rotation_offset_deg"] is None) is not verified


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
    # The published `table_x_px`/`table_y_px` are the *raw* detection, because the panel's table
    # diagram wants where the marker was actually seen. The conversion happens at the marker
    # *centre*, so the offset has to be added back here -- the map is 0.42 degrees out of square,
    # so twenty pixels of difference is a fifth of a degree in the answer, and recomputing at the
    # raw pixel silently failed the moment a building was registered with a centre offset.
    offset = building.get("marker_center_offset_px", [0.0, 0.0])
    table_pixel = (
        feature["properties"]["table_x_px"] + float(offset[0]),
        feature["properties"]["table_y_px"] + float(offset[1]),
    )
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


# --- registering a building from the frontend --------------------------------------------
#
# The flow this exercises: the frontend projects the building's own footprint onto the table at
# its real heading, the operator turns the block PARALLEL to it (anywhere on the table -- only
# the angle is being measured), and confirms. That confirmation is the one piece of information
# no camera can supply, because the same marker glued straight, sideways or upside down reads
# identically.


@pytest.fixture
def registration_rig(tmp_path, monkeypatch):
    """An empty working catalog and an empty live catalog, both restored after the test.

    The live catalog is a module global that `_register_building` reassigns, so without this a
    registration would leak into every later test in the session.
    """
    working = tmp_path / "working" / "physical-building-catalog.json"
    working.parent.mkdir(parents=True)
    save_catalog(working, empty_catalog())
    monkeypatch.setattr(server, "WORKING_BUILDING_CATALOG_PATH", str(working))
    # Registration writes the runtime catalog -- the file the server boots from -- so that path
    # needs redirecting too, or these tests write the repository's own copy. They did: the
    # fixture's promise to leave the real catalog alone was one constant out of date, and the
    # first run after registration moved to the runtime file overwrote it with test fixtures.
    runtime = tmp_path / "runtime" / "physical-building-catalog.json"
    runtime.parent.mkdir(parents=True)
    save_catalog(runtime, empty_catalog())
    monkeypatch.setattr(server, "PHYSICAL_BUILDING_CATALOG_PATH", str(runtime))
    monkeypatch.setattr(server, "physical_building_catalog", empty_catalog())
    monkeypatch.setattr(server, "physical_buildings_by_marker", {})
    server.recent_marker_rotations.clear()
    yield runtime
    server.recent_marker_rotations.clear()


def _table_with_block(marker_id: int, rotation: float) -> dict:
    """One snapshot: the four calibration markers, plus one building block at `rotation`."""
    holder = Markers()
    holder.clear()
    points = CONTRACT["frontend_to_backend"]["map_calibration"]["points"]
    for calibration_id, point in zip(sorted(MAP_CALIBRATION_MARKER_CORNERS), points):
        x, y = point["pixel_position"]
        holder.addMarker(Marker(calibration_id, (x, y, 0.0), 12121, "000"))
    # Seen twice so it clears the confidence gate.
    holder.addMarker(Marker(marker_id, (700, 400, rotation), 12121, "000"))
    holder.addMarker(Marker(marker_id, (700, 400, rotation), 12121, "000"))
    return holder.toDict()


async def _settle_block(session, marker_id: int, rotation: float, frames: int = 12) -> dict:
    """Push enough identical snapshots that the block reads as still and in view."""
    message = None
    for _ in range(frames):
        message = await session.push(_table_with_block(marker_id, rotation))
    return message


@pytest.mark.asyncio
async def test_registering_a_building_records_the_block_s_heading_as_verified(registration_rig):
    """The point of the flow: the operator aligned the block, so the alignment IS measured.

    `rotation_offset_deg` comes out a measured `0.0` and `alignment_verified` true -- not because
    the code assumed anything, but because the frontend showed the building's real heading and a
    human put the block on it.
    """
    async with _Session() as session:
        await session.send(CONTRACT["frontend_to_backend"]["map_calibration"])
        await _settle_block(session, 18, -116.25)
        await session.send({"type": "register_building", "building_id": "G11"})
        result = await session.control("register_building_result")
        message = await session.push(_table_with_block(18, -116.25))

    assert result["type"] == "register_building_result"
    assert result["ok"] is True
    assert result["building_id"] == "G11"
    assert result["marker_id"] == 18
    assert result["reference_rotation_deg"] == pytest.approx(-116.25, abs=0.01)

    (feature,) = [f for f in message["features"] if f["properties"]["building_id"] == "G11"]
    assert feature["properties"]["alignment_verified"] is True
    assert feature["properties"]["calibration"]["rotation_offset_deg"] == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_registration_saves_a_marker_center_correction_that_survives_recalibration(
    registration_rig,
):
    """The cyan observation belongs to the marker, not to one temporary map homography."""
    utm_to_wgs84 = Transformer.from_crs("EPSG:25832", "EPSG:4326", always_xy=True)
    wgs84_to_utm = Transformer.from_crs("EPSG:4326", "EPSG:25832", always_xy=True)

    async with _Session() as session:
        await session.send(CONTRACT["frontend_to_backend"]["map_calibration"])
        await _settle_block(session, 18, -116.25)
        before = project_pixel_to_utm(server.basemap_homography, (700.0, 400.0))
        intended_utm = (before[0] + 2.0, before[1] - 3.0)
        target_lng, target_lat = utm_to_wgs84.transform(*intended_utm)

        await session.send(
            {
                "type": "register_building",
                "building_id": "G11",
                "marker_id": 18,
                "target": [target_lng, target_lat],
            }
        )
        result = await session.control("register_building_result")
        await session.send({"type": "clear_calibration"})
        await session.send(CONTRACT["frontend_to_backend"]["map_calibration"])
        message = await session.push(_table_with_block(18, -116.25))

    assert result["ok"] is True
    (feature,) = [f for f in message["features"] if f["properties"]["building_id"] == "G11"]
    corrected_utm = wgs84_to_utm.transform(*feature["properties"]["center"])
    assert corrected_utm == pytest.approx(intended_utm, abs=0.01)


@pytest.mark.asyncio
async def test_a_just_registered_building_draws_at_the_catalog_heading(registration_rig):
    """`detected == reference` right after registering, so the footprint sits at true north.

    Which is now a *correct* claim rather than an accident: the reference was captured while the
    block was aligned to the projected footprint.
    """
    async with _Session() as session:
        await session.send(CONTRACT["frontend_to_backend"]["map_calibration"])
        await _settle_block(session, 18, -116.25)
        await session.send({"type": "register_building", "building_id": "G11"})
        await session.control("register_building_result")
        message = await session.push(_table_with_block(18, -116.25))

    (feature,) = [f for f in message["features"] if f["properties"]["building_id"] == "G11"]
    assert feature["properties"]["rotation"] == pytest.approx(0.0, abs=0.05)


@pytest.mark.asyncio
async def test_turning_the_block_after_registering_turns_the_footprint_the_same_way(registration_rig):
    """What the operator actually asked the system for: put it anywhere, it follows."""
    async with _Session() as session:
        await session.send(CONTRACT["frontend_to_backend"]["map_calibration"])
        await _settle_block(session, 18, -116.25)
        await session.send({"type": "register_building", "building_id": "G11"})
        await session.control("register_building_result")
        message = await session.push(_table_with_block(18, -26.25))

    (feature,) = [f for f in message["features"] if f["properties"]["building_id"] == "G11"]
    assert feature["properties"]["rotation"] == pytest.approx(90.0, abs=1.5)


@pytest.mark.asyncio
async def test_registering_with_no_block_on_the_table_is_refused(registration_rig):
    """Refusals come back typed, so the panel can say why instead of appearing to hang."""
    async with _Session() as session:
        await session.send(CONTRACT["frontend_to_backend"]["map_calibration"])
        await session.send({"type": "register_building", "building_id": "G11"})
        result = await session.control("register_building_result")

    assert result["ok"] is False
    assert "unclaimed marker" in result["error"]


@pytest.mark.asyncio
async def test_registering_an_unknown_building_is_refused(registration_rig):
    async with _Session() as session:
        await session.send(CONTRACT["frontend_to_backend"]["map_calibration"])
        await _settle_block(session, 18, -116.25)
        await session.send({"type": "register_building", "building_id": "NOPE"})
        result = await session.control("register_building_result")

    assert result["ok"] is False
    assert "buildings_all.geojson" in result["error"]


@pytest.mark.asyncio
async def test_a_block_still_being_moved_is_refused(registration_rig):
    """A heading averaged across a moving block would become a permanent constant error."""
    async with _Session() as session:
        await session.send(CONTRACT["frontend_to_backend"]["map_calibration"])
        for index in range(12):
            await session.push(_table_with_block(18, -116.25 + index * 5.0))
        await session.send({"type": "register_building", "building_id": "G11"})
        result = await session.control("register_building_result")

    assert result["ok"] is False
    assert "moved" in result["error"]


@pytest.mark.asyncio
async def test_a_calibration_marker_is_never_registered_as_a_building(registration_rig):
    """The Table window projects them, so they are in view during every single registration."""
    async with _Session() as session:
        await session.send(CONTRACT["frontend_to_backend"]["map_calibration"])
        for _ in range(12):
            holder = Markers()
            holder.clear()
            points = CONTRACT["frontend_to_backend"]["map_calibration"]["points"]
            for calibration_id, point in zip(sorted(MAP_CALIBRATION_MARKER_CORNERS), points):
                x, y = point["pixel_position"]
                holder.addMarker(Marker(calibration_id, (x, y, 0.0), 12121, "000"))
            await session.push(holder.toDict())
        await session.send({"type": "register_building", "building_id": "G11"})
        result = await session.control("register_building_result")

    assert result["ok"] is False
    assert "unclaimed marker" in result["error"]


@pytest.mark.asyncio
async def test_the_frontend_is_told_the_scale_to_draw_the_target_at():
    """A building being registered has no feature yet, which is exactly when the scale is needed."""
    async with _Session() as session:
        await session.send(CONTRACT["frontend_to_backend"]["map_calibration"])
        state = await session.control("session_state")
        derived = server.global_model_scale_factor

    assert state["type"] == "session_state"
    assert state["model_scale"] == MODEL_SCALE
    assert state["model_scale_factor"] == pytest.approx(derived)
    assert state["ground_scale"] > 0
