from rig_config import build_fixed_camera_setup


def test_fixed_camera_setup_matches_the_physical_rig():
    setup = build_fixed_camera_setup()

    assert setup["863"]["position"] == "top_left"
    assert setup["104"]["position"] == "top_right"
    assert [
        setup["863"]["calibration_markers"][corner]["id"]
        for corner in ("top_left", "top_right", "bottom_right", "bottom_left")
    ] == ["180", "181", "182", "183"]
    assert [
        setup["104"]["calibration_markers"][corner]["id"]
        for corner in ("top_left", "top_right", "bottom_right", "bottom_left")
    ] == ["190", "191", "192", "193"]
    assert setup["863"]["measurements"] == {
        "width": 80.0,
        "height": 80.0,
        "marker_offset": 3.0,
    }
    assert setup["104"]["calibration_markers"]["bottom_right"][
        "physical_position"
    ] == [77.0, 77.0]
