# CityScope object tracking

## How to run
- Install requirements
- Run server.py 
- Connect to the opened websocket to read results. (By running the COUP-TangibleTable unity project)
- A visual debug monitor will show you what the camera sees and which markers are recognized.

Depending on your light conditions you might need to choose different settings for exposure and gain in the `realsense/realsense_device_manager.py`.

---

## Calibration and Distortion Analysis
After calibration markers are detected, a distortion analysis is run to verify that:
- markers are equidistant from the image center
- sides and diagonals form a square
- angles are ~90°

Artifacts are exported to `calibration_visualizations/`:
- `camera_XXX_markers.png`: Detected calibration marker positions per camera
- `combined_calibration.png`: Side-by-side view of all cameras
- `camera_XXX_distortion_analysis.png`: Per-camera distortion report (generated after running analysis)
- `distortion_summary.png`: Overview comparison across cameras (generated after running analysis)

### Examples (exported)
Detected calibration markers:

![Camera 000 markers](calibration_visualizations/camera_000_markers.png)
![Camera 001 markers](calibration_visualizations/camera_001_markers.png)

Combined calibration view:

![Combined calibration](calibration_visualizations/combined_calibration.png)

Distortion reports (appear after you run calibration or `distortion_analysis.py`):

![Distortion summary](calibration_visualizations/distortion_summary.png)
![Camera 000 distortion](calibration_visualizations/camera_000_distortion_analysis.png)
![Camera 001 distortion](calibration_visualizations/camera_001_distortion_analysis.png)

To run the analysis standalone:
```
python distortion_analysis.py
```

---

## Stitching Process (overview)
The stitching pipeline processes each camera stream with enhancement and perspective transforms and then joins them into a unified view.

### Step-by-step (sample exports)
Raw IR frames:

![Original 000](original_000.png)
![Original 001](original_001.png)

Enhanced (sharpened/rotated):

![Enhanced 000](enhanced_000.png)
![Enhanced 001](enhanced_001.png)

Perspective transform results:

![Transformed 000](transformed_000.png)
![Transformed 001](transformed_001.png)

Warped views:

![Warped 000](warped_000.png)
![Warped 001](warped_001.png)

Stitching steps and results:

![Stitched step 000](stitched_step_000.png)
![Stitched step 001](stitched_step_001.png)

Final stitched outputs (examples):

![Latest stitched](latest_stitched.png)
![Latest stitched (simple)](latest_stitched_simple.png)
![Latest stitched (annotated)](latest_stitched_vis.png)
![Final output](final_output.png)
![Stitched output](stitched_output.png)
![Stitched result](stitched_result.png)

---

### Sources
- https://github.com/IntelRealSense/librealsense/tree/master/wrappers/python
- [realsense API](https://intelrealsense.github.io/librealsense/python_docs/_generated/pyrealsense2.html#module-pyrealsense2)
- [aruco opencv](https://docs.opencv.org/4.x/d9/d6a/group__aruco.html#gab9159aa69250d8d3642593e508cb6baa)
