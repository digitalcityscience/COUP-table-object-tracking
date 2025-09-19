# CityScope object tracking

## Executive summary

This repo powers real‑time table object tracking using IR cameras (Intel RealSense). It detects ArUco markers, calibrates multi‑camera views to a common table coordinate system, stitches cameras into a unified top‑down view, analyzes camera distortion/positioning quality, and streams tracking data to a client (e.g., Unity).

### What it does
- Marker detection: Finds ArUco markers per frame and maps them to table/object identities.
- **Enhanced calibration**: Uses 4 corner markers per camera with **visual orientation guides** to prevent coordinate system mismatches and image flipping.
- Stitching: Warps each camera to table coordinates and joins them into a single image (1×2 or 2×2 layout).
- Distortion analysis: Quantifies how "square" and centered the calibration is; exports annotated reports to guide camera positioning.
- Streaming: Runs a TCP server on localhost:8052, pushing fused marker updates for external clients (Unity).

### Key components
- `server.py`: Entry point; runs calibration, sets up stitching, processes streams, serves tracking data.
- `calibration_handler.py`: **Enhanced calibration system** with visual guides to prevent coordinate system issues.
- `camera_setup.py`: **Visual camera positioning** with orientation arrows and corner labels.
- `save_calibration_markers.py`: Marker detection with **position data preservation** to maintain coordinate system integrity.
- `show_camera_streams.py`: **Enhanced camera streams** with visual orientation guides.
- `camera_stitching.py`: Computes perspective transforms, normalizes scale, and stitches camera views.
- `distortion_analysis.py`: Rates calibration quality and exports per‑camera and summary reports.
- `detection.py`, `hud.py`, `tracker.py`: Marker detection, on‑screen HUD, and tracking utilities.

### Outputs you get
Exported to `calibration_visualizations/`:
- Per‑camera calibration visuals: `calibration_visualizations/camera_XXX_markers.png`
- Combined calibration view: `calibration_visualizations/combined_calibration.png`
- Distortion reports: `calibration_visualizations/camera_XXX_distortion_analysis.png`, `calibration_visualizations/distortion_summary.png`
- Stitched result sample: `calibration_visualizations/stitched_sample.png`

## 🆕 **Enhanced Calibration System**

### **Problem Solved: Coordinate System Mismatches**
Previous versions could suffer from image flipping issues due to coordinate system mismatches between physical marker placement and camera image coordinates. The enhanced calibration system prevents this by:

- **Visual orientation guides**: Shows TOP/BOTTOM/LEFT/RIGHT arrows in each camera view
- **Corner position labels**: Clearly marks TOP-LEFT, TOP-RIGHT, etc. in camera images  
- **Interactive guidance**: Step-by-step setup with visual feedback for each camera
- **Position data preservation**: Ensures camera position information is saved correctly
- **Coordinate system validation**: Verifies proper mapping between physical and image coordinates

### **Visual Calibration Features**
- 🎯 **Orientation arrows**: Green arrows show TOP/BOTTOM, blue arrows show LEFT/RIGHT
- 📍 **Corner labels**: Yellow labels mark each corner position in the camera view
- 📷 **Individual camera setup**: Each camera configured separately with visual guidance
- ✅ **Real-time validation**: Shows detected markers overlaid on orientation guides
- 💾 **Data integrity**: Verifies camera position data is properly saved

## How to use it

### 1) Install dependencies
```
pip install -r requirements.txt
```

### 2) Run the server
```
python server.py
```
- **NEW**: Enhanced calibration with visual orientation guides runs automatically
- Performs/loads calibration, initializes stitching, and starts streaming on `localhost:8052`.
- A debug monitor shows stitched output and detected markers.

### 3) Enhanced Calibration Process
When you first run the server (or choose to recalibrate):

1. **Camera Overview**: All cameras shown together with orientation guides
2. **Individual Camera Setup**: Each camera configured separately:
   - Visual orientation guide shows coordinate system
   - User selects camera position (top_left, top_right, etc.)
   - User identifies marker IDs in each corner
3. **Marker Detection**: System detects and saves marker positions
4. **Validation**: Verifies coordinate system integrity

### 4) Connect your client
- Unity (COUP‑TangibleTable) or any TCP client can consume updates from `localhost:8052`.
- Use `client.py` for basic testing of the TCP connection.

### 5) Inspect calibration and quality
- Check `calibration_visualizations/` for:
  - `camera_XXX_markers.png`, `combined_calibration.png`
  - Distortion analysis: `camera_XXX_distortion_analysis.png`, `distortion_summary.png`

### Optional utilities
- Run distortion analysis standalone (on `calibration_markers.json`):
```
python distortion_analysis.py
```
- Test distortion analysis with sample data:
```
python test_distortion_analysis.py
```
- Explore stitching/processing logic in `camera_stitching.py`; calibration flow in `save_calibration_markers.py`.

### Tips
- **NEW**: The enhanced calibration system prevents most coordinate system issues automatically
- Keep the camera centered over the table; aim for "Good" (≥0.90) or "Excellent" (≥0.95) distortion scores.
- If the table/camera moves, re‑run calibration to refresh transforms and quality reports.
- **NEW**: If you experience image flipping, recalibrate with the enhanced visual guides.

## How to run
- Install requirements
- Run server.py 
- **NEW**: Follow the enhanced calibration process with visual orientation guides
- Connect to the opened websocket to read results. (By running the COUP-TangibleTable unity project)
- A visual debug monitor will show you what the camera sees and which markers are recognized.

Depending on your light conditions you might need to choose different settings for exposure and gain in the `realsense/realsense_device_manager.py`.

---

## 🎯 **Enhanced Calibration and Coordinate System**

The enhanced calibration system addresses coordinate system mismatches that could cause image flipping issues in previous versions.

### **Visual Orientation System**
During calibration, each camera shows:
- **Green arrows**: TOP (↑) and BOTTOM (↓) directions in image coordinates
- **Blue arrows**: LEFT (←) and RIGHT (→) directions in image coordinates  
- **Yellow labels**: Corner position names (TOP-LEFT, TOP-RIGHT, etc.)
- **Real-time markers**: Detected ArUco markers with their IDs

### **Coordinate System Mapping**
- **Image coordinates**: Origin (0,0) at top-left, X increases right, Y increases down
- **Physical coordinates**: Origin (0,0) at physical top-left of table, X increases right, Y increases down
- **Proper mapping**: Enhanced system ensures correct correspondence between image and physical coordinates

### **Position Data Preservation**
The system now ensures that camera position information (top_left, top_right, etc.) is properly saved in the calibration file, preventing layout detection failures that could cause coordinate system issues.

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
- `stitched_sample.png`: Sample of stitched camera output (generated when server runs)

### Examples (exported)
Detected calibration markers:

![Camera 000 markers](calibration_visualizations/camera_000_markers.png)
![Camera 001 markers](calibration_visualizations/camera_001_markers.png)

Combined calibration view:

![Combined calibration](calibration_visualizations/stitched_sample.png)

Distortion reports (appear after you run calibration or `distortion_analysis.py`):

![Distortion summary](calibration_visualizations/distortion_summary.png)
![Camera 000 distortion](calibration_visualizations/camera_000_distortion_analysis.png)
![Camera 001 distortion](calibration_visualizations/camera_001_distortion_analysis.png)

To run the analysis standalone:
```
python distortion_analysis.py
```

---

## Stitching Process
The stitching pipeline processes each camera stream with enhancement (sharpening/rotation) and perspective transforms, then joins them into a unified view.

The system automatically:
1. Loads calibration from `calibration_markers.json` **with preserved position data**
2. Applies perspective transforms to align each camera to table coordinates
3. Normalizes scales and stitches cameras into a single top-down view **with correct orientation**
4. Runs real-time marker detection on the stitched result

### Detailed Stitching Pipeline

The multi-camera stitching system (`camera_stitching.py`) transforms individual camera feeds into a unified top-down table view through **two distinct phases**:

## 🔧 **PHASE 1: ONE-TIME SETUP** (at application startup)
## ⚡ **PHASE 2: REAL-TIME PROCESSING** (applied to every frame)

---

### 🔧 **PHASE 1: ONE-TIME SETUP** (Happens ONCE at startup)

#### **1. Load Calibration Data** 
- **Reads `calibration_markers.json`**: Contains pixel positions of 4 ArUco markers per camera (detected during initial calibration)
- **NEW**: **Includes camera position data** to ensure proper layout detection
- **No live camera processing**: Uses pre-recorded marker positions from calibration file

#### **2. Calculate Perspective Transforms** (`calculate_perspective_transform()`)
- **Computes transformation matrices**: Calls `cv2.getPerspectiveTransform(src_points, dst_points)` at **line 59**
- **One matrix per camera**: Each camera gets its own 3×3 transformation matrix
- **Stored for reuse**: These matrices are calculated ONCE and saved in memory

#### **3. Pre-calculate Output Dimensions**
- **Maps physical to pixels**: 15 pixels/cm scale factor applied to table measurements
- **Determines unified scale**: All cameras normalized to same pixel dimensions
- **NEW**: **Enhanced layout analysis**: Uses preserved camera position data for correct arrangement

**🎯 Result**: Setup returns a configuration object with pre-calculated transforms, dimensions, and layout - **NO MORE CALCULATIONS NEEDED**

<details>
<summary><strong>🔍 Click to see HOW the transformation matrices are calculated</strong></summary>

**How the Transformation Matrix is Created:**

The transformation matrix is calculated using OpenCV's `cv2.getPerspectiveTransform()` which maps 4 source points to 4 destination points.

1. **Source Points** (from camera view): The actual pixel positions where ArUco markers were detected
2. **Destination Points** (target rectangle): Perfect rectangle coordinates where we want those markers to appear

**Example with Real Data from Camera 000:**
```python
# Source points (distorted quadrilateral from camera):
src_points = [
    [307.07, 31.44],    # top_left marker in camera image
    [1033.14, 27.77],   # top_right marker in camera image  
    [1033.47, 762.29],  # bottom_right marker in camera image
    [311.40, 756.73]    # bottom_left marker in camera image
]

# Destination points (perfect rectangle):
# Physical table: 74cm × 74cm, scale factor: 15 pixels/cm = 1110×1110 image
# Markers are 3cm from edges, so 45 pixels offset
dst_points = [
    [45, 45],           # top_left: 3cm × 15px/cm from edges
    [1065, 45],         # top_right: (74-3)cm × 15px/cm from left edge
    [1065, 1065],       # bottom_right: (74-3)cm × 15px/cm from both edges  
    [45, 1065]          # bottom_left: 3cm × 15px/cm from right, (74-3)cm from top
]

# OpenCV calculates the 3×3 transformation matrix:
transform_matrix = cv2.getPerspectiveTransform(src_points, dst_points)  # ← LINE 59
```

**What the Matrix Does:**
- **Corrects perspective distortion**: The camera sees a trapezoid/quadrilateral, matrix transforms it to a perfect rectangle
- **Corrects rotation**: If the camera is rotated relative to the table, the matrix straightens it
- **Corrects scale**: Maps the physical 74×74cm table area to exactly 1110×1110 pixels (15 pixels/cm)
- **Corrects position**: Centers the table area in the output image with proper 3cm borders
- **NEW**: **Preserves coordinate system**: Enhanced calibration ensures proper orientation mapping

</details>

**Code reference**: `setup_camera_transforms()` calls `calculate_perspective_transform()` in `camera_stitching.py:12-60`

---

### ⚡ **PHASE 2: REAL-TIME PROCESSING** (Applied to EVERY frame at ~30 FPS)

Now the pre-calculated transforms are applied to live camera streams:

#### **1. Frame Preprocessing** (`process_and_join_streams()`)
Each incoming camera frame undergoes enhancement:
- **Image sharpening**: Applies a 3×3 convolution kernel `[[-1,-1,-1],[-1,9,-1],[-1,-1,-1]]` to enhance edge definition
- **Buffer conversion**: Transforms RealSense buffer format to OpenCV-compatible numpy arrays
- **Note**: Despite the function name `sharpen_and_rotate_image()`, it currently only performs sharpening - no rotation is applied

**Code reference**: `sharpen_and_rotate_image()` in `image.py:16-19`

#### **2. Apply Pre-calculated Perspective Transforms** 
**🚨 KEY POINT**: The transformation matrices were calculated ONCE during setup. Now we just apply them to each frame:

```python
# This happens for EVERY frame (fast matrix multiplication):
transformed = cv2.warpPerspective(processed_frame, transforms[camera_id], output_sizes[camera_id])
```

**Code reference**: `process_and_join_streams()` in `camera_stitching.py:268-275`

#### **3. Scale Normalization & Image Stitching**
- **Fast resize**: Each transformed image is resized to unified dimensions using pre-calculated values
- **Frame buffering**: Collect frames from all cameras before stitching to ensure synchronization
- **NEW**: **Correct layout-based joining**: Uses preserved camera position data for proper image arrangement

**Code reference**: `process_and_join_streams()` in `camera_stitching.py:276-291`

#### **Performance Summary:**
- **Setup**: Heavy calculations done ONCE (perspective transforms, dimensions, layout)  
- **Runtime**: Only fast operations per frame (matrix multiplication, resize, image stacking)
- **Result**: ~30 FPS real-time stitching performance **with correct orientation**

---

## 📊 **Image Stitching Details** (`create_final_stitched_image()`)
The system intelligently combines camera feeds based on detected layout:

**Enhanced Layout Detection**:
- **2×2 grid**: Full table coverage with cameras in all four positions
- **1×2 horizontal**: Cameras only in top row OR bottom row
- **NEW**: **Correct position mapping**: Uses preserved camera position data instead of guessing
- **Fallback handling**: Uses black placeholders for missing camera positions (defensive programming)

**Joining Process**:
- **Horizontal joining**: `join_images_horizontally()` combines left/right camera pairs using `np.hstack()`
- **Vertical stacking**: `join_images_vertically()` stacks top/bottom rows using `np.vstack()`
- **Size consistency**: Ensures all images have matching dimensions before joining
- **NEW**: **Correct orientation**: No more accidental image flipping due to coordinate system issues

**Code reference**: 
- Layout logic: `create_final_stitched_image()` in `camera_stitching.py:185-225`
- Joining functions: `camera_stitching.py:135-183`

#### ⚡ **6. Real-time Performance Optimization**
- **One-time setup**: All perspective transforms calculated during initialization, not per-frame
- **Frame buffering**: Collects frames from all cameras before stitching to ensure synchronization
- **Efficient processing**: Matrix operations (warpPerspective, resize, hstack/vstack) optimized for ~30 FPS performance

#### 📊 **7. Quality Assurance**
The system includes comprehensive quality monitoring:
- **Distortion analysis**: Validates that calibration markers form proper squares with 90° angles
- **Positioning feedback**: Provides scores for camera centering, uniformity, and alignment
- **Visual exports**: Generates annotated images showing calibration quality and stitching results
- **NEW**: **Coordinate system validation**: Verifies proper mapping between physical and image coordinates

**Code reference**: `distortion_analysis.py` and quality metrics in `analyze_camera_distortion()`

## 🔄 **Complete Data Flow**

### **SETUP PHASE** (Once at startup):
```
Enhanced Camera Setup → calibration_markers.json → Calculate Transform Matrices → Store in Memory
        ↓                        ↓
Visual Orientation Guides    Position Data Preserved
```

### **RUNTIME PHASE** (Every frame at 30 FPS):
```
Camera 000 (1280×720) → Sharpen → Apply Stored Matrix → Resize → ┐
                                                                  ├→ Correct Stitching → Final Image
Camera 001 (1280×720) → Sharpen → Apply Stored Matrix → Resize → ┘
                                            ↑
                                   Proper Layout Detection
```

**🚨 Key Point**: The expensive `cv2.getPerspectiveTransform()` calculation happens ONLY during setup. At runtime, we just use `cv2.warpPerspective()` with the pre-calculated matrices - this is why we achieve 30 FPS performance! **NEW**: Enhanced calibration ensures the matrices are calculated with correct coordinate system mapping.

The result is a seamless top-down view where each camera's coverage area is geometrically corrected and properly aligned with the physical table coordinates **without image flipping issues**.

---

### Sources
- https://github.com/IntelRealSense/librealsense/tree/master/wrappers/python
- [realsense API](https://intelrealsense.github.io/librealsense/python_docs/_generated/pyrealsense2.html#module-pyrealsense2)
- [aruco opencv](https://docs.opencv.org/4.x/d9/d6a/group__aruco.html#gab9159aa69250d8d3642593e508cb6baa)
