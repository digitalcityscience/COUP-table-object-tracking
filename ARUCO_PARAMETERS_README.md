# ArUco Detection Parameters Guide

This guide explains the ArUco detection parameters used in our optimization script, specifically tailored for **30×30 pixel markers on 2400×1200 images**.

## Parameter Overview

ArUco marker detection involves multiple stages: candidate detection, polygon approximation, and corner refinement. Each parameter controls a specific aspect of this process.

## Detailed Parameter Explanations

### 1. Marker Size Parameters

#### `minMarkerPerimeterRate` & `maxMarkerPerimeterRate`
```python
'minMarkerPerimeterRate': [0.008, 0.010, 0.012, 0.015, 0.018, 0.020, 0.025]
'maxMarkerPerimeterRate': [0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
```

**Purpose**: Defines the expected size range of markers as a fraction of the total image perimeter.

**Calculation**: `marker_perimeter / image_perimeter`

**For our setup**:
- Marker perimeter: 30×4 = 120 pixels
- Image perimeter: (2400+1200)×2 = 7200 pixels  
- Theoretical rate: 120/7200 = **0.0167**

**Optimization**: Values are centered around 0.0167 to precisely target our marker size.

**Effects**:
- **Too low**: Misses legitimate markers
- **Too high**: Includes noise and false positives
- **Optimal range**: Tight bounds around theoretical value for efficiency

---

### 2. Shape Detection Parameters

#### `polygonalApproxAccuracyRate`
```python
'polygonalApproxAccuracyRate': [0.015, 0.02, 0.025, 0.03, 0.035, 0.04]
```

**Purpose**: Controls how precisely detected contours must approximate a square/rectangle.

**Range**: 0.01 (very strict) to 0.1 (very loose)

**For small markers**: Tighter range (0.015-0.04) because:
- Small markers have fewer pixels for shape definition
- Precision is critical for accurate detection
- Noise has proportionally larger impact

**Effects**:
- **Too low**: Rejects slightly imperfect but valid squares
- **Too high**: Accepts non-square shapes as markers
- **Optimal**: Balance between precision and tolerance

---

### 3. Contrast and Thresholding Parameters

#### `minOtsuStdDev`
```python
'minOtsuStdDev': [2.0, 3.0, 4.0, 5.0, 6.0]
```

**Purpose**: Minimum standard deviation required for Otsu thresholding to be applied.

**Function**: Ensures sufficient contrast between marker and background before attempting binarization.

**Range explanation**:
- **2.0-3.0**: Good lighting, high contrast
- **4.0-5.0**: Normal indoor lighting
- **6.0+**: Poor lighting conditions

**Effects**:
- **Too low**: Processes low-contrast regions as potential markers (noise)
- **Too high**: Skips markers in suboptimal lighting
- **Optimal**: Matches your typical lighting conditions

---

### 4. Adaptive Thresholding Window Parameters

#### `adaptiveThreshWinSizeMin` & `adaptiveThreshWinSizeMax`
```python
'adaptiveThreshWinSizeMin': [3, 5, 7, 9]
'adaptiveThreshWinSizeMax': [15, 19, 23, 31, 39]
```

**Purpose**: Defines the range of sliding window sizes used for adaptive thresholding.

**Critical relationship to marker size**:
- Window should be **smaller than marker** but **large enough** to capture local features
- For 30px markers: windows from 3px (fine details) to 39px (marker-scale features)

**Window size guidelines**:
- **Min (3-9px)**: Captures fine marker details and edges
- **Max (15-39px)**: Covers marker features without including too much background

**Effects**:
- **Too small**: Noisy, misses larger patterns
- **Too large**: Loses local contrast, includes irrelevant background
- **Optimal range**: Covers marker scale without exceeding it significantly

#### `adaptiveThreshWinSizeStep`
```python
'adaptiveThreshWinSizeStep': [2, 4, 6, 8]
```

**Purpose**: Increment between tested window sizes (from min to max).

**Function**: Algorithm tests multiple window sizes: min, min+step, min+2×step, ..., max

**Optimization**:
- Steps of 2-8 provide good coverage for 30px markers
- Smaller steps = more thorough but slower
- Larger steps = faster but might miss optimal size

**Example**: With min=3, max=23, step=4: tests windows of 3, 7, 11, 15, 19, 23 pixels

---

### 5. Threshold Fine-tuning

#### `adaptiveThreshConstant`
```python
'adaptiveThreshConstant': [5, 7, 9, 11]
```

**Purpose**: Constant subtracted from the computed adaptive threshold value.

**Function**: Fine-tunes binarization sensitivity after local threshold calculation.

**Range selection**:
- **5-7**: More sensitive, includes more pixels as "marker"
- **9-11**: Less sensitive, stricter about what constitutes "marker"

**Effects**:
- **Too low**: Includes too much noise in thresholded image
- **Too high**: Marker features may be lost or fragmented
- **Optimal**: Balances marker preservation with noise rejection

---

### 6. Corner Refinement

#### `cornerRefinementMethod`
```python
'cornerRefinementMethod': [
    aruco.CORNER_REFINE_NONE,      # No refinement
    aruco.CORNER_REFINE_SUBPIX,    # Subpixel accuracy  
    aruco.CORNER_REFINE_CONTOUR    # Contour-based refinement
]
```

**Purpose**: Method for refining detected marker corner positions.

**Options**:
- **`CORNER_REFINE_NONE`**: 
  - Fastest processing
  - Uses initial corner detection
  - Least accurate
  
- **`CORNER_REFINE_SUBPIX`**: 
  - Good balance of speed and accuracy
  - Refines to subpixel precision
  - Suitable for most applications
  
- **`CORNER_REFINE_CONTOUR`**: 
  - Most accurate corner detection
  - Analyzes marker contour shape
  - Slowest but best for small markers

**For small markers**: `CORNER_REFINE_CONTOUR` often performs best because:
- Precision matters more with fewer pixels
- Small errors have larger relative impact
- Contour analysis helps with noisy edges

---

## Parameter Interaction Effects

### Critical Combinations

1. **Perimeter Rate + Window Size**: 
   - Must be proportionally matched
   - Small markers need small windows
   - Large perimeter rates with small windows = missed detections

2. **Polygon Accuracy + Corner Refinement**:
   - Strict accuracy works better with good corner refinement
   - Loose accuracy can compensate for basic corner detection

3. **Threshold Constant + Window Size**:
   - Larger windows may need higher constants
   - Smaller windows work with lower constants

### Common Issues

- **No detections**: Usually perimeter rates too narrow or window sizes inappropriate
- **False positives**: Perimeter rates too wide or polygon accuracy too loose  
- **Inconsistent detection**: Threshold parameters not matching lighting conditions

---

## Optimization Strategy for 30×30px Markers

### 1. **Use Option 4**: "Targeted test for 30x30px markers"
- Only 288 parameter combinations
- Focused on your exact marker size
- Fastest path to optimal settings

### 2. **Parameter Priority Order**:
1. **Perimeter rates** (most critical for size matching)
2. **Window sizes** (critical for 30px scale)
3. **Polygon accuracy** (important for small markers)
4. **Threshold parameters** (fine-tuning for your lighting)
5. **Corner refinement** (quality vs. speed trade-off)

### 3. **Expected Optimal Ranges**:
- `minMarkerPerimeterRate`: ~0.015-0.020
- `maxMarkerPerimeterRate`: ~0.2-0.3  
- `adaptiveThreshWinSizeMax`: ~19-31
- `polygonalApproxAccuracyRate`: ~0.02-0.03
- `cornerRefinementMethod`: `CORNER_REFINE_CONTOUR`

---

## Usage Notes

- **Test incrementally**: Start with Option 4, then Option 3 if needed
- **Lighting matters**: Re-test if lighting conditions change significantly
- **Multiple markers**: Optimal parameters work for all markers of the same size
- **Performance**: Tighter parameter ranges = faster detection in production

---

## Theoretical Calculations

For any marker size on any image resolution:

```python
# Calculate theoretical perimeter rate
marker_side_pixels = 30  # Your marker size
image_width = 2400       # Your image width  
image_height = 1200      # Your image height

marker_perimeter = marker_side_pixels * 4
image_perimeter = (image_width + image_height) * 2
theoretical_rate = marker_perimeter / image_perimeter

print(f"Theoretical perimeter rate: {theoretical_rate:.4f}")
# Output: Theoretical perimeter rate: 0.0167
```

Use this calculation to optimize parameters for different marker sizes or image resolutions. 