# Camera Distortion Analysis System

This system analyzes camera distortion by checking if calibration markers form a proper square with equidistant spacing from the image center. It helps detect if cameras need repositioning due to table movement or other factors.

## 🎯 Purpose

Physical markers are placed at the corners of a square, with the camera positioned at the center above them. Due to camera distortion, perspective issues, or misalignment, the markers might not appear as a perfect square in the camera image. This analysis helps detect and quantify these issues.

## 📁 Files

- `distortion_analysis.py` - Main analysis functions
- `test_distortion_analysis.py` - Test script with sample data
- `find_calibration_markers.py` - Calibration (auto-runs distortion analysis)

## 🚀 Usage

### Automatic Analysis (Recommended)
```bash
# Run calibration - distortion analysis runs automatically at the end
python find_calibration_markers.py
```

### Manual Analysis on Existing Calibration
```bash
# Analyze existing calibration file
python distortion_analysis.py
```

### Test with Sample Data
```bash
# Run test with 5 sample cameras showing different distortion levels
python test_distortion_analysis.py
```

### Programmatic Usage
```python
from distortion_analysis import analyze_camera_distortion

# Load your calibration data
cameras_config = {...}  # Your calibration data

# Run analysis
results = analyze_camera_distortion(cameras_config)

# Access results
for camera_id, result in results.items():
    print(f"Camera {camera_id}: {result['overall_score']:.3f} ({result['distortion_level']})")
```

## 📊 Metrics Explained

### 🎯 Overall Score (0.0 - 1.0)
- **0.95+**: Excellent (green) - Camera position is optimal
- **0.90+**: Good (light green) - Minor adjustments recommended  
- **0.80+**: Fair (yellow) - Some distortion detected
- **0.70+**: Poor (orange) - Camera position needs adjustment
- **<0.70**: Very Poor (red) - Significant repositioning required

### 📏 Individual Metrics

**Center Uniformity (30% weight)**
- Measures if all markers are equidistant from image center
- Low score → Camera not centered over markers

**Side Uniformity (30% weight)**  
- Measures if all quadrilateral sides are equal length
- Low score → Camera height/angle issues

**Diagonal Ratio (20% weight)**
- Measures if both diagonals are equal length
- Low score → Perspective distortion

**Angle Uniformity (20% weight)**
- Measures if all angles are 90 degrees
- Low score → Camera tilt/rotation issues

## 📈 Output Files

Results are saved to `calibration_visualizations/` (or custom directory):

- `camera_XXX_distortion_analysis.png` - Individual camera analysis
- `distortion_summary.png` - Comparison across all cameras
- Console output with scores and recommendations

## 🔧 Common Issues & Solutions

| Problem | Symptoms | Solution |
|---------|----------|----------|
| **Camera off-center** | Low center uniformity | Move camera to center over markers |
| **Camera too high/low** | Low side uniformity | Adjust camera height |
| **Camera tilted** | Low angle uniformity | Level the camera mount |
| **Table moved** | All metrics poor | Realign table or recalibrate |
| **Lens distortion** | Low diagonal ratio | Check camera settings or lens |

## 📋 Requirements

```
matplotlib
numpy
opencv-contrib-python
```

Install with: `pip install -r requirements.txt`

## 🔍 Example Output

```
Camera 000 distortion analysis:
  Overall score: 0.987 (Excellent)
  Center uniformity: 0.995
  Side uniformity: 0.992
  Diagonal ratio: 0.998
  Angle uniformity: 0.976

Camera 001 distortion analysis:
  Overall score: 0.734 (Poor)
  Center uniformity: 0.845
  Side uniformity: 0.692
  Diagonal ratio: 0.756
  Angle uniformity: 0.643
```

## 🎨 Visualization Features

- **Color-coded markers**: Each corner has a different color
- **Distance lines**: Show distances from center to each marker
- **Quality bars**: Visual representation of each metric
- **Recommendations**: Actionable advice for improvements
- **Comparison charts**: Side-by-side camera performance

## 🧪 Testing

Run the test script to see the system in action:

```bash
python test_distortion_analysis.py
```

This creates 5 sample cameras with different distortion levels:
- Camera 000: Perfect calibration (Excellent)
- Camera 001: Slightly off-center (Good) 
- Camera 002: Tilted/rotated (Fair)
- Camera 003: Heavily distorted (Poor)
- Camera 004: Extreme distortion (Very Poor)

## 💡 Tips

1. **Run after any table movement** - Even small bumps can affect calibration
2. **Check regularly** - Camera mounts can shift over time
3. **Aim for 0.90+ overall score** - Good enough for most tracking applications
4. **Use recommendations** - They provide specific guidance for improvements
5. **Compare cameras** - The summary view helps identify which cameras need attention

## 🔗 Integration

The distortion analysis is automatically integrated into the calibration workflow. After running `find_calibration_markers.py`, you'll get:

1. Standard calibration marker detection
2. Automatic distortion analysis 
3. Visual reports and recommendations
4. Ready-to-use calibration data

No additional steps needed! 🎉 