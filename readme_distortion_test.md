Summary
I've created a camera distortion analysis system that analyzes whether the calibration markers form a proper square with equidistant spacing from the camera center. Here's what the system does:
✅ Key Features:
Distortion Analysis (distortion_analysis.py):
Center Uniformity: Checks if all markers are equidistant from the image center
Side Uniformity: Verifies all sides of the quadrilateral are equal length
Diagonal Ratio: Ensures both diagonals are equal (perfect square)
Angle Uniformity: Confirms all angles are 90 degrees
Scoring System:
Overall score from 0.0 to 1.0 (1.0 = perfect)
Weighted scoring: 30% center + 30% sides + 20% diagonals + 20% angles
Quality levels: Excellent (0.95+), Good (0.90+), Fair (0.80+), Poor (0.70+), Very Poor (<0.70)
Visual Reports:
Individual camera analysis with marker positions, distances, and quality scores
Summary comparison across all cameras
Color-coded visualizations (green=excellent, red=poor)
Actionable recommendations for camera adjustments
Integration:
Automatically runs after calibration in find_calibration_markers.py
Can be run standalone on existing calibration files
Results saved to calibration_visualizations/ directory
🎯 What It Detects:
Camera positioning issues: Off-center cameras, incorrect height/angle
Table movement: Markers should form perfect square if table hasn't moved
Camera tilt/rotation: Non-90-degree angles indicate camera misalignment
Lens distortion: Uneven side lengths suggest optical distortion
📊 Output Files:
camera_XXX_distortion_analysis.png - Individual camera analysis
distortion_summary.png - Comparison across all cameras

