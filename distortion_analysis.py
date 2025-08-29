import cv2
import numpy as np
import json
import os
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Dict, List, Tuple

def analyze_camera_distortion(cameras_config: Dict, output_dir: str = "calibration_visualizations") -> Dict:
    """
    Analyze camera distortion by checking if calibration markers form a proper square
    with equidistant spacing from the image center.
    
    Args:
        cameras_config: Dictionary containing camera configurations with pixel positions
        output_dir: Directory to save distortion analysis images
    
    Returns:
        Dictionary containing distortion analysis results for each camera
    """
    print("=== Analyzing Camera Distortion ===")
    distortion_results = {}
    
    for camera_id, camera_config in cameras_config.items():
        if "calibration_markers" not in camera_config:
            print(f"No calibration markers found for camera {camera_id}")
            continue
            
        markers = camera_config["calibration_markers"]
        
        # Extract pixel positions
        positions = {}
        for position_name, marker_info in markers.items():
            if marker_info.get("pixel_position") is None:
                print(f"Warning: Camera {camera_id} marker {position_name} has no pixel position")
                continue
            positions[position_name] = marker_info["pixel_position"]
        
        if len(positions) != 4:
            print(f"Warning: Camera {camera_id} doesn't have all 4 marker positions")
            continue
            
        # Calculate distances and analyze distortion
        analysis = analyze_marker_geometry(positions, camera_id)
        distortion_results[camera_id] = analysis
        
        # Create distortion visualization
        create_distortion_visualization(positions, analysis, camera_id, output_dir)
    
    # Create summary report
    create_distortion_summary(distortion_results, output_dir)
    
    return distortion_results

def analyze_marker_geometry(positions: Dict, camera_id: str) -> Dict:
    """
    Analyze the geometry of marker positions to detect distortion
    
    Args:
        positions: Dictionary of position_name -> [x, y] coordinates
        camera_id: Camera identifier for logging
    
    Returns:
        Dictionary containing distortion analysis metrics
    """
    # Extract coordinates
    tl = np.array(positions["top_left"])      # top-left
    tr = np.array(positions["top_right"])     # top-right  
    br = np.array(positions["bottom_right"])  # bottom-right
    bl = np.array(positions["bottom_left"])   # bottom-left
    
    # Calculate image center (assuming markers are centered in image)
    image_center = np.mean([tl, tr, br, bl], axis=0)
    
    # Calculate distances from center to each marker
    distances_from_center = {
        "top_left": np.linalg.norm(tl - image_center),
        "top_right": np.linalg.norm(tr - image_center),
        "bottom_right": np.linalg.norm(br - image_center),
        "bottom_left": np.linalg.norm(bl - image_center)
    }
    
    # Calculate side lengths of the quadrilateral
    side_lengths = {
        "top": np.linalg.norm(tr - tl),        # top side
        "right": np.linalg.norm(br - tr),      # right side
        "bottom": np.linalg.norm(bl - br),     # bottom side
        "left": np.linalg.norm(tl - bl)        # left side
    }
    
    # Calculate diagonal lengths
    diagonal_lengths = {
        "main": np.linalg.norm(br - tl),       # top-left to bottom-right
        "anti": np.linalg.norm(bl - tr)        # top-right to bottom-left
    }
    
    # Calculate angles of the quadrilateral
    def calculate_angle(p1, vertex, p3):
        """Calculate angle at vertex between lines to p1 and p3"""
        v1 = p1 - vertex
        v2 = p3 - vertex
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1, 1)  # Handle numerical errors
        return math.degrees(math.acos(cos_angle))
    
    angles = {
        "top_left": calculate_angle(bl, tl, tr),
        "top_right": calculate_angle(tl, tr, br), 
        "bottom_right": calculate_angle(tr, br, bl),
        "bottom_left": calculate_angle(br, bl, tl)
    }
    
    # Calculate distortion metrics
    
    # 1. Distance from center uniformity (should all be equal for perfect centering)
    center_distances = list(distances_from_center.values())
    center_distance_mean = np.mean(center_distances)
    center_distance_std = np.std(center_distances)
    center_uniformity_score = 1.0 - (center_distance_std / center_distance_mean)  # 1.0 = perfect
    
    # 2. Side length uniformity (should all be equal for perfect square)
    sides = list(side_lengths.values())
    side_length_mean = np.mean(sides)
    side_length_std = np.std(sides)
    side_uniformity_score = 1.0 - (side_length_std / side_length_mean)  # 1.0 = perfect
    
    # 3. Diagonal length uniformity (should be equal for perfect square)
    diagonals = list(diagonal_lengths.values())
    diagonal_ratio = min(diagonals) / max(diagonals)  # 1.0 = perfect
    
    # 4. Angle uniformity (should all be 90 degrees for perfect square)
    angle_values = list(angles.values())
    angle_deviations = [abs(angle - 90.0) for angle in angle_values]
    max_angle_deviation = max(angle_deviations)
    angle_uniformity_score = max(0, 1.0 - (max_angle_deviation / 45.0))  # 1.0 = perfect, 0 = 45+ deg off
    
    # 5. Overall distortion score (weighted average)
    overall_score = (
        center_uniformity_score * 0.3 +    # 30% weight for center distance uniformity
        side_uniformity_score * 0.3 +      # 30% weight for side length uniformity  
        diagonal_ratio * 0.2 +              # 20% weight for diagonal ratio
        angle_uniformity_score * 0.2        # 20% weight for angle uniformity
    )
    
    # Determine distortion level
    if overall_score >= 0.95:
        distortion_level = "Excellent"
    elif overall_score >= 0.90:
        distortion_level = "Good"
    elif overall_score >= 0.80:
        distortion_level = "Fair"
    elif overall_score >= 0.70:
        distortion_level = "Poor"
    else:
        distortion_level = "Very Poor"
    
    print(f"Camera {camera_id} distortion analysis:")
    print(f"  Overall score: {overall_score:.3f} ({distortion_level})")
    print(f"  Center uniformity: {center_uniformity_score:.3f}")
    print(f"  Side uniformity: {side_uniformity_score:.3f}")
    print(f"  Diagonal ratio: {diagonal_ratio:.3f}")
    print(f"  Angle uniformity: {angle_uniformity_score:.3f}")
    
    return {
        "camera_id": camera_id,
        "image_center": image_center.tolist(),
        "distances_from_center": distances_from_center,
        "side_lengths": side_lengths,
        "diagonal_lengths": diagonal_lengths,
        "angles": angles,
        "center_uniformity_score": center_uniformity_score,
        "side_uniformity_score": side_uniformity_score,
        "diagonal_ratio": diagonal_ratio,
        "angle_uniformity_score": angle_uniformity_score,
        "overall_score": overall_score,
        "distortion_level": distortion_level,
        "recommendations": generate_distortion_recommendations(overall_score, center_uniformity_score, side_uniformity_score, angle_uniformity_score)
    }

def generate_distortion_recommendations(overall_score: float, center_score: float, side_score: float, angle_score: float) -> List[str]:
    """Generate recommendations based on distortion analysis"""
    recommendations = []
    
    if overall_score < 0.80:
        recommendations.append("Camera position needs adjustment")
        
        if center_score < 0.90:
            recommendations.append("• Markers not equidistant from center - check camera centering")
        
        if side_score < 0.90:
            recommendations.append("• Uneven side lengths detected - check camera height/angle")
            
        if angle_score < 0.90:
            recommendations.append("• Non-square angles detected - check camera tilt/rotation")
            
        recommendations.append("• Try adjusting camera position slightly")
        recommendations.append("• Ensure table hasn't been moved or tilted")
    
    elif overall_score < 0.95:
        recommendations.append("Good calibration with minor distortion")
        recommendations.append("• Consider minor position adjustments for optimal performance")
    
    else:
        recommendations.append("Excellent calibration quality")
        recommendations.append("• Camera position is optimal")
    
    return recommendations

def create_distortion_visualization(positions: Dict, analysis: Dict, camera_id: str, output_dir: str):
    """Create a visual representation of distortion analysis"""
    # Extract coordinates
    tl = np.array(positions["top_left"])
    tr = np.array(positions["top_right"])
    br = np.array(positions["bottom_right"])
    bl = np.array(positions["bottom_left"])
    center = np.array(analysis["image_center"])
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Plot 1: Marker positions with distortion indicators
    ax1.set_title(f'Camera {camera_id} - Marker Positions & Distortion')
    ax1.set_aspect('equal')
    
    # Plot markers
    markers = [tl, tr, br, bl]
    labels = ['TL', 'TR', 'BR', 'BL']
    colors = ['red', 'green', 'blue', 'orange']
    
    for i, (marker, label, color) in enumerate(zip(markers, labels, colors)):
        ax1.plot(marker[0], marker[1], 'o', color=color, markersize=10)
        ax1.annotate(label, (marker[0], marker[1]), xytext=(5, 5), textcoords='offset points')
    
    # Draw the quadrilateral
    quad = patches.Polygon([tl, tr, br, bl], fill=False, edgecolor='black', linewidth=2)
    ax1.add_patch(quad)
    
    # Draw center point and distances
    ax1.plot(center[0], center[1], 'x', color='purple', markersize=12, markeredgewidth=3)
    ax1.annotate('Center', center, xytext=(5, 5), textcoords='offset points')
    
    # Draw distance lines to center
    for marker, label, color in zip(markers, labels, colors):
        ax1.plot([center[0], marker[0]], [center[1], marker[1]], '--', color=color, alpha=0.5)
        
        # Add distance text
        mid_point = (center + marker) / 2
        distance = analysis["distances_from_center"][{"TL": "top_left", "TR": "top_right", "BR": "bottom_right", "BL": "bottom_left"}[label]]
        ax1.annotate(f'{distance:.1f}px', mid_point, ha='center', fontsize=8, 
                    bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.7))
    
    # Invert y-axis to match image coordinates
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('X (pixels)')
    ax1.set_ylabel('Y (pixels)')
    
    # Plot 2: Distortion metrics
    ax2.set_title(f'Camera {camera_id} - Distortion Analysis')
    
    # Create bar chart of distortion metrics
    metrics = ['Center\nUniformity', 'Side\nUniformity', 'Diagonal\nRatio', 'Angle\nUniformity', 'Overall\nScore']
    scores = [
        analysis['center_uniformity_score'],
        analysis['side_uniformity_score'], 
        analysis['diagonal_ratio'],
        analysis['angle_uniformity_score'],
        analysis['overall_score']
    ]
    
    # Color bars based on score quality
    bar_colors = []
    for score in scores:
        if score >= 0.95:
            bar_colors.append('green')
        elif score >= 0.90:
            bar_colors.append('lightgreen')
        elif score >= 0.80:
            bar_colors.append('yellow')
        elif score >= 0.70:
            bar_colors.append('orange')
        else:
            bar_colors.append('red')
    
    bars = ax2.bar(metrics, scores, color=bar_colors, alpha=0.7)
    ax2.set_ylim(0, 1.0)
    ax2.set_ylabel('Quality Score (1.0 = Perfect)')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add score values on bars
    for bar, score in zip(bars, scores):
        ax2.annotate(f'{score:.3f}', (bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01),
                    ha='center', va='bottom', fontweight='bold')
    
    # Add overall assessment
    fig.suptitle(f'Camera {camera_id} - Distortion Level: {analysis["distortion_level"]} (Score: {analysis["overall_score"]:.3f})', 
                 fontsize=14, fontweight='bold')
    
    # Add recommendations text
    recommendations_text = '\n'.join(analysis['recommendations'])
    fig.text(0.02, 0.02, f'Recommendations:\n{recommendations_text}', 
             fontsize=9, verticalalignment='bottom',
             bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.3))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make room for recommendations
    
    # Save the visualization
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"camera_{camera_id}_distortion_analysis.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved distortion analysis for camera {camera_id} to {output_path}")

def create_distortion_summary(distortion_results: Dict, output_dir: str):
    """Create a summary report of distortion analysis for all cameras"""
    if not distortion_results:
        print("No distortion results to summarize")
        return
    
    # Create summary figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Overall scores comparison
    camera_ids = list(distortion_results.keys())
    overall_scores = [distortion_results[cam_id]['overall_score'] for cam_id in camera_ids]
    distortion_levels = [distortion_results[cam_id]['distortion_level'] for cam_id in camera_ids]
    
    # Color bars based on distortion level
    level_colors = {
        'Excellent': 'green',
        'Good': 'lightgreen', 
        'Fair': 'yellow',
        'Poor': 'orange',
        'Very Poor': 'red'
    }
    bar_colors = [level_colors[level] for level in distortion_levels]
    
    bars1 = ax1.bar(camera_ids, overall_scores, color=bar_colors, alpha=0.7)
    ax1.set_title('Overall Distortion Scores by Camera')
    ax1.set_ylabel('Overall Quality Score')
    ax1.set_ylim(0, 1.0)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add score values and levels on bars
    for bar, score, level in zip(bars1, overall_scores, distortion_levels):
        ax1.annotate(f'{score:.3f}\n({level})', 
                    (bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01),
                    ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Detailed metrics comparison
    metrics = ['Center Uniformity', 'Side Uniformity', 'Diagonal Ratio', 'Angle Uniformity']
    x = np.arange(len(camera_ids))
    width = 0.2
    
    for i, metric_key in enumerate(['center_uniformity_score', 'side_uniformity_score', 'diagonal_ratio', 'angle_uniformity_score']):
        metric_scores = [distortion_results[cam_id][metric_key] for cam_id in camera_ids]
        ax2.bar(x + i*width, metric_scores, width, label=metrics[i], alpha=0.7)
    
    ax2.set_title('Detailed Distortion Metrics by Camera')
    ax2.set_ylabel('Quality Score')
    ax2.set_xlabel('Camera ID')
    ax2.set_xticks(x + width * 1.5)
    ax2.set_xticklabels(camera_ids)
    ax2.set_ylim(0, 1.0)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save summary
    summary_path = os.path.join(output_dir, "distortion_summary.png")
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved distortion summary to {summary_path}")
    
    # Print text summary
    print("\n=== DISTORTION ANALYSIS SUMMARY ===")
    for camera_id in camera_ids:
        result = distortion_results[camera_id]
        print(f"\nCamera {camera_id}:")
        print(f"  Overall Score: {result['overall_score']:.3f} ({result['distortion_level']})")
        print(f"  Key Issues:")
        for rec in result['recommendations']:
            print(f"    {rec}")

def load_and_analyze_calibration(calibration_file: str = "calibration_markers.json", output_dir: str = "calibration_visualizations") -> Dict:
    """
    Load calibration from file and run distortion analysis
    
    Args:
        calibration_file: Path to calibration markers JSON file
        output_dir: Directory to save analysis results
    
    Returns:
        Dictionary containing distortion analysis results
    """
    if not os.path.exists(calibration_file):
        raise FileNotFoundError(f"Calibration file not found: {calibration_file}")
    
    with open(calibration_file, 'r') as f:
        cameras_config = json.load(f)
    
    return analyze_camera_distortion(cameras_config, output_dir)

if __name__ == "__main__":
    # Example usage - analyze existing calibration
    try:
        results = load_and_analyze_calibration()
        print("Distortion analysis complete! Check the calibration_visualizations/ directory for detailed reports.")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Run calibration first using find_calibration_markers.py") 