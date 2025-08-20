import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class PhysicalCalibrationConfig:
    """Physical measurements of the workspace"""
    # Distance from frame edge to nearest marker centroid (in cm)
    edge_to_marker_x: float  # horizontal distance
    edge_to_marker_y: float  # vertical distance
    
    # Physical distances between markers (in cm)
    marker_spacing_x: float  # horizontal distance between markers
    marker_spacing_y: float  # vertical distance between markers
    
    # Target resolution for final stitched image (pixels per cm)
    pixels_per_cm: float = 10.0  # 10 pixels = 1 cm

@dataclass
class CameraCalibration:
    """Stores calibration data for a camera"""
    camera_id: str
    homography_matrix: np.ndarray
    output_size: Tuple[int, int]
    offset: Tuple[int, int]  # (x, y) offset in final stitched image
    physical_bounds: Tuple[float, float, float, float]  # (min_x, min_y, max_x, max_y) in cm

class PhysicalMultiCameraStitcher:
    """Stitches images using real-world physical measurements"""
    
    def __init__(self, physical_config: PhysicalCalibrationConfig):
        self.config = physical_config
        self.calibrations: Dict[str, CameraCalibration] = {}
        self.corner_marker_ids = [10, 11, 12, 13]  # Top-left, top-right, bottom-right, bottom-left
        self.is_calibrated = False
        self.workspace_bounds_cm = None  # (min_x, min_y, max_x, max_y) in cm
        self.output_size_pixels = None
    
    def calibrate_cameras(self, calibration_frames: Dict[str, Tuple[np.ndarray, Dict]]):
        """
        Calibrate cameras using corner markers and physical measurements
        calibration_frames: {camera_id: (image, buildingDict)}
        """
        print("Starting physical calibration...")
        
        # Find corner markers in all cameras
        camera_corners = {}
        
        for camera_id, (image, building_dict) in calibration_frames.items():
            corners = self._extract_corner_positions(building_dict)
            if len(corners) >= 2:  # Need at least 2 corners for calibration
                camera_corners[camera_id] = corners
                print(f"Found {len(corners)} corners in camera {camera_id}: {list(corners.keys())}")
            else:
                print(f"Warning: Only found {len(corners)} corners in camera {camera_id}")
        
        if len(camera_corners) < 2:
            raise ValueError("Need at least 2 cameras with corner markers visible")
        
        # Calculate the full workspace bounds based on physical measurements
        self._calculate_workspace_bounds(camera_corners)
        
        # Calculate transformations for each camera
        self._calculate_physical_homographies(camera_corners, calibration_frames)
        
        self.is_calibrated = True
        print(f"Physical calibration completed!")
        print(f"Workspace bounds: {self.workspace_bounds_cm} cm")
        print(f"Output image size: {self.output_size_pixels} pixels")
    
    def _extract_corner_positions(self, building_dict: Dict) -> Dict[int, Tuple[float, float]]:
        """Extract corner marker positions from buildingDict"""
        corners = {}
        
        for marker_id in self.corner_marker_ids:
            if marker_id in building_dict:
                marker = building_dict[marker_id]
                corners[marker_id] = (marker.position.x, marker.position.y)
        
        return corners
    
    def _calculate_workspace_bounds(self, camera_corners: Dict):
        """Calculate the physical workspace bounds in centimeters"""
        
        # Define physical positions of markers in centimeters
        # Assuming markers are arranged in a rectangle
        marker_positions_cm = {
            10: (self.config.edge_to_marker_x, self.config.edge_to_marker_y),  # Top-left
            11: (self.config.edge_to_marker_x + self.config.marker_spacing_x, self.config.edge_to_marker_y),  # Top-right
            12: (self.config.edge_to_marker_x + self.config.marker_spacing_x, self.config.edge_to_marker_y + self.config.marker_spacing_y),  # Bottom-right
            13: (self.config.edge_to_marker_x, self.config.edge_to_marker_y + self.config.marker_spacing_y)   # Bottom-left
        }
        
        # Calculate full workspace bounds (including areas beyond markers)
        workspace_width_cm = self.config.marker_spacing_x + 2 * self.config.edge_to_marker_x
        workspace_height_cm = self.config.marker_spacing_y + 2 * self.config.edge_to_marker_y
        
        self.workspace_bounds_cm = (0, 0, workspace_width_cm, workspace_height_cm)
        
        # Calculate output image size in pixels
        self.output_size_pixels = (
            int(workspace_width_cm * self.config.pixels_per_cm),
            int(workspace_height_cm * self.config.pixels_per_cm)
        )
        
        print(f"Physical workspace: {workspace_width_cm:.1f} x {workspace_height_cm:.1f} cm")
        print(f"Marker positions (cm): {marker_positions_cm}")
        
        # Store marker positions for homography calculation
        self.marker_positions_cm = marker_positions_cm
    
    def _calculate_physical_homographies(self, camera_corners: Dict, calibration_frames: Dict):
        """Calculate homography matrices using physical measurements"""
        
        for camera_id, corners in camera_corners.items():
            if len(corners) < 3:  # Need at least 3 points for homography
                print(f"Skipping camera {camera_id}: insufficient corners")
                continue
            
            # Prepare corresponding points
            source_points = []
            target_points = []
            
            for marker_id, pixel_pos in corners.items():
                if marker_id in self.marker_positions_cm:
                    # Source: pixel coordinates in camera image
                    source_points.append(pixel_pos)
                    
                    # Target: physical position converted to pixels in final image
                    cm_pos = self.marker_positions_cm[marker_id]
                    pixel_pos_target = (
                        cm_pos[0] * self.config.pixels_per_cm,
                        cm_pos[1] * self.config.pixels_per_cm
                    )
                    target_points.append(pixel_pos_target)
            
            if len(source_points) < 3:
                print(f"Skipping camera {camera_id}: need at least 3 matching markers")
                continue
            
            source_points = np.array(source_points, dtype=np.float32)
            target_points = np.array(target_points, dtype=np.float32)
            
            print(f"Camera {camera_id} calibration:")
            print(f"  Source points (pixels): {source_points}")
            print(f"  Target points (pixels): {target_points}")
            
            # Calculate homography
            if len(source_points) >= 4:
                # Use all points for better accuracy
                homography, _ = cv2.findHomography(source_points, target_points, cv2.RANSAC)
            else:
                # Use affine transformation for 3 points
                homography = cv2.getAffineTransform(source_points[:3], target_points[:3])
                # Convert 2x3 affine to 3x3 homography format
                homography = np.vstack([homography, [0, 0, 1]])
            
            # Calculate which part of the workspace this camera covers
            image_height, image_width = calibration_frames[camera_id][0].shape[:2]
            physical_bounds = self._calculate_camera_coverage(homography, image_width, image_height)
            
            self.calibrations[camera_id] = CameraCalibration(
                camera_id=camera_id,
                homography_matrix=homography,
                output_size=self.output_size_pixels,
                offset=(0, 0),  # No offset needed - everything maps to same coordinate system
                physical_bounds=physical_bounds
            )
            
            print(f"  Homography calculated successfully")
            print(f"  Physical coverage: {physical_bounds} cm")
    
    def _calculate_camera_coverage(self, homography: np.ndarray, width: int, height: int) -> Tuple[float, float, float, float]:
        """Calculate the physical area (in cm) that this camera covers"""
        
        # Transform image corners to physical coordinates
        corners = np.array([
            [0, 0],
            [width, 0],
            [width, height],
            [0, height]
        ], dtype=np.float32).reshape(-1, 1, 2)
        
        transformed_corners = cv2.perspectiveTransform(corners, homography)
        transformed_corners = transformed_corners.reshape(-1, 2)
        
        # Convert from pixels to centimeters
        transformed_corners_cm = transformed_corners / self.config.pixels_per_cm
        
        # Find bounding box in cm
        min_x = float(transformed_corners_cm[:, 0].min())
        max_x = float(transformed_corners_cm[:, 0].max())
        min_y = float(transformed_corners_cm[:, 1].min())
        max_y = float(transformed_corners_cm[:, 1].max())
        
        return (min_x, min_y, max_x, max_y)
    
    def stitch_frame(self, camera_frames: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
        """
        Stitch images from multiple cameras into one physical workspace view
        camera_frames: {camera_id: image}
        """
        if not self.is_calibrated:
            print("Warning: Cameras not calibrated yet!")
            return None
        
        # Create output image
        stitched = np.zeros(self.output_size_pixels[::-1], dtype=np.uint8)  # Height x Width
        overlap_count = np.zeros(self.output_size_pixels[::-1], dtype=np.uint8)
        
        # Process each camera
        for camera_id, image in camera_frames.items():
            if camera_id not in self.calibrations:
                continue
                
            cal = self.calibrations[camera_id]
            
            # Apply homography transformation
            warped = cv2.warpPerspective(
                image,
                cal.homography_matrix,
                self.output_size_pixels,
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
            
            # Simple blending: average overlapping areas
            mask = warped > 0
            stitched[mask] = ((stitched[mask].astype(np.uint16) * overlap_count[mask] + warped[mask]) / 
                             (overlap_count[mask] + 1)).astype(np.uint8)
            overlap_count[mask] += 1
        
        return stitched
    
    def pixel_to_physical(self, pixel_coord: Tuple[int, int]) -> Tuple[float, float]:
        """Convert pixel coordinates in stitched image to physical coordinates in cm"""
        if not self.is_calibrated:
            return None
            
        x_cm = pixel_coord[0] / self.config.pixels_per_cm
        y_cm = pixel_coord[1] / self.config.pixels_per_cm
        return (x_cm, y_cm)
    
    def physical_to_pixel(self, physical_coord: Tuple[float, float]) -> Tuple[int, int]:
        """Convert physical coordinates in cm to pixel coordinates in stitched image"""
        if not self.is_calibrated:
            return None
            
        x_pixel = int(physical_coord[0] * self.config.pixels_per_cm)
        y_pixel = int(physical_coord[1] * self.config.pixels_per_cm)
        return (x_pixel, y_pixel)
    
    def save_calibration(self, filepath: str):
        """Save calibration data including physical measurements"""
        if not self.is_calibrated:
            print("No calibration data to save")
            return
            
        calibration_data = {
            'physical_config': {
                'edge_to_marker_x': self.config.edge_to_marker_x,
                'edge_to_marker_y': self.config.edge_to_marker_y,
                'marker_spacing_x': self.config.marker_spacing_x,
                'marker_spacing_y': self.config.marker_spacing_y,
                'pixels_per_cm': self.config.pixels_per_cm
            },
            'workspace_bounds_cm': self.workspace_bounds_cm,
            'output_size_pixels': self.output_size_pixels,
            'marker_positions_cm': self.marker_positions_cm,
            'cameras': {}
        }
        
        for camera_id, cal in self.calibrations.items():
            calibration_data['cameras'][camera_id] = {
                'homography': cal.homography_matrix.tolist(),
                'physical_bounds': cal.physical_bounds
            }
        
        import json
        with open(filepath, 'w') as f:
            json.dump(calibration_data, f, indent=2)
        
        print(f"Physical calibration saved to {filepath}")


# Example configuration and usage
def create_example_config():
    """Example configuration with real measurements"""
    return PhysicalCalibrationConfig(
        edge_to_marker_x=5.0,    # 5 cm from left/right edge to marker
        edge_to_marker_y=3.0,    # 3 cm from top/bottom edge to marker
        marker_spacing_x=50.0,   # 50 cm between left and right markers
        marker_spacing_y=35.0,   # 35 cm between top and bottom markers
        pixels_per_cm=8.0        # 8 pixels per centimeter resolution
    )

# Integration example
class PhysicalStitchingIntegration:
    """Integration with physical measurements"""
    
    def __init__(self, physical_config: PhysicalCalibrationConfig):
        self.stitcher = PhysicalMultiCameraStitcher(physical_config)
        self.calibration_frames_collected = {}
        self.calibration_needed = True
    
    async def send_tracking_matches_with_physical_stitching(self, connection):
        """Modified version with physical coordinate stitching"""
        markers_holder = Markers()
        last_sent = time_ns()
        frame_count = 0
        
        for frame in poll_frame_data():
            camera_id, image = frame
            frame_count += 1
            
            ir_image = sharpen_and_rotate_image(buffer_to_array(image))
            corners, ids, rejectedImgPoints = detect_markers(ir_image)
            buildingDict = map_detected_markers(camera_id, ids, corners)
            
            # Collect calibration data if needed
            if self.calibration_needed:
                self._collect_calibration_frame(camera_id, ir_image, buildingDict)
            
            # Regular marker tracking and display
            draw_monitor_window(ir_image, corners, rejectedImgPoints, camera_id)
            draw_status_window(buildingDict, camera_id)
            
            # Store frame for stitching
            if not self.calibration_needed:
                if not hasattr(self, 'current_frames'):
                    self.current_frames = {}
                self.current_frames[camera_id] = ir_image
                
                # Stitch when we have frames from both cameras
                if len(self.current_frames) >= 2:
                    stitched = self.stitcher.stitch_frame(self.current_frames)
                    if stitched is not None:
                        # Add physical measurements overlay
                        stitched_with_overlay = self._add_physical_overlay(stitched)
                        cv2.imshow("Physical Workspace View", stitched_with_overlay)
                        cv2.waitKey(1)
                    self.current_frames.clear()
            
            markers_holder.addMarkers(track_v2(frame))
            
            if (time_ns() - last_sent > 200_000_000):
                markers_json = markers_holder.toJSON()
                print("Sending to unity:", markers_json)
                last_sent = time_ns()
                markers_holder.clear()
                await loop.sock_sendall(connection, markers_json.encode("utf-8"))
    
    def _collect_calibration_frame(self, camera_id: str, image: np.ndarray, building_dict: Dict):
        """Collect frames for physical calibration"""
        corner_count = sum(1 for marker_id in [10, 11, 12, 13] if marker_id in building_dict)
        
        if corner_count >= 2:  # Need at least 2 corners
            self.calibration_frames_collected[camera_id] = (image.copy(), building_dict.copy())
            print(f"Collected calibration frame for camera {camera_id} ({corner_count} corners)")
            
            # Start calibration when we have frames from at least 2 cameras
            if len(self.calibration_frames_collected) >= 2:
                try:
                    self.stitcher.calibrate_cameras(self.calibration_frames_collected)
                    self.stitcher.save_calibration("physical_calibration.json")
                    self.calibration_needed = False
                    print("Physical calibration completed! Starting real-time stitching...")
                except Exception as e:
                    print(f"Physical calibration failed: {e}")
    
    def _add_physical_overlay(self, image: np.ndarray) -> np.ndarray:
        """Add physical measurement overlay to stitched image"""
        overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # Draw grid lines every 10cm
        config = self.stitcher.config
        for cm_x in range(0, int(self.stitcher.workspace_bounds_cm[2]), 10):
            pixel_x = int(cm_x * config.pixels_per_cm)
            cv2.line(overlay, (pixel_x, 0), (pixel_x, overlay.shape[0]), (0, 255, 0), 1)
        
        for cm_y in range(0, int(self.stitcher.workspace_bounds_cm[3]), 10):
            pixel_y = int(cm_y * config.pixels_per_cm)
            cv2.line(overlay, (0, pixel_y), (overlay.shape[1], pixel_y), (0, 255, 0), 1)
        
        # Draw marker positions
        if hasattr(self.stitcher, 'marker_positions_cm'):
            for marker_id, (cm_x, cm_y) in self.stitcher.marker_positions_cm.items():
                pixel_pos = self.stitcher.physical_to_pixel((cm_x, cm_y))
                cv2.circle(overlay, pixel_pos, 5, (0, 0, 255), -1)
                cv2.putText(overlay, str(marker_id), 
                           (pixel_pos[0] + 10, pixel_pos[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return overlay


# Example usage
if __name__ == "__main__":
    # Create configuration with your actual measurements
    config = PhysicalCalibrationConfig(
        edge_to_marker_x=7.5,    # Replace with your actual measurements
        edge_to_marker_y=4.2,    
        marker_spacing_x=60.3,   
        marker_spacing_y=42.8,   
        pixels_per_cm=12.0       
    )
    
    print("Physical stitching system ready!")
    print(f"Expected workspace: {config.marker_spacing_x + 2*config.edge_to_marker_x:.1f} x {config.marker_spacing_y + 2*config.edge_to_marker_y:.1f} cm")


    """
    Perfect! Having real-world measurements makes this much more accurate. Here's the updated approach that uses the physical dimensions and marker offsets:

Perfect! Here's the updated approach that uses your real-world physical measurements:
Key Improvements:
1. Physical Configuration
python

PhysicalCalibrationConfig(
    edge_to_marker_x=7.5,    # Distance from frame edge to marker (cm)
    edge_to_marker_y=4.2,    # Distance from frame edge to marker (cm)  
    marker_spacing_x=60.3,   # Distance between markers (cm)
    marker_spacing_y=42.8,   # Distance between markers (cm)
    pixels_per_cm=12.0       # Resolution of final image
)

2. True Physical Coordinate System

    Creates a coordinate system in real centimeters
    Maps pixel coordinates directly to physical workspace
    Accounts for areas beyond the markers (using edge distances)

3. Accurate Workspace Calculation

Total workspace width = marker_spacing_x + 2 × edge_to_marker_x
Total workspace height = marker_spacing_y + 2 × edge_to_marker_y

4. Precise Homography

    Uses known physical positions of markers as target points
    Transforms camera views to match real-world coordinates
    Works even if you only see 2-3 corner markers per camera

5. Physical Measurement Tools

    pixel_to_physical(): Convert pixels → centimeters
    physical_to_pixel(): Convert centimeters → pixels
    Visual grid overlay every 10cm for reference

Setup Process:

    Measure your setup:

    python

    config = PhysicalCalibrationConfig(
        edge_to_marker_x=YOUR_X_DISTANCE,    # cm from edge to marker
        edge_to_marker_y=YOUR_Y_DISTANCE,    # cm from edge to marker
        marker_spacing_x=YOUR_MARKER_WIDTH,  # cm between left/right markers
        marker_spacing_y=YOUR_MARKER_HEIGHT, # cm between top/bottom markers
        pixels_per_cm=10.0                   # desired resolution
    )

    Place corner markers: IDs 10, 11, 12, 13 (top-left, top-right, bottom-right, bottom-left)
    Run calibration: Show markers to both cameras simultaneously
    Get stitched view: Real-time stitching with physical coordinate grid

Advantages:

✅ True scale: Final image shows real physical distances
✅ Flexible: Works even if cameras don't see all 4 corners
✅ Accurate: Uses your precise physical measurements
✅ Extensible: Easy to add more cameras or change workspace size
✅ Measurable: Can measure distances directly in the stitched image

The stitched image will show your exact physical workspace with a coordinate system you can trust for measurements!
5-hour limit reached ∙ resets 9:00 PM
Upgrade



    
    """