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
    pixels_per_cm_x: float = 10.0  # x-direction resolution
    pixels_per_cm_y: float = 10.0  # y-direction resolution
    
    # Workspace configuration
    total_workspace_width_cm: float = 160.0  # Total width of workspace in cm
    
    # Optional camera parameters
    field_of_view: Optional[Tuple[float, float]] = None  # horizontal and vertical FOV in degrees
    resolution: Optional[Tuple[int, int]] = None  # camera resolution (width, height)
    distance_to_surface: Optional[float] = None  # distance from camera to surface in cm

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
        # Camera-specific corner marker IDs: Top-left, top-right, bottom-right, bottom-left
        self.camera_corner_ids = {
            "000": [48, 44, 41, 43],
            "001": [68, 62, 69, 135]
        }
        self.corner_marker_mapping = {}  # Maps camera_id -> {corner_position: marker_id}
        self.is_calibrated = False
        self.workspace_bounds_cm = None  # (min_x, min_y, max_x, max_y) in cm
        self.output_size_pixels = None
    
    def configure_corner_markers(self, camera_corner_mapping: Dict[str, Dict[str, int]]):
        """
        Configure corner markers for each camera
        camera_corner_mapping: {
            camera_id: {
                'top_left': marker_id,
                'top_right': marker_id, 
                'bottom_left': marker_id,
                'bottom_right': marker_id
            }
        }
        """
        self.corner_marker_mapping = camera_corner_mapping
        print("Corner marker mapping configured:")
        for camera_id, corners in camera_corner_mapping.items():
            print(f"  Camera {camera_id}: {corners}")
    
    def auto_configure_corner_markers_from_detected(self, detected_corners: Dict[str, Dict[str, Tuple[int, Tuple[float, float]]]]):
        """
        Automatically configure corner markers based on detected positions
        detected_corners: {
            camera_id: {
                'top_left': (marker_id, (x, y)),
                'top_right': (marker_id, (x, y)),
                'bottom_left': (marker_id, (x, y)), 
                'bottom_right': (marker_id, (x, y))
            }
        }
        """
        self.corner_marker_mapping = {}
        for camera_id, corners in detected_corners.items():
            self.corner_marker_mapping[camera_id] = {}
            for position, (marker_id, _) in corners.items():
                self.corner_marker_mapping[camera_id][position] = marker_id
        
        print("Auto-configured corner marker mapping:")
        for camera_id, corners in self.corner_marker_mapping.items():
            print(f"  Camera {camera_id}: {corners}")
    
    def calibrate_cameras(self, calibration_frames: Dict[str, Tuple[np.ndarray, Dict]]):
        """
        Calibrate cameras using corner markers and physical measurements
        calibration_frames: {camera_id: (image, buildingDict)}
        """
        print("Starting physical calibration...")
        
        # Find corner markers in all cameras
        camera_corners = {}
        
        for camera_id, (image, building_dict) in calibration_frames.items():
            corners = self._extract_corner_positions(building_dict, camera_id)
            if len(corners) >= 2:  # Need at least 2 corners for calibration
                camera_corners[camera_id] = corners
                print(f"Found {len(corners)} corners in camera {camera_id}: {list(corners.keys())}")
            else:
                print(f"Warning: Only found {len(corners)} corners in camera {camera_id}")
        
        if len(camera_corners) < 2:
            raise ValueError("Need at least 2 cameras with corner markers visible")
        
        # Calculate pixels per cm if possible
        self._calculate_pixels_per_cm(camera_corners)
        
        # Calculate the full workspace bounds based on physical measurements
        self._calculate_workspace_bounds(camera_corners)
        
        # Calculate transformations for each camera
        self._calculate_physical_homographies(camera_corners, calibration_frames)
        
        self.is_calibrated = True
        print(f"Physical calibration completed!")
        print(f"Workspace bounds: {self.workspace_bounds_cm} cm")
        print(f"Output image size: {self.output_size_pixels} pixels")
        
    def _extract_corner_positions(self, building_dict: Dict, camera_id: str = None) -> Dict[int, Tuple[float, float]]:
        """Extract corner marker positions from buildingDict"""
        corners = {}
        
        # Use camera-specific marker IDs
        if camera_id and camera_id in self.camera_corner_ids:
            camera_markers = self.camera_corner_ids[camera_id]
            for marker_id in camera_markers:
                if marker_id in building_dict:
                    marker = building_dict[marker_id]
                    corners[marker_id] = (marker.position[0], marker.position[1])
        
        return corners
    
    def _calculate_pixels_per_cm(self, camera_corners: Dict):
        """Calculate pixels per cm based on marker positions in images"""
        print("Calculating pixels per cm from marker positions...")
        
        # We'll collect measurements from all cameras and take the average
        x_measurements = []
        y_measurements = []
        
        for camera_id, corners in camera_corners.items():
            # Get corner marker IDs for this camera
            if camera_id not in self.camera_corner_ids:
                print(f"No corner marker configuration for camera {camera_id}")
                continue
                
            # Get marker IDs for this camera: Top-left, top-right, bottom-right, bottom-left
            tl_id, tr_id, br_id, bl_id = self.camera_corner_ids[camera_id]
            
            # Check if we have horizontal marker pairs (top edge or bottom edge)
            if tl_id in corners and tr_id in corners:
                # Top edge
                pixel_distance_x = np.linalg.norm(
                    np.array(corners[tr_id]) - np.array(corners[tl_id])
                )
                x_measurements.append(pixel_distance_x / self.config.marker_spacing_x)
                print(f"Camera {camera_id}: Horizontal marker distance ({tl_id}-{tr_id}): {pixel_distance_x:.1f} pixels / {self.config.marker_spacing_x} cm")
                
            if bl_id in corners and br_id in corners:
                # Bottom edge
                pixel_distance_x = np.linalg.norm(
                    np.array(corners[br_id]) - np.array(corners[bl_id])
                )
                x_measurements.append(pixel_distance_x / self.config.marker_spacing_x)
                print(f"Camera {camera_id}: Horizontal marker distance ({bl_id}-{br_id}): {pixel_distance_x:.1f} pixels / {self.config.marker_spacing_x} cm")
                
            # Check if we have vertical marker pairs (left edge or right edge)
            if tl_id in corners and bl_id in corners:
                # Left edge
                pixel_distance_y = np.linalg.norm(
                    np.array(corners[bl_id]) - np.array(corners[tl_id])
                )
                y_measurements.append(pixel_distance_y / self.config.marker_spacing_y)
                print(f"Camera {camera_id}: Vertical marker distance ({tl_id}-{bl_id}): {pixel_distance_y:.1f} pixels / {self.config.marker_spacing_y} cm")
                
            if tr_id in corners and br_id in corners:
                # Right edge
                pixel_distance_y = np.linalg.norm(
                    np.array(corners[br_id]) - np.array(corners[tr_id])
                )
                y_measurements.append(pixel_distance_y / self.config.marker_spacing_y)
                print(f"Camera {camera_id}: Vertical marker distance ({tr_id}-{br_id}): {pixel_distance_y:.1f} pixels / {self.config.marker_spacing_y} cm")
        
        # Calculate average pixels per cm if we have measurements
        if x_measurements:
            avg_pixels_per_cm_x = sum(x_measurements) / len(x_measurements)
            self.config.pixels_per_cm_x = avg_pixels_per_cm_x
            print(f"Calculated pixels_per_cm_x: {avg_pixels_per_cm_x:.2f}")
        else:
            print(f"Using default pixels_per_cm_x: {self.config.pixels_per_cm_x}")
            
        if y_measurements:
            avg_pixels_per_cm_y = sum(y_measurements) / len(y_measurements)
            self.config.pixels_per_cm_y = avg_pixels_per_cm_y
            print(f"Calculated pixels_per_cm_y: {avg_pixels_per_cm_y:.2f}")
        else:
            print(f"Using default pixels_per_cm_y: {self.config.pixels_per_cm_y}")
            
        # Fallback to camera parameters if no measurements and parameters are available
        if not x_measurements and not y_measurements and self.config.field_of_view is not None and self.config.resolution is not None and self.config.distance_to_surface is not None:
            self._calculate_pixels_per_cm_from_camera_params()

    def _calculate_pixels_per_cm_from_camera_params(self):
        """Calculate pixels per cm from camera parameters as fallback"""
        if not hasattr(self.config, 'field_of_view') or self.config.field_of_view is None:
            return
        if not hasattr(self.config, 'resolution') or self.config.resolution is None:
            return
        if not hasattr(self.config, 'distance_to_surface') or self.config.distance_to_surface is None:
            return
            
        # Calculate field of view width and height at the given distance
        fov_h_rad = np.radians(self.config.field_of_view[0])
        fov_v_rad = np.radians(self.config.field_of_view[1])
        
        # Width and height of view at the given distance (in cm)
        view_width_cm = 2 * self.config.distance_to_surface * np.tan(fov_h_rad / 2)
        view_height_cm = 2 * self.config.distance_to_surface * np.tan(fov_v_rad / 2)
        
        # Calculate pixels per cm
        pixels_per_cm_x = self.config.resolution[0] / view_width_cm
        pixels_per_cm_y = self.config.resolution[1] / view_height_cm
        
        print(f"Calculated from camera parameters:")
        print(f"  Field of view at {self.config.distance_to_surface} cm: {view_width_cm:.1f} x {view_height_cm:.1f} cm")
        print(f"  pixels_per_cm_x: {pixels_per_cm_x:.2f}")
        print(f"  pixels_per_cm_y: {pixels_per_cm_y:.2f}")
        
        # Only use these values if we don't have measurements from markers
        if self.config.pixels_per_cm_x == 10.0:  # Default value
            self.config.pixels_per_cm_x = pixels_per_cm_x
        if self.config.pixels_per_cm_y == 10.0:  # Default value
            self.config.pixels_per_cm_y = pixels_per_cm_y
    
    def _calculate_workspace_bounds(self, camera_corners: Dict):
        """Calculate the physical workspace bounds in centimeters"""
        
        # Define physical positions of markers in centimeters
        marker_positions_cm = {}
        camera_physical_bounds = {}
        
        # Use exact known physical positions for the markers (in cm)
        # For camera 000: [48, 44, 41, 43] (top-left, top-right, bottom-right, bottom-left)
        # For camera 001: [68, 62, 69, 135] (top-left, top-right, bottom-right, bottom-left)
        marker_positions_cm = {
            # Camera 000 markers
            48: (3, 3),    # top-left
            44: (77, 3),   # top-right
            41: (77, 77),  # bottom-right
            43: (3, 77),   # bottom-left
            
            # Camera 001 markers
            68: (83, 3),   # top-left
            62: (157, 3),  # top-right
            69: (157, 77), # bottom-right
            135: (83, 77), # bottom-left
        }
        
        # Physical dimensions for each camera view
        camera_physical_bounds = {
            "000": (0, 0, 80, 80),      # (min_x, min_y, max_x, max_y) in cm
            "001": (80, 0, 160, 80)     # (min_x, min_y, max_x, max_y) in cm
        }
        
        # Total workspace dimensions
        workspace_width_cm = 160  # Total width of both tables
        workspace_height_cm = 80   # Height of the tables
        
        self.workspace_bounds_cm = (0, 0, workspace_width_cm, workspace_height_cm)
        self.camera_physical_bounds = camera_physical_bounds
        
        # Calculate output image size in pixels
        self.output_size_pixels = (
            int(workspace_width_cm * self.config.pixels_per_cm_x),
            int(workspace_height_cm * self.config.pixels_per_cm_y)
        )
        
        print(f"Physical workspace: {workspace_width_cm:.1f} x {workspace_height_cm:.1f} cm")
        print(f"Marker positions (cm): {marker_positions_cm}")
        print(f"Camera physical bounds: {camera_physical_bounds}")
        
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
                        cm_pos[0] * self.config.pixels_per_cm_x,
                        cm_pos[1] * self.config.pixels_per_cm_y
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
            
            # Use the pre-calculated physical bounds for panoramic stitching
            physical_bounds = self.camera_physical_bounds[camera_id]
            print(f"  Using panoramic physical bounds: {physical_bounds} cm")
            
            # Calculate offset for this camera in the final image
            offset = (
                int(physical_bounds[0] * self.config.pixels_per_cm_x),
                int(physical_bounds[1] * self.config.pixels_per_cm_y)
            )
            
            self.calibrations[camera_id] = CameraCalibration(
                camera_id=camera_id,
                homography_matrix=homography,
                output_size=self.output_size_pixels,
                offset=offset,
                physical_bounds=physical_bounds
            )
            
            print(f"  Homography calculated successfully")
            print(f"  Physical coverage: {physical_bounds} cm")
            print(f"  Offset in stitched image: {offset} pixels")
    
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
        transformed_corners_cm = np.zeros_like(transformed_corners)
        transformed_corners_cm[:, 0] = transformed_corners[:, 0] / self.config.pixels_per_cm_x
        transformed_corners_cm[:, 1] = transformed_corners[:, 1] / self.config.pixels_per_cm_y
        
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
        
        # Sort cameras by x-offset for consistent processing order (left to right)
        sorted_camera_ids = sorted(
            [cam_id for cam_id in camera_frames if cam_id in self.calibrations],
            key=lambda cam_id: self.calibrations[cam_id].physical_bounds[0]  # Sort by min_x of physical bounds
        )
        
        # Process each camera in order (left to right)
        for camera_id in sorted_camera_ids:
            if camera_id not in self.calibrations:
                continue
                
            image = camera_frames[camera_id]
            cal = self.calibrations[camera_id]
            
            # Apply direct homography transformation to place the camera in its correct position
            # in the overall stitched image
            warped = cv2.warpPerspective(
                image,
                cal.homography_matrix,
                self.output_size_pixels,
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
            
            # Create mask for this camera's contribution
            mask = warped > 0
            
            # Apply the warped image to the stitched result
            if np.sum(overlap_count) == 0:
                # First camera - just copy
                stitched[mask] = warped[mask]
            else:
                # Handle overlapping areas with improved blending
                overlap_mask = np.logical_and(mask, overlap_count > 0)
                if np.any(overlap_mask):
                    # Calculate center of overlap region
                    overlap_y, overlap_x = np.where(overlap_mask)
                    if len(overlap_x) > 0:
                        min_x = np.min(overlap_x)
                        max_x = np.max(overlap_x)
                        mid_x = (min_x + max_x) // 2
                        
                        # Create a gradient weight for blending
                        for x, y in zip(overlap_x, overlap_y):
                            # Weight transitions from 0.9 to 0.1 across the overlap region
                            # (favors left camera on the left side of overlap and right camera on right side)
                            if max_x > min_x:
                                alpha = 0.9 - 0.8 * (x - min_x) / (max_x - min_x)
                            else:
                                alpha = 0.5
                                
                            # Apply weighted blend
                            stitched[y, x] = int(alpha * stitched[y, x] + (1-alpha) * warped[y, x])
                
                # For non-overlapping areas, just copy
                new_mask = np.logical_and(mask, overlap_count == 0)
                stitched[new_mask] = warped[new_mask]
            
            # Update overlap count
            overlap_count[mask] += 1
        
        return stitched
    
    def pixel_to_physical(self, pixel_coord: Tuple[int, int]) -> Tuple[float, float]:
        """Convert pixel coordinates in stitched image to physical coordinates in cm"""
        if not self.is_calibrated:
            return None
            
        x_cm = pixel_coord[0] / self.config.pixels_per_cm_x
        y_cm = pixel_coord[1] / self.config.pixels_per_cm_y
        return (x_cm, y_cm)
    
    def physical_to_pixel(self, physical_coord: Tuple[float, float]) -> Tuple[int, int]:
        """Convert physical coordinates in cm to pixel coordinates in stitched image"""
        if not self.is_calibrated:
            return None
            
        x_pixel = int(physical_coord[0] * self.config.pixels_per_cm_x)
        y_pixel = int(physical_coord[1] * self.config.pixels_per_cm_y)
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
                'pixels_per_cm_x': self.config.pixels_per_cm_x,
                'pixels_per_cm_y': self.config.pixels_per_cm_y
            },
            'workspace_bounds_cm': self.workspace_bounds_cm,
            'output_size_pixels': self.output_size_pixels,
            'marker_positions_cm': self.marker_positions_cm,
            'cameras': {}
        }
        
        # Add camera parameters if available
        if hasattr(self.config, 'field_of_view') and self.config.field_of_view is not None:
            calibration_data['physical_config']['field_of_view'] = self.config.field_of_view
        if hasattr(self.config, 'resolution') and self.config.resolution is not None:
            calibration_data['physical_config']['resolution'] = self.config.resolution
        if hasattr(self.config, 'distance_to_surface') and self.config.distance_to_surface is not None:
            calibration_data['physical_config']['distance_to_surface'] = self.config.distance_to_surface
        
        for camera_id, cal in self.calibrations.items():
            calibration_data['cameras'][camera_id] = {
                'homography': cal.homography_matrix.tolist(),
                'physical_bounds': cal.physical_bounds
            }
        
        import json
        with open(filepath, 'w') as f:
            json.dump(calibration_data, f, indent=2)
        
        print(f"Physical calibration saved to {filepath}")

    def export_calibration_points_json(self, filepath=None):
        """
        Export calibration points to a JSON file with timestamp
        If filepath is not provided, it will generate one with timestamp
        """
        if not self.is_calibrated or not hasattr(self, 'marker_positions_cm'):
            print("No calibration data to export")
            return
        
        # Generate filename with timestamp if not provided
        import datetime
        import os
        if filepath is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            os.makedirs("calibration_exports", exist_ok=True)
            filepath = f"calibration_exports/calibration_points_{timestamp}.json"
        
        # Create export data structure
        export_data = {
            'timestamp': datetime.datetime.now().isoformat(),
            'marker_positions_cm': {},
            'marker_positions_pixels': {},
            'cameras': {},
            'physical_config': {
                'edge_to_marker_x': self.config.edge_to_marker_x,
                'edge_to_marker_y': self.config.edge_to_marker_y,
                'marker_spacing_x': self.config.marker_spacing_x,
                'marker_spacing_y': self.config.marker_spacing_y,
                'pixels_per_cm_x': self.config.pixels_per_cm_x,
                'pixels_per_cm_y': self.config.pixels_per_cm_y
            },
            'workspace_bounds_cm': self.workspace_bounds_cm,
            'output_size_pixels': self.output_size_pixels
        }
        
        # Add marker positions in cm
        for marker_id, position in self.marker_positions_cm.items():
            export_data['marker_positions_cm'][str(marker_id)] = {
                'x': float(position[0]),
                'y': float(position[1])
            }
            # Also convert to pixels
            pixel_pos = self.physical_to_pixel(position)
            export_data['marker_positions_pixels'][str(marker_id)] = {
                'x': int(pixel_pos[0]),
                'y': int(pixel_pos[1])
            }
        
        # Add camera calibration data
        for camera_id, calibration in self.calibrations.items():
            export_data['cameras'][camera_id] = {
                'physical_bounds_cm': {
                    'min_x': float(calibration.physical_bounds[0]),
                    'min_y': float(calibration.physical_bounds[1]),
                    'max_x': float(calibration.physical_bounds[2]),
                    'max_y': float(calibration.physical_bounds[3])
                },
                'offset_pixels': {
                    'x': int(calibration.offset[0]),
                    'y': int(calibration.offset[1])
                },
                'homography_matrix': calibration.homography_matrix.tolist()
            }
        
        # Export to JSON file
        import json
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"Calibration points exported to {filepath}")
        return filepath


# Example configuration and usage
def create_example_config():
    """Example configuration with real measurements"""
    return PhysicalCalibrationConfig(
        edge_to_marker_x=5.0,    # 5 cm from left/right edge to marker
        edge_to_marker_y=3.0,    # 3 cm from top/bottom edge to marker
        marker_spacing_x=50.0,   # 50 cm between left and right markers
        marker_spacing_y=35.0,   # 35 cm between top and bottom markers
        pixels_per_cm_x=8.0,        # 8 pixels per centimeter resolution
        pixels_per_cm_y=8.0        # 8 pixels per centimeter resolution
    )

# Integration example
# Global calibration settings
MIN_MARKERS_PER_CAMERA = 4  # Minimum number of markers required per camera for calibration

class PhysicalStitchingIntegration:
    """Integration with physical measurements"""
    
    def __init__(self, physical_config: PhysicalCalibrationConfig):
        self.stitcher = PhysicalMultiCameraStitcher(physical_config)
        # Define camera-specific corner marker IDs: Top-left, top-right, bottom-right, bottom-left
        camera_corner_ids = {
            "000": [48, 44, 41, 43],
            "001": [68, 62, 69, 135]
        }
        self.stitcher.camera_corner_ids = camera_corner_ids
        self.calibration_frames_collected = {}
        self.calibration_needed = True
        self.marker_collection_start_time = None
        self.marker_collection_timeout = 120  # Wait up to 2 minutes for markers
        self.detected_corner_markers = {}  # Track which corner markers we've seen
    
    async def send_tracking_matches_with_physical_stitching(self, connection):
        """Modified version with physical coordinate stitching"""
        from tracker import Markers
        from time import time_ns
        markers_holder = Markers()
        last_sent = time_ns()
        frame_count = 0
        
        from mock_camera import poll_frame_data
        for frame in poll_frame_data():
            camera_id, image = frame
            frame_count += 1
            
            from image import sharpen_and_rotate_image, buffer_to_array
            from detection import detect_markers, map_detected_markers
            ir_image = sharpen_and_rotate_image(buffer_to_array(image))
            corners, ids, rejectedImgPoints = detect_markers(ir_image)
            buildingDict = map_detected_markers(camera_id, ids, corners)
            
            # Collect calibration data if needed
            if self.calibration_needed:
                self._collect_calibration_frame(camera_id, ir_image, buildingDict)
            
            # Regular marker tracking and display
            from hud import draw_monitor_window, draw_status_window
            draw_monitor_window(ir_image, corners, rejectedImgPoints, camera_id, ids)
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
            
            from tracker import track_v2
            markers_holder.addMarkers(track_v2(frame))
            
            if (time_ns() - last_sent > 200_000_000):
                markers_json = markers_holder.toJSON()
                print("Sending to unity:", markers_json)
                last_sent = time_ns()
                markers_holder.clear()
                import asyncio
                await asyncio.get_event_loop().sock_sendall(connection, markers_json.encode("utf-8"))
    
    def _collect_calibration_frame(self, camera_id: str, image: np.ndarray, building_dict: Dict):
        """Collect frames for physical calibration"""
        # Initialize start time when we first start collecting
        if self.marker_collection_start_time is None:
            import time
            self.marker_collection_start_time = time.time()
            print(f"Starting marker collection, will wait up to {self.marker_collection_timeout} seconds to see all markers")
            
        # Initialize tracking for this camera if not already done
        if camera_id not in self.detected_corner_markers:
            self.detected_corner_markers[camera_id] = set()
            
        # Use camera-specific corner marker IDs if available
        corner_markers = []
        if camera_id in self.stitcher.camera_corner_ids:
            corner_markers = self.stitcher.camera_corner_ids[camera_id]
        else:
            print(f"Warning: No corner marker configuration for camera {camera_id}")
            return
            
        # Track which corner markers we've seen for this camera
        for marker_id in corner_markers:
            if marker_id in building_dict:
                self.detected_corner_markers[camera_id].add(marker_id)
                
        # Print progress
        detected_counts = {cam: len(markers) for cam, markers in self.detected_corner_markers.items()}
        expected_counts = {cam: len(self.stitcher.camera_corner_ids[cam]) for cam in self.detected_corner_markers.keys() 
                          if cam in self.stitcher.camera_corner_ids}
        
        print(f"Marker collection progress: {detected_counts} of {expected_counts} (need minimum {MIN_MARKERS_PER_CAMERA} per camera)")
        
        # Store the current frame for this camera
        self.calibration_frames_collected[camera_id] = (image.copy(), building_dict.copy())
        
        # Check if we have enough markers or if we've timed out
        import time
        elapsed_time = time.time() - self.marker_collection_start_time
        
        # Check if we have minimum required markers for each camera
        ready_for_calibration = all(len(markers) >= MIN_MARKERS_PER_CAMERA for markers in self.detected_corner_markers.values())
        
        # Check if we have all cameras
        all_cameras_present = all(cam in self.detected_corner_markers for cam in self.stitcher.camera_corner_ids)
        
        # Check if we've timed out
        timed_out = elapsed_time > self.marker_collection_timeout
        
        # Attempt calibration when ready or timed out
        if (ready_for_calibration and all_cameras_present) or timed_out:
            print(f"Attempting calibration after {elapsed_time:.1f} seconds")
            if timed_out:
                print(f"Warning: Timed out waiting for all markers. Proceeding with available markers.")
                
            try:
                self.stitcher.calibrate_cameras(self.calibration_frames_collected)
                self.stitcher.save_calibration("physical_calibration.json")
                self.calibration_needed = False
                print("Physical calibration completed! Starting real-time stitching...")
            except Exception as e:
                print(f"Physical calibration failed: {e}")
                # Reset start time to try again
                self.marker_collection_start_time = time.time()
    
    def _add_physical_overlay(self, image: np.ndarray) -> np.ndarray:
        """Add physical measurement overlay to stitched image"""
        overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # Draw grid lines every 10cm
        config = self.stitcher.config
        for cm_x in range(0, int(self.stitcher.workspace_bounds_cm[2]), 10):
            pixel_x = int(cm_x * config.pixels_per_cm_x)
            cv2.line(overlay, (pixel_x, 0), (pixel_x, overlay.shape[0]), (0, 255, 0), 1)
        
        for cm_y in range(0, int(self.stitcher.workspace_bounds_cm[3]), 10):
            pixel_y = int(cm_y * config.pixels_per_cm_y)
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
        edge_to_marker_x=3,    # Distance from frame edge to marker (cm)
        edge_to_marker_y=3,    # Distance from frame edge to marker (cm)
        marker_spacing_x=74,   # Distance between markers (cm)
        marker_spacing_y=74,   # Distance between markers (cm)
        # No need to specify pixels_per_cm values - they will be calculated automatically
        # Optional camera parameters as fallback
        field_of_view=(69.4, 42.5),  # FOV in degrees (horizontal, vertical)
        resolution=(1280, 720),      # Camera resolution
        distance_to_surface=65.0     # Distance in cm
    )
    
    print("Physical stitching system ready!")
    print(f"Expected workspace: {config.marker_spacing_x + 2*config.edge_to_marker_x:.1f} x {config.marker_spacing_y + 2*config.edge_to_marker_y:.1f} cm")
    
    # Example of how the system works:
    print("\nWorkflow:")
    print("1. Place markers with IDs 10, 11, 12, 13 at the corners of your workspace")
    print("2. Capture frames from all cameras showing these markers")
    print("3. The system will automatically calculate:")
    print("   - Pixels per cm in both X and Y directions from marker positions")
    print("   - Homography matrices for each camera")
    print("   - Physical bounds of the workspace")
    print("4. The stitched image will maintain accurate physical proportions")
    print("5. You can convert between pixel and physical coordinates")
    
    # Example of how pixels_per_cm is calculated:
    print("\nExample calculation:")
    print("- If markers 10 and 11 are 500 pixels apart in the image")
    print(f"- And their physical distance is {config.marker_spacing_x} cm")
    print(f"- Then pixels_per_cm_x = 500 / {config.marker_spacing_x} = {500/config.marker_spacing_x:.2f}")
    print("- This is calculated automatically from the marker positions")