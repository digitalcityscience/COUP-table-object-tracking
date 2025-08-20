import asyncio
import socket

from marker import Markers, map_detected_markers
from camera import poll_frame_data
from tracker import track, track_v2
from time import time_ns
from detection import detect_markers
from hud import draw_monitor_window, draw_status_window
from image import buffer_to_array, sharpen_and_rotate_image



from collections import defaultdict
from typing import Dict, List
import os
from datetime import datetime



socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
SERVER_SETTINGS = ("localhost", 8052)
print(f"Listening to socket connections on: {SERVER_SETTINGS}")
socket.bind(SERVER_SETTINGS)
socket.listen(1)
socket.setblocking(False)
loop = asyncio.new_event_loop()


class SimpleMarkerTracker:
    """Simple tracker that works with the existing buildingDict"""
    
    def __init__(self, output_dir: str = "marker_stats"):
        self.output_dir = output_dir
        # Structure: {marker_id: {camera_id: {'positions': [(x, y)], 'count': int}}}
        self.marker_data = defaultdict(lambda: defaultdict(lambda: {'positions': [], 'count': 0}))
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    def add_frame_data(self, camera_id: str, building_dict: Dict[int, any]):
        """Add data from buildingDict for this frame"""

        markers_of_interest = []  # TODO what are the markers in the corners?

        for marker_id, marker in building_dict.items():

            if marker_id not in markers_of_interest:
                continue

            # Extract position (assuming marker.position has x, y attributes)
            x, y = marker.position.x, marker.position.y
            
            # Add to tracking data
            self.marker_data[marker_id][camera_id]['positions'].append((x, y))
            self.marker_data[marker_id][camera_id]['count'] += 1
    
    def calculate_stats(self, marker_id: int, camera_id: str) -> Dict:
        """Calculate simple statistics for a marker"""
        data = self.marker_data[marker_id][camera_id]
        positions = data['positions']
        
        if len(positions) < 2:
            return None
        
        # Extract coordinates
        x_coords = [pos[0] for pos in positions]
        y_coords = [pos[1] for pos in positions]
        
        # Calculate statistics
        stats = {
            'marker_id': marker_id,
            'camera_id': camera_id,
            'total_detections': len(positions),
            'min_position': {'x': min(x_coords), 'y': min(y_coords)},
            'max_position': {'x': max(x_coords), 'y': max(y_coords)},
            'mean_position': {'x': np.mean(x_coords), 'y': np.mean(y_coords)},
            'std_deviation': {'x': np.std(x_coords), 'y': np.std(y_coords)},
            'position_range': {
                'x': max(x_coords) - min(x_coords),
                'y': max(y_coords) - min(y_coords)
            }
        }
        
        # Simple stability score (lower = more stable)
        x_stability = stats['std_deviation']['x'] / max(stats['position_range']['x'], 1.0)
        y_stability = stats['std_deviation']['y'] / max(stats['position_range']['y'], 1.0)
        stats['stability_score'] = np.sqrt(x_stability**2 + y_stability**2)
        
        # Add timestamp
        stats['timestamp'] = datetime.now().isoformat()
        
        return stats
    
    def write_stats_files(self):
        """Write statistics files for all tracked markers"""
        written_files = []
        
        for marker_id in self.marker_data:
            for camera_id in self.marker_data[marker_id]:
                stats = self.calculate_stats(marker_id, camera_id)
                
                if stats is None:
                    continue
                
                filename = f"marker_{marker_id}_camera_{camera_id}.json"
                filepath = os.path.join(self.output_dir, filename)
                
                with open(filepath, 'w') as f:
                    json.dump(stats, f, indent=2)
                
                written_files.append(filepath)
                print(f"Stats written: Marker {marker_id} (Camera {camera_id}) - "
                      f"{stats['total_detections']} detections, "
                      f"stability: {stats['stability_score']:.3f}")
        
        return written_files



async def main():
    while True:
        connection, client_address = await loop.sock_accept(socket)
        print(f"Connection from: {client_address}")
        loop.create_task(send_tracking_matches(connection))


async def send_tracking_matches(connection):
    markers_holder = Markers()
    marker_tracker = SimpleMarkerTracker()
    last_sent = time_ns()
    for frame in poll_frame_data():
        camera_id, image = frame
        ir_image = sharpen_and_rotate_image(buffer_to_array(image))
        corners, ids, rejectedImgPoints = detect_markers(ir_image)
        buildingDict = map_detected_markers(camera_id, ids, corners)

        # Add this line to track the markers
        marker_tracker.add_frame_data(camera_id, buildingDict)    

        draw_monitor_window(ir_image, corners, rejectedImgPoints, camera_id)
        draw_status_window(buildingDict, camera_id)

        markers_holder.addMarkers(track_v2(frame))
        if (time_ns() - last_sent > 200_000_000):
            markers_json = markers_holder.toJSON()
            print("Sending to unity:", markers_json)
            last_sent = time_ns()
            markers_holder.clear()
            await loop.sock_sendall(connection, markers_json.encode("utf-8"))
            

            
async def test():
    markers_holder = Markers()
    last_sent = time_ns()
    for frame in poll_frame_data():
        markers_holder.addMarkers(track_v2(frame))
        if (time_ns() - last_sent > 200_000_000):
            markers_json = markers_holder.toJSON()
            print("Sending to unity:", markers_json)
            last_sent = time_ns()
            markers_holder.clear()
            markers_json.encode("utf-8")


loop.run_until_complete(main())
