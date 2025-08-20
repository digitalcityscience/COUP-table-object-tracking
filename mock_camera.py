import time
from functools import lru_cache
from time import sleep
from typing import Any, Iterable, Tuple
import cv2
import numpy as np
from image import write_to_file
# from realsense.realsense_device_manager import DeviceManager  # Commented out for debugging

FRAMES_PER_SECOND = 30
Frame = Iterable[Tuple[int, Any]]

class MockDeviceManager:
    """Mock device manager that uses MP4 files instead of RealSense cameras"""
    
    def __init__(self, video_files: list):
        self.video_files = video_files
        self.captures = []
        self.device_ids = []
        
        # Initialize video captures
        for i, video_file in enumerate(video_files):
            cap = cv2.VideoCapture(video_file)
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_file}")
            
            # Set FPS to match expected frame rate
            cap.set(cv2.CAP_PROP_FPS, FRAMES_PER_SECOND)
            
            self.captures.append(cap)
            # Create mock camera IDs similar to RealSense format
            self.device_ids.append(f"mock_camera_{i:03d}")
        
        print(f"Initialized {len(self.captures)} mock video streams")
    
    def get_enabled_devices_ids(self):
        return self.device_ids
    
    def poll_frames(self):
        """Poll frames from all video files simultaneously"""
        frames = {}
        
        for i, cap in enumerate(self.captures):
            ret, frame = cap.read()
            
            if not ret:
                # If we've reached the end of any video, restart it
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
            
            if ret:
                # Convert to grayscale to simulate infrared stream
                if len(frame.shape) == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Mock the frame format similar to RealSense
                mock_frame = MockFrame(frame)
                frames[self.device_ids[i]] = {"infrared": mock_frame}
        
        return frames
    
    def disable_streams(self):
        """Clean up video captures"""
        for cap in self.captures:
            cap.release()
        print("All video streams disabled")

class MockFrame:
    """Mock frame class to simulate RealSense frame behavior"""
    
    def __init__(self, data):
        self.data = data
    
    def get_data(self):
        return self.data

@lru_cache(1)
def get_device_manager() -> MockDeviceManager:
    print("initializing mock realsense device manager with MP4 files")
    
    # short videos
    short_video_files = [
        "record_20250813_130932_cid_104.mp4",  
        "record_20250813_130932_cid_863.mp4"
    ]

    # long videos
    long_video_files = [
        "record_20250813_131131_cid_104.mp4",
        "record_20250813_131131_cid_863.mp4"  
    ]
    

    # choose your input files
    video_files = short_video_files

    device_manager = MockDeviceManager(video_files)
    print(f"Active device serial numbers: {device_manager.get_enabled_devices_ids()}")
    return device_manager

def poll_frame_data() -> Frame:
    device_manager = None
    try:
        device_manager = get_device_manager()
        while True:
            frames = device_manager.poll_frames()
            for camera_id in frames:
                frame = frames[camera_id]
                frame_value = list(frame.values())[0]  # Get the infrared frame
                frame_data = frame_value.get_data()
                short_camera_id = camera_id[-3:]  # Get last 3 characters
                yield short_camera_id, frame_data
            
            # Add a small delay to control frame rate
            sleep(1.0 / FRAMES_PER_SECOND)
            
    finally:
        if device_manager:
            device_manager.disable_streams()


# Example usage for testing
if __name__ == "__main__":
    print("Starting debug mode with MP4 files...")
    
    try:
        frame_count = 0
        for camera_id, frame_data in poll_frame_data():
            print(f"Received frame from camera {camera_id}, shape: {frame_data.shape}")
            frame_count += 1
            
            # Stop after a few frames for testing
            if frame_count > 10:
                break
                
    except KeyboardInterrupt:
        print("Stopping debug session...")
    except Exception as e:
        print(f"Error during debugging: {e}")