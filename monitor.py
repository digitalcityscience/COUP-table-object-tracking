import cv2
import time
import os
from typing import Dict
from marker import map_detected_markers
from camera import poll_frame_data
from detection import detect_markers
from hud import draw_monitor_window, draw_status_window
from image import buffer_to_array, sharpen_and_rotate_image

# Video recording setup
video_writers: Dict[str, cv2.VideoWriter] = {}
fps = 30
frame_size = (1280, 800)  # From camera.py: 1280x800 resolution
video_codec = cv2.VideoWriter_fourcc(*'mp4v')

# Create output directory for videos
output_dir = "videos"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def get_video_writer(camera_id: str) -> cv2.VideoWriter:
    """Get or create a video writer for the specified camera."""
    if camera_id not in video_writers:
        timestamp = int(time.time())
        filename = os.path.join(output_dir, f"camera_{camera_id}_{timestamp}.mp4")
        video_writers[camera_id] = cv2.VideoWriter(filename, video_codec, fps, frame_size, False)
        print(f"Started recording for camera {camera_id}: {filename}")
    return video_writers[camera_id]

def cleanup_video_writers():
    """Release all video writers."""
    for camera_id, writer in video_writers.items():
        writer.release()
        print(f"Stopped recording for camera {camera_id}")
    video_writers.clear()

try:
    while True:
        frames = poll_frame_data()
        
        for frame in frames:
            camera_id, image = frame
            ir_image = sharpen_and_rotate_image(buffer_to_array(image))
            corners, ids, rejectedImgPoints = detect_markers(ir_image)

            # Record the processed frame to video
            video_writer = get_video_writer(camera_id)
            video_writer.write(ir_image)

            markerDict = map_detected_markers(camera_id, ids, corners)
            draw_monitor_window(ir_image, corners, rejectedImgPoints, camera_id)
            draw_status_window(markerDict, camera_id)

except KeyboardInterrupt:
    print("\nStopping video recording...")
    cleanup_video_writers()
    cv2.destroyAllWindows()
