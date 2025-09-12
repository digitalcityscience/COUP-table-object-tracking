import cv2

from camera import poll_frame_data
from image import sharpen_and_rotate_image, buffer_to_array
from detection import detect_markers

def show_camera_streams():
        for cam_id, image_data in poll_frame_data():
            ir_image = sharpen_and_rotate_image(buffer_to_array(image_data))
            corners, ids, _ = detect_markers(ir_image)
            marker_image = ir_image.copy()

            if ids is not None:
                marker_image = cv2.aruco.drawDetectedMarkers(marker_image, corners, ids)
            
            cv2.imshow(f"CAMERA ID {cam_id}", marker_image)

            # Break on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Manually stopped marker detection")
                break


        cv2.destroyAllWindows()