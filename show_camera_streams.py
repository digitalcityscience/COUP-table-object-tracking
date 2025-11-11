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


def show_single_camera_with_guide(target_cam_id: int):
    """
    Enhanced camera stream display that shows coordinate system orientation
    to help users correctly identify camera positions and marker locations
    """
    for cam_id, image_data in poll_frame_data():
        if cam_id != target_cam_id:
            continue

        ir_image = sharpen_and_rotate_image(buffer_to_array(image_data))
        corners, ids, _ = detect_markers(ir_image)
        
        # Create annotated image
        annotated_image = ir_image.copy()
        
        # Convert to color for better annotations
        if len(annotated_image.shape) == 2:  # Grayscale
            annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_GRAY2BGR)
        
        # Draw detected markers
        if ids is not None:
            annotated_image = cv2.aruco.drawDetectedMarkers(annotated_image, corners, ids)
        
        # Add coordinate system orientation guide
        height, width = annotated_image.shape[:2]
        
        # Draw orientation arrows and labels
        # TOP arrow (pointing up from center-top)
        cv2.arrowedLine(annotated_image, 
                       (width//2, 50), (width//2, 20), 
                       (0, 255, 0), 3, tipLength=0.3)
        cv2.putText(annotated_image, "TOP", 
                   (width//2 - 25, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # BOTTOM arrow (pointing down from center-bottom)  
        cv2.arrowedLine(annotated_image, 
                       (width//2, height-50), (width//2, height-20), 
                       (0, 255, 0), 3, tipLength=0.3)
        cv2.putText(annotated_image, "BOTTOM", 
                   (width//2 - 35, height-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # LEFT arrow (pointing left from center-left)
        cv2.arrowedLine(annotated_image, 
                       (50, height//2), (20, height//2), 
                       (255, 0, 0), 3, tipLength=0.3)
        cv2.putText(annotated_image, "LEFT", 
                   (5, height//2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        
        # RIGHT arrow (pointing right from center-right)
        cv2.arrowedLine(annotated_image, 
                       (width-50, height//2), (width-20, height//2), 
                       (255, 0, 0), 3, tipLength=0.3)
        cv2.putText(annotated_image, "RIGHT", 
                   (width-80, height//2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        
        # Add corner position labels
        cv2.putText(annotated_image, "TOP-LEFT", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(annotated_image, "TOP-RIGHT", 
                   (width-120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(annotated_image, "BOTTOM-LEFT", 
                   (10, height-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(annotated_image, "BOTTOM-RIGHT", 
                   (width-150, height-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Add camera coordinate system explanation
        cv2.putText(annotated_image, f"Camera {cam_id} - Image Coordinate System", 
                   (10, height-40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(annotated_image, "Place markers according to what YOU see in THIS view", 
                   (10, height-25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        cv2.imshow(f"CAMERA {cam_id} - ORIENTATION GUIDE", annotated_image)
        
        # Break on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Manually stopped camera stream")
            break
    
    cv2.destroyAllWindows()

