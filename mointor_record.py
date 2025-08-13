# monitor_record.py
import os
import time
import cv2
from camera import poll_frame_data, FRAMES_PER_SECOND  # 30 FPS aus camera.py
from detection import detect_markers
from image import buffer_to_array, sharpen_and_rotate_image
from hud import draw_monitor_window, draw_status_window

def ensure_dir(path: str):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)

def main(record_seconds: int = 60, out_dir: str = "recordings"):
    ensure_dir(out_dir)

    writers = {}          # camera_id -> VideoWriter
    start_ts = time.time()
    ts_str = time.strftime("%Y%m%d_%H%M%S")

    try:
        for (camera_id, image) in poll_frame_data():  # liefert frames aller aktiven Kameras
            # Preprocessing & Detection
            ir_image = sharpen_and_rotate_image(buffer_to_array(image))
            corners, ids, rejected = detect_markers(ir_image)

            # Fenster zeichnen (IR + Overlays) und genau dieses Frame zurückbekommen
            rendered = draw_monitor_window(ir_image, corners, rejected, camera_id)
            draw_status_window({}, camera_id)  # optional: Status-Overlay; gib hier dein Dict rein falls gewünscht

            # VideoWriter pro Kamera lazy-initialisieren
            if camera_id not in writers:
                h, w = rendered.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # mp4
                filename = os.path.join(out_dir, f"record_{ts_str}_cid_{camera_id}.mp4")
                writers[camera_id] = cv2.VideoWriter(filename, fourcc, FRAMES_PER_SECOND, (w, h), True)

                if not writers[camera_id].isOpened():
                    raise RuntimeError(f"Could not open VideoWriter for camera {camera_id} → {filename}")

            # Frame schreiben (BGR, 3-Kanal)
            writers[camera_id].write(rendered)

            # Nach Ablauf stoppen
            if time.time() - start_ts >= record_seconds:
                break

    finally:
        # sauber schließen
        for w in writers.values():
            try:
                w.release()
            except:
                pass
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main(record_seconds=60)
