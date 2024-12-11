import cv2
import numpy as np

from share.constants.config import (CLASSES_FILE, DATA_VIDEO, FRAME_SKIP,
                                    GREEN_COLOR, MOVEMENT_THRESHOLD_METRO_MOVE,
                                    MOVEMENT_THRESHOLD_METRO_STOP, ROI_COLOR,
                                    ROIS, STOPPED_FRAME_THRESHOLD,
                                    YOLO_MODEL_PATH)
from share.utils.band_detection import (check_train_movement_in_rois,
                                        detect_yellow_band, select_roi)
from share.utils.drawing_utils import draw_detections
from share.utils.yolo_utils import (load_classes, load_yolo_model,
                                    perform_yolo_detection)


def draw_rois(frame):
    for x1, y1, x2, y2 in ROIS:
        cv2.rectangle(frame, (x1, y1), (x2, y2), ROI_COLOR, 2)

def evaluate_train_state(frame, prev_frame, train_moving, train_stopped_counter, wait_frames, consecutive_moving_frames):
    """
    Evaluates the state of the train based on the movement in the regions of interest (ROIs).

    Args:
        frame (np.ndarray): The current frame of the video.
        prev_frame (np.ndarray): The previous frame of the video.
        train_moving (bool): Current state of the train (moving or stopped).
        train_stopped_counter (int): Counter of consecutive frames without movement.
        wait_frames (int): Counter to wait before reprocessing ROIs.
        consecutive_moving_frames (int): Counter of consecutive frames with movement.

    Returns:
        tuple: (train_moving, train_stopped_counter, wait_frames, consecutive_moving_frames)
    """
    if wait_frames > 0:
        return train_moving, train_stopped_counter, wait_frames - 1, consecutive_moving_frames

    threshold = MOVEMENT_THRESHOLD_METRO_STOP if train_moving else MOVEMENT_THRESHOLD_METRO_MOVE
    
    train_moving_in_rois = check_train_movement_in_rois(prev_frame, ROIS, frame, threshold)

    if train_moving_in_rois:
        consecutive_moving_frames += 1
        # If the train has been moving for long enough, mark it as moving
        if consecutive_moving_frames >= STOPPED_FRAME_THRESHOLD:
            train_moving = True
            train_stopped_counter = 0
    else:
        train_stopped_counter += 1
        # If he has been detained for long enough, mark him as detained
        if train_stopped_counter >= STOPPED_FRAME_THRESHOLD:
            consecutive_moving_frames = 0
            train_moving = False
            wait_frames = 18  # Avoid rapid re-evaluations

    return train_moving, train_stopped_counter, wait_frames, consecutive_moving_frames



def main():
    model = load_yolo_model(YOLO_MODEL_PATH)
    classes = load_classes(CLASSES_FILE)

    cap = cv2.VideoCapture(DATA_VIDEO)
    if not cap.isOpened():
        print("Error: Could not open the video.")
        return

    last_boxes, last_class_ids = [], [], []
    yellow_band_points = None
    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        return

    frame_count = 0
    train_stopped_counter = 0
    frames_wait_procces_roi = 0
    train_moving = False
    consecutive_moving_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect sidelines if not already detected
        if yellow_band_points is None:
            yellow_band_points = detect_yellow_band(frame)
            
        # Process frame every FRAME SKIP frames
        if frame_count % FRAME_SKIP == 0:
            last_boxes, last_class_ids = perform_yolo_detection(model, frame)
            train_moving, train_stopped_counter, frames_wait_procces_roi, consecutive_moving_frames = evaluate_train_state(
                frame, prev_frame, train_moving, train_stopped_counter, frames_wait_procces_roi, consecutive_moving_frames
            )

        # Update previous frame and draw ROIs
        prev_frame = frame.copy()
        draw_rois(frame)
        
        if train_moving:
            draw_detections(frame, last_boxes, last_class_ids, yellow_band_points, classes)
        else:
            cv2.putText(frame, "Tren detenido - Alerta desactivada", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, GREEN_COLOR, 2)        
        cv2.imshow('Deteccion de Personas y Franja Amarilla', frame)

        frame_count += 1
        
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
