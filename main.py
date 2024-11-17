import cv2
import numpy as np

from share.constants.config import (CLASSES_FILE, DATA_VIDEO, FRAME_SKIP,
                                    GREEN_COLOR, ROI_COLOR, ROIS,
                                    YOLO_MODEL_PATH)
from share.utils.band_detection import (check_train_movement_in_rois,
                                        detect_yellow_and_green_bands,
                                        select_roi)
from share.utils.drawing_utils import draw_detections
from share.utils.yolo_utils import (load_classes, load_yolo_model,
                                    perform_yolo_detection)

MOVEMENT_THRESHOLD = 5000  # Umbral para considerar que hay movimiento significativo
STOPPED_FRAME_THRESHOLD = 12  # Número de cuadros consecutivos para considerar el tren detenido


def main():
    model = load_yolo_model(YOLO_MODEL_PATH)
    classes = load_classes(CLASSES_FILE)

    cap = cv2.VideoCapture(DATA_VIDEO)
    if not cap.isOpened():
        print("Error: Could not open the video.")
        return

    last_boxes, last_class_ids, last_confidences = [], [], []
    yellow_band_points, green_band_points = None, None
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

        if yellow_band_points is None or green_band_points is None:
            yellow_band_points, green_band_points = detect_yellow_and_green_bands(frame)
        
        if frame_count % FRAME_SKIP == 0:
            last_boxes, last_class_ids, last_confidences = perform_yolo_detection(model, frame)
        
        if frames_wait_procces_roi > 0:
            frames_wait_procces_roi -= 1
        else:
            train_moving_in_rois = check_train_movement_in_rois(prev_frame, ROIS, frame)
            print("train_moving_in_rois---------------",train_moving_in_rois)
            if train_moving_in_rois:
                train_stopped_counter = 0
                consecutive_moving_frames += 1
                if consecutive_moving_frames >= STOPPED_FRAME_THRESHOLD:
                    train_moving = True
                    frames_wait_procces_roi = 0  # Reset the wait process if train is moving again
            else:
                consecutive_moving_frames = 0
                train_stopped_counter += 1
                if train_stopped_counter >= STOPPED_FRAME_THRESHOLD:
                    train_moving = False
                    frames_wait_procces_roi = 28

        prev_frame = frame.copy()
        for roi in ROIS:
            x1, y1, x2, y2 = roi
            cv2.rectangle(frame, (x1, y1), (x2, y2), ROI_COLOR, 2)
            
        if train_moving:
            draw_detections(frame, last_boxes, last_class_ids, last_confidences, yellow_band_points, green_band_points, classes)
        elif train_stopped_counter >= STOPPED_FRAME_THRESHOLD:
            cv2.putText(frame, "Tren detenido - Alerta desactivada", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, GREEN_COLOR, 2)

        cv2.imshow('Deteccion de Personas y Franja Amarilla', frame)
        frame_count += 1

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
