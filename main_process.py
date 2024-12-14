import cv2
import numpy as np
import os 

from share.constants.config import (CLASSES_FILE, DATA_VIDEO, FRAME_SKIP,
                                    GREEN_COLOR, MOVEMENT_THRESHOLD_METRO_MOVE,
                                    MOVEMENT_THRESHOLD_METRO_STOP, ROI_COLOR,
                                     STOPPED_FRAME_THRESHOLD,
                                    YOLO_MODEL_PATH)
from share.utils.band_detection import (check_train_movement_in_rois,
                                        detect_yellow_band, select_roi)
from share.utils.drawing_utils import draw_detections
from share.utils.yolo_utils import (load_classes, load_yolo_model,
                                    perform_yolo_detection)


def draw_rois(frame, ROIS):
    for x1, y1, x2, y2 in ROIS:
        cv2.rectangle(frame, (x1, y1), (x2, y2), ROI_COLOR, 2)

def evaluate_train_state(frame, prev_frame, train_moving, train_stopped_counter, wait_frames, consecutive_moving_frames,ROIS):
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


def process_video(video_path, output_path, ROIS):
    model = load_yolo_model(YOLO_MODEL_PATH)
    classes = load_classes(CLASSES_FILE)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: No se pudo abrir el video.")
        return

    # Propiedades del video
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Validar ROIs
    for roi in ROIS:
        x1, y1, x2, y2 = roi
        if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
            print(f"ROI inválido: ({x1}, {y1}, {x2}, {y2}).")
            return

    # Crear el archivo de salida
    output_file = os.path.join(output_path, "video_procesado.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    if not out.isOpened():
        print(f"Error: No se pudo abrir el archivo de salida en {output_file}.")
        return

    # Variables de procesamiento
    last_boxes, last_class_ids = [], []
    yellow_band_points = None
    ret, prev_frame = cap.read()
    if not ret:
        print("Error: No se pudo leer el primer frame.")
        return

    frame_count = 0
    train_stopped_counter = 0
    frames_wait_procces_roi = 0
    train_moving = False
    consecutive_moving_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Error: No se pudo leer un frame.")
            break

        # Detectar bandas amarillas si no se detectaron previamente
        if yellow_band_points is None:
            yellow_band_points = detect_yellow_band(frame)

        # Procesar frame
        last_boxes, last_class_ids = perform_yolo_detection(model, frame)
        train_moving, train_stopped_counter, frames_wait_procces_roi, consecutive_moving_frames = evaluate_train_state(
            frame, prev_frame, train_moving, train_stopped_counter, frames_wait_procces_roi, consecutive_moving_frames, ROIS
        )

        # Dibujar ROIs
        prev_frame = frame.copy()
        draw_rois(frame, ROIS)
        
        if train_moving:
            draw_detections(frame, last_boxes, last_class_ids, yellow_band_points, classes)
        else:
            cv2.putText(frame, "Tren detenido - Alerta desactivada", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, GREEN_COLOR, 2)

        # Guardar el frame procesado en el archivo de salida
        cv2.imshow('Deteccion de Personas y Franja Amarilla', frame)
        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Video procesado guardado en: {output_file}")

        
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Video procesado guardado en: {output_path}")

