import cv2
import numpy as np
import os 

from share.constants.config import (CLASSES_FILE,
                                    GREEN_COLOR, MOVEMENT_THRESHOLD_METRO_MOVE,
                                    MOVEMENT_THRESHOLD_METRO_STOP, ROI_COLOR,
                                     STOPPED_FRAME_THRESHOLD,
                                    YOLO_MODEL_PATH)
from share.utils.band_detection import (check_train_movement_in_polygon,
                                        detect_yellow_band, select_roi)
from share.utils.drawing_utils import draw_detections
from share.utils.yolo_utils import (load_classes, load_yolo_model,
                                    perform_yolo_detection)


def draw_polygon(frame, polygon):
    """
    Dibuja un polígono en el frame.

    Args:
        frame (np.ndarray): El frame en el que se dibujará el polígono.
        polygon (list): Lista de puntos [(x1, y1), (x2, y2), ...] que forman el polígono.
    """
    # Verificar si el formato es correcto: una lista de puntos (tuplas)
    if isinstance(polygon, list) and all(isinstance(point, tuple) and len(point) == 2 for point in polygon):
        polygon_points = np.array(polygon, np.int32)
        polygon_points = polygon_points.reshape((-1, 1, 2))  # Necesario para polylines
        cv2.polylines(frame, [polygon_points], isClosed=True, color=(0, 255, 0), thickness=2)  # Color verde
        cv2.imshow("Polygon ROI", frame)
        cv2.waitKey(1)
    else:
        print(f"Formato de polígono inválido: {polygon}.")
        

def evaluate_train_state(frame, prev_frame, train_moving, train_stopped_counter, wait_frames, consecutive_moving_frames, polygon_metro):
    if wait_frames > 0:
        return train_moving, train_stopped_counter, wait_frames - 1, consecutive_moving_frames

    # Determinar el umbral adecuado según el estado actual
    threshold = MOVEMENT_THRESHOLD_METRO_STOP if train_moving else MOVEMENT_THRESHOLD_METRO_MOVE

    # Evaluar movimiento en el polígono
    train_moving_in_rois = check_train_movement_in_polygon(prev_frame, polygon_metro, frame, threshold)

    # Dibuja el polígono en el frame para visualización
    cv2.polylines(frame, [np.array(polygon_metro, dtype=np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)

    # Actualizar contadores según el movimiento detectado
    if train_moving_in_rois:
        consecutive_moving_frames += 1
        if consecutive_moving_frames >= STOPPED_FRAME_THRESHOLD:
            train_moving = True
            train_stopped_counter = 0
    else:
        train_stopped_counter += 1
        if train_stopped_counter >= STOPPED_FRAME_THRESHOLD:
            consecutive_moving_frames = 0
            train_moving = False
            wait_frames = 5  # Evitar reevaluaciones rápidas

    return train_moving, train_stopped_counter, wait_frames, consecutive_moving_frames



def process_video(video_path, output_path, polygon_safe_zone, polygon_metro, display_frame, progress_bar=None):
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
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Total de frames

    # Crear el archivo de salida
    output_file = os.path.join(output_path, "video_procesado.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    if not out.isOpened():
        print(f"Error: No se pudo abrir el archivo de salida en {output_file}.")
        return

    # Variables de procesamiento
    last_boxes, last_class_ids = [], []
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

        # Procesar frame
        last_boxes, last_class_ids = perform_yolo_detection(model, frame)
        train_moving, train_stopped_counter, frames_wait_procces_roi, consecutive_moving_frames = evaluate_train_state(
            frame, prev_frame, train_moving, train_stopped_counter, frames_wait_procces_roi, consecutive_moving_frames, polygon_metro
        )

        # Dibujar ROIs
        
        draw_polygon(frame, polygon_metro)
        print("TREN EN MOVIMIENTO", train_moving)
        if train_moving:
            draw_detections(frame, last_boxes, last_class_ids, polygon_safe_zone, classes)
        else:
            cv2.putText(frame, "Tren detenido", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, GREEN_COLOR, 2)
            cv2.putText(frame, "Alerta desactivada", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, GREEN_COLOR, 2)

        # Guardar el frame procesado en el archivo de salida
        display_frame(frame)
        out.write(frame)
        prev_frame = frame.copy()

        # Actualizar la barra de progreso (si se ha pasado como parámetro)
        if progress_bar:
            frame_count += 1
            progress = (frame_count / total_frames) * 100
            progress_bar["value"] = progress
            progress_bar.update_idletasks()  # Actualizar la barra visualmente

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Video procesado guardado en: {output_file}")
