import cv2
import numpy as np
import torch
from ultralytics import YOLO

from utils.line_detection import get_line_yellow_of_frame

# Parámetros de detección
CONFIDENCE_THRESHOLD = 0.5
FRAME_SKIP = 3  # Detectar cada 3 frames

DATA_VIDEO = './data/example01.mp4'

def perform_yolo_detection(model, frame):
    results = model(frame)
    boxes, confidences, class_ids = [], [], []

    for result in results:
        # Obtener información de las detecciones
        for box in result.boxes:
            if box.conf[0] > CONFIDENCE_THRESHOLD and int(box.cls[0]) == 0:  # Class '0' is 'person'
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                boxes.append([x1, y1, x2 - x1, y2 - y1])
                confidences.append(confidence)
                class_ids.append(int(box.cls[0]))

    return boxes, class_ids, confidences

def draw_detections(frame, boxes, class_ids, confidences, line_x, classes):
    """Draw the people detections and the yellow line."""
    for i in range(len(boxes)):
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = confidences[i]

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Check if the person crosses the yellow line
        person_center_x = x + w // 2
        if line_x is not None and person_center_x > line_x:
            cv2.putText(frame, "ALERTA: Persona cruzando la linea", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

def main():
    model = YOLO("./models/yolov8/yolov8n.pt").to("cuda" if torch.cuda.is_available() else "cpu")
    
    with open('./models/yolo/coco.names', "r") as f:
        classes = f.read().strip().split("\n")

    cap = cv2.VideoCapture(DATA_VIDEO)
    if not cap.isOpened():
        print("Error: Could not open the video.")
        return

    first_yellow_line = None
    frame_count = 0
    last_boxes, last_class_ids, last_confidences = [], [], []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detectar línea amarilla
        if first_yellow_line is None:
            first_yellow_line = get_line_yellow_of_frame(frame)
        
        line_x = first_yellow_line

        # Draw the yellow line on the frame
        if line_x is not None:
            cv2.line(frame, (line_x, 0), (line_x, frame.shape[0]), (0, 255, 255), 2)
        
        # Only perform detection every 3 frames
        if frame_count % FRAME_SKIP == 0:
            last_boxes, last_class_ids, last_confidences = perform_yolo_detection(model, frame)
        
        draw_detections(frame, last_boxes, last_class_ids, last_confidences, line_x, classes)

        # Show the video
        cv2.imshow('Deteccion de Personas y Linea Amarilla Vertical', frame)

        frame_count += 1

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
