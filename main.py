import cv2
import numpy as np
import torch
from ultralytics import YOLO

from share.utils.metroUtil import check_train_movement_in_rois

CONFIDENCE_THRESHOLD = 0.5
FRAME_SKIP = 3 
SAFE_WIDTH = 20
DATA_VIDEO = './data/example01.mp4'
rois = [(274, 193, 310, 288), (295, 321, 416, 473), (358, 590, 471, 843)]

def detect_yellow_and_green_bands(frame, scale_factor=1.4):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([20, 60, 100])
    upper_yellow = np.array([40, 180, 255])
    yellow_mask = cv2.inRange(hsv_frame, lower_yellow, upper_yellow)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 20))
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)
    yellow_mask = cv2.dilate(yellow_mask, kernel, iterations=2)
    yellow_contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    yellow_band, green_extension_band = None, None
    if yellow_contours:
        largest_yellow_contour = max(yellow_contours, key=cv2.contourArea)
        yellow_band = cv2.convexHull(largest_yellow_contour)
        M = cv2.moments(yellow_band)
        cx = int(M["m10"] / M["m00"]) if M["m00"] != 0 else 0

        green_extension_band = []
        for point in yellow_band:
            x, y = point[0]
            new_x = int(cx + scale_factor * (x - cx))
            green_extension_band.append([[new_x, y]])
        
        green_extension_band = np.array(green_extension_band, dtype=np.int32)
        green_right = np.max(green_extension_band[:, 0, 0])
        yellow_right = np.max(yellow_band[:, 0, 0])
        displacement = yellow_right - green_right
        green_extension_band[:, 0, 0] += displacement

    return yellow_band, green_extension_band


def perform_yolo_detection(model, frame):
    results = model(frame)
    boxes, confidences, class_ids = [], [], []

    for result in results:
        for box in result.boxes:
            if box.conf[0] > CONFIDENCE_THRESHOLD and int(box.cls[0]) == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                boxes.append([x1, y1, x2 - x1, y2 - y1])
                confidences.append(confidence)
                class_ids.append(int(box.cls[0]))

    return boxes, class_ids, confidences

def draw_detections(frame, boxes, class_ids, confidences, yellow_band, green_band, classes):
    for i in range(len(boxes)):
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = confidences[i]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        person_center_x = x + w // 2
        if green_band is not None:
            if cv2.pointPolygonTest(green_band, (person_center_x, y + h), False) >= 0:
                cv2.putText(frame, "ALERTA: Persona cruzando la linea", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    if yellow_band is not None:
        cv2.drawContours(frame, [yellow_band], -1, (0, 255, 255), 2)
    if green_band is not None:
        cv2.drawContours(frame, [green_band], -1, (0, 255, 0), 2)

def main():
    model = YOLO("./models/yolov8/yolov8n.pt").to("cuda" if torch.cuda.is_available() else "cpu")
    
    with open('./models/yolov8/coco.names', "r") as f:
        classes = f.read().strip().split("\n")

    cap = cv2.VideoCapture(DATA_VIDEO)
    if not cap.isOpened():
        print("Error: Could not open the video.")
        return

    last_boxes, last_class_ids, last_confidences = [], [], []
    band_points, green_band_points = None, None
    
    # Lee el primer cuadro
    ret, prev_frame = cap.read()
    frame_count = 0
    train_moving = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if (band_points is None or green_band_points is None or not band_points.size or not green_band_points.size):
            band_points, green_band_points = detect_yellow_and_green_bands(frame)
        #band_points, green_band_points = detect_yellow_and_green_bands(frame)
        
        if frame_count % FRAME_SKIP == 0:
            last_boxes, last_class_ids, last_confidences = perform_yolo_detection(model, frame)
          
        # Detecta movimiento solo en la ROI del tren
        train_moving = check_train_movement_in_rois(prev_frame,rois, frame)
        
        # Dibuja la ROI en el cuadro actual
        for roi in rois:
            x1, y1, x2, y2 = roi
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2) 
        
        # Muestra estado del tren
        #text = "Tren en movimiento" if train_moving else "Tren detenido"
        #color = (0, 0, 255) if train_moving else (0, 255, 0)
        #cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Actualiza el cuadro anterior
        prev_frame = frame
        
        if train_moving:
           draw_detections(frame, last_boxes, last_class_ids, last_confidences, band_points, green_band_points, classes)
        else:
            cv2.putText(frame, "Tren detenido - Alerta desactivada", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Mostrar el video
        cv2.imshow('Deteccion de Personas y Franja Amarilla', frame)
        frame_count += 1

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    #select_roi()