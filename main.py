import cv2
import numpy as np
from utils.line_detection import get_line_yellow_of_frame
 

# Configuración de YOLO
YOLO_WEIGHTS = './models/yolo/yolov3.weights'
YOLO_CONFIG = './models/yolo/yolov3.cfg'
COCO_NAMES = './models/yolo/coco.names'

# Parámetros de detección
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4
FRAME_SKIP = 3  # Detectar cada 3 frames
YOLO_INPUT_SIZE = (416, 416)

DATA_VIDEO = './data/example02.mp4'

def load_yolo():
    """Cargar el modelo YOLO."""
    net = cv2.dnn.readNet(YOLO_WEIGHTS, YOLO_CONFIG)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return net, output_layers

def perform_yolo_detection(net, output_layers, frame, classes):
    """Realizar detección de personas usando YOLO."""
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.00392, YOLO_INPUT_SIZE, (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    boxes, confidences, class_ids = [], [], []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > CONFIDENCE_THRESHOLD and class_id == classes.index("person"):
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w, h = int(detection[2] * width), int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Aplicar Non-Maximum Suppression
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    return boxes, class_ids, indexes

def draw_detections(frame, boxes, class_ids, indexes, line_x, classes):
    """Dibujar las detecciones de personas y la línea amarilla."""
    for i in indexes.flatten():  # Asegúrate de aplanar los índices
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Verificar si la persona cruza la línea amarilla
        person_center_x = x + w // 2
        if line_x is not None and person_center_x > line_x:
            cv2.putText(frame, "ALERTA: Persona cruzando la linea", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

def main():
    """Función principal para procesar el video y detectar personas y línea amarilla."""
    # Cargar modelos y clases
    net, output_layers = load_yolo()
    with open(COCO_NAMES, "r") as f:
        classes = f.read().strip().split("\n")

    # Capturar video
    cap = cv2.VideoCapture(DATA_VIDEO)
    if not cap.isOpened():
        print("Error: No se pudo abrir el video.")
        return

    first_yellow_line = None
    frame_count = 0
    last_boxes, last_class_ids, last_indexes = [], [], []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detectar línea amarilla
        if first_yellow_line is None:
            first_yellow_line = get_line_yellow_of_frame(frame)
        
        line_x = first_yellow_line

        # Dibujar la línea amarilla en el frame
        if line_x is not None:
            cv2.line(frame, (line_x, 0), (line_x, frame.shape[0]), (0, 255, 255), 2)
        
        # Solo realizar detección cada 3 frames
        if frame_count % FRAME_SKIP == 0:
            last_boxes, last_class_ids, last_indexes = perform_yolo_detection(net, output_layers, frame, classes)
        
        draw_detections(frame, last_boxes, last_class_ids, last_indexes, line_x, classes)

        # Mostrar el video
        cv2.imshow('Deteccion de Personas y Linea Amarilla Vertical', frame)

        # Incrementar contador de frames
        frame_count += 1

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
