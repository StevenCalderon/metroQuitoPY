import cv2
import numpy as np
from utils.line_detection import get_line_yellow_of_frame  

# Cargar YOLOv3
net = cv2.dnn.readNet('./models/yolo/yolov3.weights', './models/yolo/yolov3.cfg')
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
cap = cv2.VideoCapture('./data/example01.mp4')

# Cargar el archivo de nombres de las clases
classes = []
with open("./models/yolo/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Leer el primer frame para detectar la línea amarilla
ret, frame = cap.read()
if not ret:
    print("No se pudo leer el video.")
    cap.release()
    cv2.destroyAllWindows()
    exit()

#line_x = get_line_yellow_of_frame(frame)

# Inicializar contador de frames
frame_count = 0
frame_skip = 3  # Saltar 2 frames, es decir, detectar cada 3 frames

# Ciclo para procesar el video
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    line_x = get_line_yellow_of_frame(frame)

    # Dibujar la línea amarilla detectada en todos los frames
    if line_x is not None:
        cv2.line(frame, (line_x, 0), (line_x, frame.shape[0]), (0, 255, 255), 2)

    # Solo realizar inferencia con YOLOv3 cada 3 frames
    if frame_count % frame_skip == 0:
        height, width, channels = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5 and class_id == classes.index("person"):  # Detectar solo personas
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Aplicar Non-Maximum Suppression para eliminar cajas redundantes
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Dibujar cajas alrededor de las personas detectadas y verificar si cruzan la línea amarilla
    for i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Mejorar la detección del cruce, usando el centro del bounding box
        person_center_x = x + w // 2
        if line_x is not None and person_center_x > line_x:
            cv2.putText(frame, "ALERTA: Persona cruzando la linea", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Mostrar el video con las detecciones
    cv2.imshow('Deteccion de Personas y Linea Amarilla Vertical', frame)

    # Incrementar el contador de frames
    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
