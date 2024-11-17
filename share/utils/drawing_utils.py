import cv2
from share.constants.config import ALERT_COLOR, GREEN_COLOR, ROI_COLOR, ROIS, YELLOW_COLOR

def draw_detections(frame, boxes, class_ids, confidences, yellow_band, green_band, classes):
    """
    Dibuja las detecciones sobre el frame, incluyendo alertas.
    """     
    for i in range(len(boxes)):
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = confidences[i]
        cv2.rectangle(frame, (x, y), (x + w, y + h), GREEN_COLOR, 2)
        cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN_COLOR, 2)

        person_center_x = x + w // 2
        if green_band is not None:
            if cv2.pointPolygonTest(green_band, (person_center_x, y + h), False) >= 0:
                cv2.putText(frame, "ALERTA: Persona cruzando la linea", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, ALERT_COLOR, 2)

    if yellow_band is not None:
        cv2.drawContours(frame, [yellow_band], -1, YELLOW_COLOR, 2)
    if green_band is not None:
        cv2.drawContours(frame, [green_band], -1, GREEN_COLOR, 2)