
import cv2
import numpy as np

from share.constants.config import MOVEMENT_THRESHOLD


def detect_yellow_and_green_bands(frame, scale_factor=1.4):
    """
    Detecta la banda amarilla y su extensión en verde.
    """
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


def check_train_movement_in_rois(frame, rois, prev_frame):
    movement_detected = False
    for roi in rois:
        x1, y1, x2, y2 = roi
        roi_frame = frame[y1:y2, x1:x2]
        prev_roi_frame = prev_frame[y1:y2, x1:x2]
        
        # Realiza la detección de movimiento en cada sub-ROI
        movement_in_roi = detect_movement_in_roi(prev_roi_frame, roi_frame, (0, 0, x2-x1, y2-y1))
        
        if movement_in_roi:
            return True  # Si se detecta movimiento en cualquier ROI, podemos salir y devolver True
        
    return movement_detected  # Devuelve False si no se detecta movimiento en ninguna ROI

def select_roi():
    # Cargar la imagen
    image = cv2.imread('./data/example01.jpg')

    # Seleccionar la región de interés manualmente
    roi = cv2.selectROI("Select ROI", image, fromCenter=False, showCrosshair=True)

    # Mostrar la imagen con la ROI seleccionada
    x, y, w, h = roi
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow("ROI Selected", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # Imprimir las coordenadas de la ROI
    print(f"ROI Coordinates: (x1={x}, y1={y}, x2={x + w}, y2={y + h})")


def detect_movement_in_roi(prev_frame, curr_frame, roi_coords, threshold=MOVEMENT_THRESHOLD):
    x1, y1, x2, y2 = roi_coords
        
    # Extrae la región de interés de ambos cuadros
    prev_roi = prev_frame[y1:y2, x1:x2]
    curr_roi = curr_frame[y1:y2, x1:x2]
    
    # Convierte la ROI a escala de grises
    prev_gray = cv2.cvtColor(prev_roi, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_roi, cv2.COLOR_BGR2GRAY)
    
    # Calcula la diferencia absoluta entre cuadros consecutivos en la ROI
    diff = cv2.absdiff(prev_gray, curr_gray)
    
    # Aplica un umbral para reducir el ruido y enfocar los cambios significativos
    _, diff_thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    
    # Cuenta los píxeles diferentes (movimiento) en la ROI
    movement_pixels = np.sum(diff_thresh) / 255
    
    # Detecta si hay movimiento suficiente para considerar que el tren está en movimiento
    return movement_pixels > threshold
