import cv2
import numpy as np
from share.constants.configConstants import MOVEMENT_THRESHOLD

def detect_train_movement(prev_frame, curr_frame, threshold=MOVEMENT_THRESHOLD):
    # Convierte los cuadros a escala de grises
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    
    # Calcula la diferencia absoluta entre cuadros consecutivos
    diff = cv2.absdiff(prev_gray, curr_gray)
    
    # Aplica un umbral para reducir el ruido y enfocar los cambios significativos
    _, diff_thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    
    # Cuenta los píxeles diferentes (movimiento)
    movement_pixels = np.sum(diff_thresh) / 255
    
    # Detecta si hay movimiento suficiente para considerar que el tren está en movimiento
    return movement_pixels > threshold

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

def check_train_movement_in_rois(frame, rois, prev_frame):
    movement_detected = False
    for roi in rois:
        x1, y1, x2, y2 = roi
        roi_frame = frame[y1:y2, x1:x2]
        prev_roi_frame = prev_frame[y1:y2, x1:x2]
        
        # Realiza la detección de movimiento en cada sub-ROI
        movement_in_roi = detect_movement_in_roi(prev_roi_frame, roi_frame, (0, 0, x2-x1, y2-y1))
        
        if movement_in_roi:
            movement_detected = True
            break  # Si se detecta movimiento en cualquier ROI, podemos salir
        
    return movement_detected