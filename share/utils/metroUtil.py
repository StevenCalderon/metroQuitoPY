import cv2
import numpy as np

from share.constants.config import MOVEMENT_THRESHOLD


def detect_train_movementaaaaaa(prev_frame, curr_frame, threshold=MOVEMENT_THRESHOLD):
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
