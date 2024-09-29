import cv2
import numpy as np

def get_line_yellow_of_frame(frame):
    line = detect_line_yellow(frame)
    if line is not None:
        x1, y1, x2, y2 = line[0]
        line_x = (x1 + x2) // 2 
        print(f"Línea detectada en x: {line_x}")
    else:
        line_x = None 
        print("No se detectó ninguna línea amarilla.")
    return line_x
    

def get_longest_line(lines):
    if not lines:
        return None
    return max(lines, key=lambda line: abs(line[0][2] - line[0][0]))

def detect_line_yellow(frame):
    # Convertir la imagen a espacio de color HSV
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Definir un rango de color amarillo ajustado
    lower_yellow = np.array([20, 40, 100])  # Rango inferior ajustado
    upper_yellow = np.array([40, 255, 255])  # Rango superior ajustado
    
    # Crear una máscara que capture solo los pixeles amarillos
    mask_yellow = cv2.inRange(hsv_frame, lower_yellow, upper_yellow)
    
    # Detectar líneas usando HoughLines o HoughLinesP
    lines = cv2.HoughLinesP(mask_yellow, 1, np.pi/180, 50, minLineLength=frame.shape[0]//2, maxLineGap=20)
    
    # Dibujar la línea amarilla detectada en la imagen
    vertical_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(x2 - x1) < 10:  # Ajusta este valor según sea necesario
                vertical_lines.append(line)
    return get_longest_line(vertical_lines)
