MOVEMENT_THRESHOLD = 700 
# Constantes de Configuracion
CONFIDENCE_THRESHOLD = 0.5
FRAME_SKIP = 3 
SAFE_WIDTH = 20
DATA_VIDEO = './data/example01.mp4'
YOLO_MODEL_PATH = "./models/yolov8/yolov8n.pt"
CLASSES_FILE = './models/yolov8/coco.names'
ROIS = [(291, 196, 347, 418),
(325, 325, 426, 692),
(291, 336, 414, 544)]

# Colores
YELLOW_COLOR = (0, 255, 255)
GREEN_COLOR = (0, 255, 0)
ALERT_COLOR = (0, 0, 255)
ROI_COLOR = (180, 180, 180)