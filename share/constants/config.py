MOVEMENT_THRESHOLD = 700
CONFIDENCE_THRESHOLD = 0.6
FRAME_SKIP = 2
SAFE_WIDTH = 20
DATA_VIDEO = './data/example01.mp4'
YOLO_MODEL_PATH = "./models/yolov8/yolov8n.pt"
CLASSES_FILE = './models/yolov8/coco.names'
ROIS = [(291, 196, 347, 418),
(325, 325, 426, 692),
(291, 336, 414, 544)]

# Colors
YELLOW_COLOR = (0, 255, 255)
GREEN_COLOR = (0, 255, 0)
ALERT_COLOR = (0, 0, 255)
ROI_COLOR = (180, 180, 180)

# Thresholds for significant movement on the train
MOVEMENT_THRESHOLD_METRO_MOVE = 0.12 # Threshold for detecting train movement
MOVEMENT_THRESHOLD_METRO_STOP = 0.11 # Threshold to detect when the train stops
STOPPED_FRAME_THRESHOLD = 5  # Increased to require more stopped frames