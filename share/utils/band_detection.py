
import cv2
import numpy as np

from share.constants.config import MOVEMENT_THRESHOLD


def detect_yellow_band(frame):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([20, 60, 100])
    upper_yellow = np.array([40, 180, 255])
    yellow_mask = cv2.inRange(hsv_frame, lower_yellow, upper_yellow)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 20))
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)
    yellow_mask = cv2.dilate(yellow_mask, kernel, iterations=2)
    yellow_contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    yellow_band = None
    if yellow_contours:
        largest_yellow_contour = max(yellow_contours, key=cv2.contourArea)
        yellow_band = cv2.convexHull(largest_yellow_contour)
        
    return yellow_band


def check_train_movement_in_rois(frame, rois, prev_frame,threshold):
    movement_detected = False
    for roi in rois:
        x1, y1, x2, y2 = roi
        roi_frame = frame[y1:y2, x1:x2]
        prev_roi_frame = prev_frame[y1:y2, x1:x2]
        
        movement_in_roi = detect_movement_in_roi(prev_roi_frame, roi_frame, (0, 0, x2-x1, y2-y1),threshold )
        
        if movement_in_roi:
            return True
        
    return movement_detected

def select_roi():
    image = cv2.imread('./data/example01.jpg')

    # Select the region of interest manually
    roi = cv2.selectROI("Select ROI", image, fromCenter=False, showCrosshair=True)

    # Display the image with the ROI selected
    x, y, w, h = roi
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow("ROI Selected", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(f"ROI Coordinates: (x1={x}, y1={y}, x2={x + w}, y2={y + h})")


def detect_movement_in_roi(prev_frame, curr_frame, roi_coords, threshold=MOVEMENT_THRESHOLD):
    PIXEL_MAX_VALUE = 255
    x1, y1, x2, y2 = roi_coords
        
    # Extract the region of interest from both frames
    prev_roi = prev_frame[y1:y2, x1:x2]
    curr_roi = curr_frame[y1:y2, x1:x2]
     
    diff = cv2.absdiff(prev_roi, curr_roi)
    
    # Apply a threshold to reduce noise and focus on significant changes
    _, diff_thresh = cv2.threshold(diff, 30, PIXEL_MAX_VALUE, cv2.THRESH_BINARY)
    
    # Counts the different pixels (motion) in the ROI
    movement_pixels = np.sum(diff_thresh) / PIXEL_MAX_VALUE
    
    # Detects if there is enough movement to consider that the train is moving
    return movement_pixels > threshold
