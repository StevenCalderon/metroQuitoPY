import cv2
import numpy as np
import os
import pygame
from threading import Thread
import time

from share.constants.config import (ALERT_COLOR, GREEN_COLOR, ROI_COLOR,
                                    YELLOW_COLOR)

class SoundPlayer:
    def __init__(self):
        current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.sound_file = os.path.join(current_dir, "resources", "sounds", "beep.mp3")
        self.is_playing = False
        self.play_thread = None
        pygame.mixer.init()

    def play_loop(self):
        try:
            if os.path.exists(self.sound_file):
                pygame.mixer.music.load(self.sound_file)
                pygame.mixer.music.play(-1)
            else:
                print(f"Error: Sound file not found in {self.sound_file}")
        except Exception as e:
            print(f"Error playing sound: {e}")

    def stop(self):
        try:
            pygame.mixer.music.stop()
        except Exception as e:
            print(f"Error stopping sound: {e}")

sound_player = SoundPlayer()

def draw_detections(frame, boxes, class_ids, polygon_safe_zone, classes, train_moving=True):
    global sound_player
    person_in_zone = False

    for i, box in enumerate(boxes):
        x, y, w, h = box

        w = int(w * 1.2)
        x = x - int(w * 0.1)
        x = max(x, 0)

        # Check if the person crosses the yellow stripe using cv2.pointPolygonTest
        if polygon_safe_zone is not None:
            # Convert yellow_band_points to a NumPy array with correct shape (n,1,2)
            yellow_band_polygon = np.array(polygon_safe_zone, dtype=np.int32).reshape((-1, 1, 2))

            # Get the center of the person's box
            person_center = (x + w // 2, y + h // 2)

            # Use cv2.pointPolygonTest to check if the center of the person is inside the polygon
            distance = cv2.pointPolygonTest(yellow_band_polygon, person_center, False)

            # If the point is inside the polygon, the distance will be positive.
            if distance >= 0:
                person_in_zone = True
                cv2.putText(frame, "Cruzando la franja amarilla", (x, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ALERT_COLOR, 2)
                cv2.putText(frame, "ALERTA: Cruzan la linea", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, ALERT_COLOR, 2)

        if polygon_safe_zone is not None:
            cv2.drawContours(frame, [yellow_band_polygon], -1, YELLOW_COLOR, 2)

        # Draw the boxes with the new width
        color = GREEN_COLOR
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    if person_in_zone and train_moving and not sound_player.is_playing:
        sound_player.is_playing = True
        sound_player.play_thread = Thread(target=sound_player.play_loop)
        sound_player.play_thread.start()
    elif (not person_in_zone or not train_moving) and sound_player.is_playing:
        sound_player.stop()
        sound_player.is_playing = False