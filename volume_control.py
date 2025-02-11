"""
This script contains a class to control the volume based on hand gestures using Mediapipe and OpenCV.
It processes hand landmarks to calculate the distance between two points (thumb tip and index finger tip)
and adjusts the volume accordingly.
"""

import cv2
import mediapipe as mp
import math
import subprocess
from contextlib import contextmanager
from drawer import draw_dashed_line, draw_slider

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def log_call(func):
    def wrapper(*args, **kwargs):
        # Debug: print(f"Calling {func.__name__}")
        return func(*args, **kwargs)
    return wrapper

@contextmanager
def run_subprocess(command: list):
    try:
        subprocess.run(command)
        yield
    except Exception as e:
        raise e

class VolumeController:
    @log_call
    def __call__(self, frame, landmarks) -> None:
        """
        Adjusts the volume based on the distance between the thumb tip and index finger tip.
        """
        point_A = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        point_B = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        
        height, width, _ = frame.shape
        x1, y1 = int(point_A.x * width), int(point_A.y * height)
        x2, y2 = int(point_B.x * width), int(point_B.y * height)
        
        draw_dashed_line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2, 8, 16)
        cv2.circle(frame, (x1, y1), 15, (0, 0, 255), -1)
        cv2.circle(frame, (x2, y2), 15, (0, 0, 255), -1)
        
        distance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        volume_level = int(min(100, max(0, distance / 3.8 - 8)))
        
        cv2.putText(frame, f"Volume level: {volume_level}%", (40, height - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 5, cv2.LINE_AA)
        volume = max(0, min(100, volume_level))
        with run_subprocess(["osascript", "-e", f"set volume output volume {volume}"]):
            pass
        
        draw_slider(frame, volume_level)

# Expose a callable instance to preserve the original interface
volume_controller = VolumeController()
