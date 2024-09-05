"""
This script contains a function to control the volume based on hand gestures using Mediapipe and OpenCV.
It processes hand landmarks to calculate the distance between two points (thumb tip and index finger tip) and adjusts the volume accordingly.

Dependencies:
    - OpenCV (cv2)
    - Mediapipe (mp)
    - Math (math)
    - Subprocess (subprocess)
    - Custom Drawer (drawer)
"""

import cv2
import mediapipe as mp
import math
import subprocess
from drawer import draw_dashed_line, draw_slider

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def volume_controller(frame, landmarks):
    """
    Adjusts the volume based on the distance between the thumb tip and index finger tip.
    """
    
    point_A = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    point_B = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    
    # Convert normalized coordinates to image coordinates
    height, width, _ = frame.shape
    x1, y1 = int(point_A.x * width), int(point_A.y * height)
    x2, y2 = int(point_B.x * width), int(point_B.y * height)
    
    # Draw dashed line and circles on the detected points
    draw_dashed_line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2, 8, 16)
    cv2.circle(frame, (x1, y1), 15, (0, 0, 255), -1)
    cv2.circle(frame, (x2, y2), 15, (0, 0, 255), -1)
    
    # Calculate the Euclidean distance between the points
    distance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    
    # Map the distance to the volume range (0-100 for macOS)
    volume_level = int(min(100, max(0, distance/3.8 - 8)))
    
    # Display volume level and adjust volume
    cv2.putText(frame, f"Volume level: {volume_level}%", (40, height-40), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 5, cv2.LINE_AA)
    volume = max(0, min(100, volume_level))
    subprocess.run(["osascript", "-e", f"set volume output volume {volume}"])
    
    # Draw the slider to indicate the volume level
    draw_slider(frame, volume_level)