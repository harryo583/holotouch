"""
This script contains a function to control the screen brightness based on hand gestures using Mediapipe and OpenCV.
It processes hand landmarks to calculate the distance between two points (thumb tip and index finger tip) and adjusts the screen brightness accordingly.

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
from drawer import draw_dashed_line
    
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def brightness_controller(frame, landmarks):
    """
    Adjusts the screen brightness based on the distance between the thumb tip and index finger tip.
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
        
    # Map the distance to the brightness range as percentage
    brightness_level = int(min(100, max(0, distance/3.8 - 8)))
    
    # Display brightness level and adjust brightness
    cv2.putText(frame, f"Brightness level: {brightness_level}%", (40, height-40), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 5, cv2.LINE_AA)
    brightness = int(brightness_level/10)/10
    subprocess.run(["brightness", str(brightness)])