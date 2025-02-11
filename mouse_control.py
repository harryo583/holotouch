"""
This script enables controlling the mouse cursor and performing mouse clicks based on hand gestures
detected using Mediapipe and OpenCV.
It captures video from the webcam, processes hand landmarks to identify gestures, and translates these gestures into mouse movements and clicks.
"""

import cv2
import mediapipe as mp
import math
import pyautogui
from timer import Timer
from drawer import draw_dashed_line

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

resolution_x = 1280
resolution_y = 720

def log_call(func):
    def wrapper(*args, **kwargs):
        # Debug: print(f"Executing {func.__name__}")
        return func(*args, **kwargs)
    return wrapper

class MouseController:
    def __init__(self):
        self.pinch_timer = Timer()
        self.last_pinch_time = None
        self.current_mouse_x, self.current_mouse_y = pyautogui.position()

    @log_call
    def draw_mouse(self, frame, x: int, y: int) -> None:
        """
        Draws a visual indicator for the mouse position on the frame.
        """
        radius = 10
        color = (0, 255, 0)
        thickness = 2
        cv2.circle(frame, (x, y), radius, color, thickness)
        cv2.putText(frame, f"Mouse: ({x}, {y})", (x + 20, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    @log_call
    def move_mouse(self, x: int, y: int, smooth_factor: float = 0.9) -> None:
        """
        Moves the mouse cursor to the specified position with optional smoothing.
        """
        screen_width, screen_height = pyautogui.size()
        target_x = int((x / resolution_x) * screen_width)
        target_y = int((y / resolution_y) * screen_height)
        
        self.current_mouse_x = self.current_mouse_x * (1 - smooth_factor) + target_x * smooth_factor
        self.current_mouse_y = self.current_mouse_y * (1 - smooth_factor) + target_y * smooth_factor
        
        pyautogui.moveTo(int(self.current_mouse_x), int(self.current_mouse_y))

    @log_call
    def perform_click(self, distance: float) -> None:
        """
        Performs a mouse click based on the distance between thumb and index finger.
        """
        pinch_threshold = 30
        double_click_gap = 0.3  # Time gap to register a double click
        long_pinch_duration = 2  # Duration for recognizing a long pinch
        
        if distance < pinch_threshold:
            if self.pinch_timer.start_time is None:
                self.pinch_timer.start()  # Start the pinch timer
            elif self.pinch_timer.get_elapsed_time() > long_pinch_duration:
                pyautogui.rightClick()  # Long pinch, right click
                self.pinch_timer.reset()  # Reset after a right click
            elif self.last_pinch_time is not None and self.pinch_timer.get_elapsed_time() < double_click_gap:
                pyautogui.doubleClick()  # Double pinch
                self.pinch_timer.reset()  # Reset the timer
                self.last_pinch_time = None
            else:
                if self.pinch_timer.get_elapsed_time() < 1:
                    pyautogui.click()  # Short pinch, left click
                    self.last_pinch_time = self.pinch_timer.get_elapsed_time()
                    self.pinch_timer.reset()  # Reset the timer
        else:
            self.pinch_timer.reset()  # Reset the timer if no pinch

    @log_call
    def process(self, frame, landmarks) -> None:
        """
        Processes hand landmarks to control the mouse cursor and perform clicks.
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
        
        self.move_mouse(x2, y2)
        self.perform_click(distance)
        self.draw_mouse(frame, x2, y2)

# Expose a callable instance to preserve the original interface
_mouse_controller_instance = MouseController()
mouse_controller = _mouse_controller_instance.process
