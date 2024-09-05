"""
This script enables controlling the mouse cursor and performing mouse clicks based on hand gestures detected using Mediapipe and OpenCV.
It captures video from the webcam, processes hand landmarks to identify gestures, and translates these gestures into mouse movements and clicks.

Dependencies:
    - OpenCV (cv2)
    - Mediapipe (mp)
    - Math (math)
    - Pyautogui (pyautogui)
    - Custom Timer (timer)
    - Custom Drawer (drawer)
"""

import cv2
import mediapipe as mp
import math
import pyautogui
from timer import Timer
from drawer import draw_dashed_line

resolution_x = 1280
resolution_y = 720

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def draw_mouse(frame, x, y):
    """
    Draws a visual indicator for the mouse position on the frame.
    """
    radius = 10
    color = (0, 255, 0)
    thickness = 2
    cv2.circle(frame, (x, y), radius, color, thickness)
    cv2.putText(frame, f"Mouse: ({x}, {y})", (x + 20, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

def move_mouse(x, y, smooth_factor=0.9):
    """
    Moves the mouse cursor to the specified position with optional smoothing.
    """
    screen_width, screen_height = pyautogui.size()
    target_x = int((x / resolution_x) * screen_width)
    target_y = int((y / resolution_y) * screen_height)
    
    global current_mouse_x, current_mouse_y
    current_mouse_x = current_mouse_x * (1 - smooth_factor) + target_x * smooth_factor
    current_mouse_y = current_mouse_y * (1 - smooth_factor) + target_y * smooth_factor
    
    pyautogui.moveTo(int(current_mouse_x), int(current_mouse_y))

# Initialize the pinch timer and set other variables
pinch_timer = Timer()
last_pinch_time = None
pinched = False

def perform_click(distance):
    """
    Performs a mouse click based on the distance between thumb and index finger.
    """
    global last_pinch_time, pinched
    pinch_threshold = 30
    double_click_gap = 0.3  # Time gap to register a double click
    long_pinch_duration = 2  # Duration for recognizing a long pinch
    
    if distance < pinch_threshold:
        if pinch_timer.start_time is None:
            pinch_timer.start()  # Start the pinch timer
        elif pinch_timer.get_elapsed_time() > long_pinch_duration:
            pyautogui.rightClick()  # Long pinch, right click
            pinch_timer.reset()  # Reset after a right click
        elif last_pinch_time is not None and pinch_timer.get_elapsed_time() < double_click_gap:
            pyautogui.doubleClick()  # Double pinch
            pinch_timer.reset()  # Reset the timer
            last_pinch_time = None
        else:
            if pinch_timer.get_elapsed_time() < 1:
                pyautogui.click()  # Short pinch, left click
                last_pinch_time = pinch_timer.get_elapsed_time()
                pinch_timer.reset()  # Reset the timer
    else:
        pinch_timer.reset()  # Reset the timer if no pinch

def mouse_controller(frame, landmarks):
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
    
    move_mouse(x2, y2)
    perform_click(distance)
    draw_mouse(frame, x2, y2)

# Initialize the current mouse position
current_mouse_x, current_mouse_y = pyautogui.position()

# Example usage; called only when script is run
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            
            # Draw hand landmarks
            if results.multi_hand_landmarks:
                for landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
                    mouse_controller(frame, landmarks)
            
            cv2.imshow('Mouse Controller', frame)
            
            if cv2.waitKey(1) == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
