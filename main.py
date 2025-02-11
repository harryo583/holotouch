"""
Gesture-Controlled PC Interaction System

This script uses Mediapipe and TensorFlow to control PC functions (volume, brightness, and mouse) through hand gestures.
It captures video from the webcam, processes hand landmarks to identify gestures, and then triggers corresponding control actions.
"""

import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np
from collections import deque
from timer import Timer
import volume_control
import brightness_control
import mouse_control

##############
## Constants
##############

ZERO = 0        # Empty fist
INDEX = 1       # Index finger pointing up
PEACE = 2       # Two fingers extended
THREE = 3       # Three fingers extended
PALM = 4        # All fingers extended
THUMB = 5       # Thumb extended

##############################
## User-adjustable constants
##############################

wait_time = 1.5  # Time in seconds a gesture must be held to trigger a control action change

resolution_x = 1280
resolution_y = 720

mouse_control_activation_signal = INDEX
mouse_control_deactivation_signal = ZERO
brightness_control_signal = PEACE
volume_control_signal = THREE

brightness_control_time = 3  # seconds given for brightness control
volume_control_time = 3      # seconds given for volume control

##########################
## Function definitions
##########################

def preprocess(landmarks, label):
    """
    Preprocesses the hand landmarks to normalize and prepare input for the model.
    Returns a list of normalized landmark coordinates, with the label appended if provided.
    """
    row = []
    base_x, base_y = landmarks.landmark[0].x, landmarks.landmark[0].y
    
    for landmark in landmarks.landmark:
        row.extend([landmark.x - base_x, landmark.y - base_y])
    
    max_value = max(abs(min(row)), max(row))
    row = list(map(lambda x: x / max_value, row))
    
    if label is not None:
        row.append(label)
    
    return row

class VideoCaptureCM:
    """
    Context manager for cv2.VideoCapture.
    """
    def __init__(self, source=0):
        self.cap = cv2.VideoCapture(source)
    
    def __enter__(self):
        if not self.cap.isOpened():
            raise RuntimeError("Error: failed to access local camera device")
        return self.cap
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cap.release()

def main():
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    model = tf.keras.models.load_model("model/hand_landmarks_model.h5")
    cache_size = int(wait_time * 10)
    gesture_cache = deque(maxlen=cache_size)
    
    # Initialize timers and control flags
    activation_timer = Timer()
    activation_timer.start()
    brightness_activated = False
    volume_activated = False
    mouse_activated = False
    mouse_cleared = True

    with VideoCaptureCM(0) as cap, mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5) as hands:
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print('Failed to grab frame')
                break
            
            frame = cv2.flip(frame, 1)  # Mirror view
            color_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(color_frame)
            
            if results.multi_hand_landmarks:
                landmarks = results.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
                input_tensor = tf.expand_dims(tf.convert_to_tensor(preprocess(landmarks, None)), axis=0)
                prediction = model.predict(input_tensor)
    
                # Update gesture cache with latest prediction
                if np.max(prediction) >= 0.9:
                    gesture = int(np.argmax(prediction, axis=1)[0])
                    gesture_cache.append(gesture)
                
                # Mouse control activation
                if (len(gesture_cache) == cache_size and all(g == mouse_control_activation_signal for g in gesture_cache)) or mouse_activated:
                    if not mouse_activated:
                        mouse_activated = True
                        print("Mouse control activated")
                    mouse_control.mouse_controller(frame, landmarks)
                    continue
    
                # Mouse control deactivation
                if (len(gesture_cache) == cache_size and all(g == mouse_control_deactivation_signal for g in gesture_cache)) and mouse_activated:
                    if activation_timer.get_elapsed_time() > 1.5:
                        mouse_activated = False
                        mouse_cleared = True
                        gesture_cache.clear()
                        gesture_cache.extend([-1] * cache_size)
                        print("Mouse control deactivated")
                    continue
    
                # Brightness control
                if (len(gesture_cache) == cache_size and all(g == brightness_control_signal for g in gesture_cache)) or brightness_activated:
                    brightness_control.brightness_controller(frame, landmarks)
                    if brightness_activated:
                        if activation_timer.get_elapsed_time() > brightness_control_time:
                            brightness_activated = False
                            gesture_cache.clear()
                            gesture_cache.extend([-1] * cache_size)
                            print("Brightness control deactivated")
                    else:
                        activation_timer.reset()
                        brightness_activated = True
                        print("Brightness control activated")
                    continue
    
                # Volume control
                if (len(gesture_cache) == cache_size and all(g == volume_control_signal for g in gesture_cache)) or volume_activated:
                    volume_control.volume_controller(frame, landmarks)
                    if volume_activated:
                        if activation_timer.get_elapsed_time() > volume_control_time:
                            volume_activated = False
                            gesture_cache.clear()
                            gesture_cache.extend([-1] * cache_size)
                            print("Volume control deactivated")
                    else:
                        activation_timer.reset()
                        volume_activated = True
                        print("Volume control activated")
    
            cv2.imshow('Hand Landmark Detection', frame)
            
            if cv2.waitKey(1) == ord('q'):
                break
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
