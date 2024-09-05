"""
Gesture-Controlled PC Interaction System

This script uses Mediapipe and TensorFlow to control PC functions (volume, brightness, and mouse) through hand gestures.
It captures video from the webcam, processes hand landmarks to identify gestures, and then triggers corresponding control actions.

Dependencies:
    - OpenCV (cv2)
    - Mediapipe (mp)
    - TensorFlow (tf)
    - NumPy (np)
    - Custom modules for volume, brightness, and mouse control

Adjustable Constants:
    - wait_time
    - resolution_x
    - resolution_y
    - mouse_control_activation_signal
    - mouse_control_deactivation_signal
    - brightness_control_signal
    - volume_control_signal
    - brightness_control_time
    - volume_control_time
"""

import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np
import timer
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

# Time in seconds a gesture must be held to trigger a control action change
wait_time = 1.5

# Webcam resolution dimensions
resolution_x = 1280
resolution_y = 720

# Gesture signals for activating/deactivating functionality
mouse_control_activation_signal = INDEX
mouse_control_deactivation_signal = ZERO
brightness_control_signal = PEACE
volume_control_signal = THREE

# Time in seconds given for activating/deactivating brightness and volume control
brightness_control_time = 3
volume_control_time = 3

##########################
## Function definition(s)
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

##############
## Main Loop
##############

# Initialize Mediapipe Hands and model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
model = tf.keras.models.load_model("model/hand_landmarks_model.h5")
gesture_cache = []
cache_size = int(wait_time * 10)

hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0) # Start video capture from webcam
activation_timer = timer.Timer()
activation_timer.start()
brightness_activated = False
volume_activated = False
mouse_activated = False
mouse_cleared = True

if not cap.isOpened():
    print('Error: failed to access local camera device')
    exit()

while True:
    ret, frame = cap.read()
    
    if not ret:
        print('Failed to grab frame')
        break
    
    frame = cv2.flip(frame, 1) # Flip the frame for a mirror view
    color_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convert frame to RGB for Mediapipe processing
    results = hands.process(color_frame)
    
    if results.multi_hand_landmarks:
        landmarks = results.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
        input = tf.expand_dims(tf.convert_to_tensor(preprocess(landmarks, None)), axis=0)
        prediction = model.predict(input)

        # Update gesture cache with latest prediction
        if np.max(prediction) >= 0.9:
            gesture = np.argmax(prediction, axis=1)[0]
            gesture_cache.append(gesture)
            if len(gesture_cache) > cache_size:
                gesture_cache = gesture_cache[-cache_size:]
        
        # Mouse control activation
        if (len(gesture_cache) == cache_size and all(map(lambda x: x == mouse_control_activation_signal, gesture_cache))) or mouse_activated:
            if not mouse_activated:
                mouse_activated = True
                print("Mouse control activated")
            mouse_control.mouse_controller(frame, landmarks)  # Call mouse controller function
            continue
        
        # Mouse control deactivation
        if (len(gesture_cache) == cache_size and all(map(lambda x: x == mouse_control_deactivation_signal, gesture_cache))) and mouse_activated:
            if activation_timer.get_elapsed_time() > 1.5:
                mouse_activated = False
                mouse_cleared = True
                gesture_cache[:] = [-1] * cache_size
                print("Mouse control deactivated")
            continue
        
        # Brightness control
        if (len(gesture_cache) == cache_size and all(map(lambda x: x == brightness_control_signal, gesture_cache))) or brightness_activated:
            brightness_control.brightness_controller(frame, landmarks) # Activate brightness control
            if brightness_activated: # Continue/discontinue activation
                if activation_timer.get_elapsed_time() > brightness_control_time:
                    brightness_activated = False
                    gesture_cache[:] = [-1] * cache_size
                    print("Brightness control deactivated")
            else: # Trigger activation
                activation_timer.reset()
                brightness_activated = True
                print("Brightness control activated")
            continue
        
        # Volume control
        if (len(gesture_cache) == cache_size and all(map(lambda x: x == volume_control_signal, gesture_cache))) or volume_activated:
            volume_control.volume_controller(frame, landmarks) # Activate volume control
            if volume_activated: # Continue/discontinue activation
                if activation_timer.get_elapsed_time() > volume_control_time:
                    volume_activated = False
                    gesture_cache[:] = [-1] * cache_size
                    print("Volume control deactivated")
            else: # Trigger activation
                activation_timer.reset()
                volume_activated = True
                print("Volume control activated")
    
    cv2.imshow('Hand Landmark Detection', frame)
    
    # Exit loop if 'q' key is pressed
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
