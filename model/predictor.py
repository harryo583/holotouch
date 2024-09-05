"""
This script performs real-time gesture recognition using a trained TensorFlow (Keras) model and webcam input.
It uses Mediapipe to detect hand landmarks and OpenCV to capture video frames and display results.

Steps:
    1. Load the trained TensorFlow model for gesture recognition.
    2. Initialize Mediapipe Hand module for hand landmark detection.
    3. Capture video from the webcam.
    4. Process each frame to detect hand landmarks and predict the gesture using the model.
    5. Display the predicted gesture on the video frame.

Dependencies:
    - OpenCV (cv2)
    - NumPy (numpy)
    - Mediapipe (mp)
    - TensorFlow (tf)
    - Custom Data (data)
"""

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import data

# Load the trained model from a specified path
model = tf.keras.models.load_model("hand_landmarks_model.h5")

# Initialize Mediapipe Hands module for hand landmark detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1)

# Start webcam capture
cap = cv2.VideoCapture(0)

while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # Clear the frame text
    frame_text = ""
    if results.multi_hand_landmarks:
        # Get the landmarks for the first detected hand
        landmarks = results.multi_hand_landmarks[0]

        # Draw the hand landmarks on the frame
        mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

        # Preprocess the landmarks and convert to tensor for model prediction
        input = tf.expand_dims(tf.convert_to_tensor(
            data.preprocess(landmarks, None)), axis=0)
        prediction = model.predict(input)

        # Check if the model's confidence is high enough and gets the most probable class
        if np.max(prediction) >= 0.9:
            predicted_class = np.argmax(prediction, axis=1)[0]
            frame_text = f'{predicted_class}'

    # Display the predicted class on the frame
    if frame_text:
        cv2.putText(frame, frame_text, (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3, cv2.LINE_AA)

    # Show the frame with the gesture recognition result
    cv2.imshow('Gesture Recognition', frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# # Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
