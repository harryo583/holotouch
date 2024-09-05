"""
This script captures video frames from a webcam, detects hand landmarks using Mediapipe, preprocesses the landmark data, 
and saves it to a CSV file. It uses OpenCV for video capture and display, and Mediapipe for hand landmark detection.

Dependencies:
    - OpenCV (cv2)
    - Mediapipe (mp)
    - CSV (csv)
    - Time (time)
"""

import cv2
import mediapipe as mp
import csv
import time


def preprocess(landmarks, label):
    """
    Preprocesses the hand landmarks to normalize and prepare input for the model.
    Returns a list of normalized landmark coordinates, with the label appended if provided.
    """
    row = []
    base_x, base_y = landmarks.landmark[0].x, landmarks.landmark[0].y

    # Extract and normalize landmark coordinates
    for landmark in landmarks.landmark:
        row.extend([landmark.x - base_x, landmark.y - base_y])

    # Normalize the data
    max_value = max(abs(min(row)), max(row))
    row = list(map(lambda x: x / max_value, row))

    if label is not None:
        row.append(label)

    return row


def main():
    """
    Captures video frames, detects hand landmarks, preprocesses the data, and saves it to a CSV file.
    """
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    # Initialize Mediapipe Hands module
    hands = mp_hands.Hands(static_image_mode=False,
                           max_num_hands=1,
                           min_detection_confidence=0.7,
                           min_tracking_confidence=0.5)

    # Initialize video capture
    cap = cv2.VideoCapture(0)
    data = "hand_landmarks.csv"
    header = []

    # Create CSV header
    for i in range(21):
        header.extend([f'landmark_{i}_x', f'landmark_{i}_y'])

    header.append('class')

    # Uncomment this block if you want to write the header to the CSV file
    # with open(data, mode='w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(header)

    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame to detect hand landmarks
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks on the frame
                mp_drawing.draw_landmarks(
                    frame, landmarks, mp_hands.HAND_CONNECTIONS)
                # Preprocess the landmarks and save to CSV
                row = preprocess(landmarks, 5)
                with open(data, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(row)
                print(row)

        # Display the frame
        cv2.imshow('Hand Landmarks', frame)

        # Exit loop if 'q' is pressed
        if cv2.waitKey(1) == ord('q'):
            break

        # Control frame rate
        elapsed_time = time.time() - start_time
        wait_time = 200 - int(elapsed_time * 1000)

        if wait_time > 0:
            cv2.waitKey(wait_time)

    # Release video capture and close windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
