# Importing required libraries
import mediapipe as mp
import cv2
import numpy as np
from math import dist

# Initialize MediaPipe Hands and Drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize variables
points = []

# Function to get orientation of the hand
def orientation(coordinate_landmark_0, coordinate_landmark_9):
    x0, y0 = coordinate_landmark_0
    x9, y9 = coordinate_landmark_9
    if abs(x9 - x0) < 0.05:
        m = 1000000000
    else:
        m = abs((y9 - y0) / (x9 - x0))
    if 0 <= m <= 1:
        return "Right" if x9 > x0 else "Left"
    if m > 1:
        return "Up" if y9 < y0 else "Down"

# Function to get x, y coordinates of a landmark
def get_coordinates(landmark):
    x = landmark.x
    y = landmark.y
    return x, y

# Main function
def main():
    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # Flip and convert the image
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = hands.process(image)

            # Draw landmarks
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Get orientation
                    lm0 = get_coordinates(hand_landmarks.landmark[0])
                    lm9 = get_coordinates(hand_landmarks.landmark[9])
                    orient = orientation(lm0, lm9)

                    # Check for "Okay" gesture
                    lm4 = get_coordinates(hand_landmarks.landmark[4])
                    lm5 = get_coordinates(hand_landmarks.landmark[5])
                    if orient == "Right" and lm4[0] < lm5[0]:
                        cv2.putText(image, "Okay!!", (500, 200), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 2)

                    # Drawing with index finger
                    lm8 = get_coordinates(hand_landmarks.landmark[8])
                    points.append((int(lm8[0] * image.shape[1]), int(lm8[1] * image.shape[0])))
                    for i in range(len(points) - 1):
                        cv2.line(image, points[i], points[i + 1], (255, 255, 0), 1)

            # Show the image
            cv2.imshow('MediaPipe Hands', image)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
