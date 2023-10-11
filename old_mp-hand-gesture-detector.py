# cv2 for openCV image processing, display, and labeling
import cv2
import mediapipe as mp
import math
# MediaPipe for hand/face tracking
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
# For gesture-triggered hotkey functionality
import pyautogui
import time

# Initialize MediaPipe Hands and Drawing utilities
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Initialize GestureRecognizer
base_options = python.BaseOptions(model_asset_path='gesture_recognizer.task')
options = vision.GestureRecognizerOptions(base_options=base_options)
recognizer = vision.GestureRecognizer.create_from_options(options)

# Function to resize and show image (for debugging)
def resize_and_show(image):
    h, w = image.shape[:2]
    DESIRED_HEIGHT = 480
    DESIRED_WIDTH = 480
    if h < w:
        img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h / (w / DESIRED_WIDTH))))
    else:
        img = cv2.resize(image, (math.floor(w / (h / DESIRED_HEIGHT)), DESIRED_HEIGHT))
    cv2.imshow('Debug', img)

def fingers_on_face(image, hand_landmarks, x_min, x_max, y_min, y_max, bounds_multiplier = 1):
    # get the x, y coordinates of the index and middle fingers
    index_finger = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_finger = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_finger = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]

    # convert the coordinates to pixel values
    index_x, index_y = int(index_finger.x * image.shape[1]), int(index_finger.y * image.shape[0])
    middle_x, middle_y = int(middle_finger.x * image.shape[1]), int(middle_finger.y * image.shape[0])

    # get the bounds of the face
    mid_x = image.shape[1] // 2 # convert to pixel values
    mid_y = image.shape[0] // 2 # convert to pixel values
    x_min = mid_x - (mid_x - x_min) * bounds_multiplier
    x_max = mid_x + (x_max - mid_x) * bounds_multiplier
    y_min = mid_x - (mid_x - y_min) * bounds_multiplier
    y_max = mid_x + (y_max - mid_x) * bounds_multiplier

    # if ring finger is on face, return -1
    if x_min <= ring_finger.x <= x_max and y_min <= ring_finger.y <= y_max:
        return -1

    # count fingers on face
    fingers_on_face = 0
    if x_min <= index_x <= x_max and y_min <= index_y <= y_max:
        cv2.circle(image, (index_x, index_y), 10, (0, 0, 255), -1)
        cv2.putText(image, f"x_min { x_min } x_max { x_max } \ny_min { y_min } y_max { y_max }", (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        fingers_on_face += 1

    if x_min <= middle_x <= x_max and y_min <= middle_y <= y_max:
        cv2.circle(image, (middle_x, middle_y), 10, (0, 0, 255), -1)
        cv2.putText(image, 'Middle in face', (middle_x + 10, middle_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        fingers_on_face += 1

    return fingers_on_face

def draw_hands_landmarks(image_rgb, results):
    # Convert the image back to BGR and write
    image_rgb.flags.writeable = True
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    hand_landmarks = None

    # Draw landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image_bgr, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    return image_bgr, hand_landmarks

def gesture_recognizer(image):
    # Save the OpenCV image to a temporary file for Recognizer to read
    cv2.imwrite("temp_image.jpg", image_bgr)
    # Load the image using MediaPipe's create_from method
    mp_image = mp.Image.create_from_file("temp_image.jpg")
    # Perform gesture recognition
    recognition_result = recognizer.recognize(mp_image)
    if recognition_result.gestures:
        # mp_image.append(mp_image) # Draw gesture label
        top_gesture = recognition_result.gestures[0][0]
        # Sample attributes from top_gesture object
        display_name = top_gesture.display_name
        score = top_gesture.score
        # ... any other attribute
        # Combine them into a list of strings
        lines = [
            f"Display Name: {display_name}",
            f"Score: {score}",
            # ... any other lines
        ]
        cv2.putText(image_bgr, f"Gesture: {lines}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

# Check for "Okay" gesture
def check_ok_gesture(image, hand_landmarks):

    # Function to get orientation of the hand. NOTE: Unused
    def cog_get_orientation(coordinate_landmark_0, coordinate_landmark_9):
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
    
    # get coordinates of fingertips
    index = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    thumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    other_fingers_array = [hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP], hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP], hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]]
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    pinky_knuckle = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]
    
    # A reference distance for scaling distances between fingers
    max_extent = math.sqrt(math.pow((pinky_knuckle.x - wrist.x), 2) + math.pow((pinky_knuckle.y - wrist.y), 2))
    # The min distance between index and thumb (0 to 1)
    thumb_index_dist = math.sqrt(math.pow((index.x - thumb.x), 2) + math.pow((index.y - thumb.y), 2))

    # Init misc vars for gesture recognition
    max_dist_to_others = 1      # Max distance is 1
    lowest_other_finger = None  # The other finger that is lowest on the screen
    distances_to_others = []    # List of distances from the index finger to all other fingers

    # Is index & thumb close together?
    if thumb_index_dist < 0.05:
        # A. Compute the conditional values for determining which gesture, if any, is being made
        # i) Find the lowest finger that is not the index or thumb
        for other_finger in other_fingers_array:
            # Get the lowest finger vertically
            lowest_other_finger = min(
                other_fingers_array, 
                key=lambda other_finger: other_finger.y
            )
            # Array of distances from the index finger to "other fingers"
            distances_to_others.append(math.sqrt(math.pow((index.x - other_finger.x), 2) + math.pow((index.y - other_finger.y), 2)))

        max_dist_to_others = max(distances_to_others)
        normalized_max_dist_to_others = max_dist_to_others / max_extent


        # B. Determine which gesture, if any, is being made
        # i) Is an OK gesture? NOTE: Y is inverted in OpenCV (0 at top, 1 at bottom)
        if index.y > lowest_other_finger.y and normalized_max_dist_to_others > 0.3:
            cv2.putText(image, "Okay!!", (500, 200), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 2)
            return "OK"
        # ii) Is a PINCH gesture?
        elif normalized_max_dist_to_others < 0.5:
            print("normalized_max_dist_to_others: " + str(normalized_max_dist_to_others))
            cv2.putText(image, "Pinch!!", (500, 200), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 2)
            #print(f"Max extent: {max_extent}, thumb-index dist: {thumb_index_dist}, normalized thumb-index dist: {thumb_index_dist / max_extent}")
            return "PINCH"
    
    return "NONE"

def draw_face_landmarks_box(image_rgb, results):
    # Draw face bounding box
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            x_min = y_min = float('inf')
            x_max = y_max = float('-inf')
            for id, lm in enumerate(face_landmarks.landmark):
                x, y = int(lm.x * image_rgb.shape[1]), int(lm.y * image_rgb.shape[0])
                x_min, y_min = min(x, x_min), min(y, y_min)
                x_max, y_max = max(x, x_max), max(y, y_max)
            cv2.rectangle(image_rgb, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    
        return x_min, x_max, y_min, y_max
    
    return None, None, None, None

debounce = 0.5
debounce_timer = 0
last_time = time.time()
gesturing = False

def toggle_keystroke_on_gesture(gesture):
    global gesturing, debounce_timer
    #print(str(gesture) + " " + str(gesturing))
    if gesture == "Victory" and not gesturing:
        pyautogui.PAUSE = 0.25
        pyautogui.hotkey('alt', 'm')
        gesturing = True
        print("Gesture action: Start")

    elif gesture is None:
        gesturing = False

def check_gesture_ended():
    global debounce_timer, gesturing
    if not gesturing: 
        return
    if debounce_timer < 0:
        pyautogui.PAUSE = 0.25
        pyautogui.hotkey('alt', 'm')
        gesturing = False
        print("Gesture action: End")

# Main function
def main():
    global last_time, gesturing, debounce_timer, debounce
    print("Starting capture...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    print("Capture started")
    with mp_hands.Hands(min_detection_confidence=0.3, min_tracking_confidence=0.3) as hands, mp_face.FaceMesh() as face_mesh:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # First, convert the image
            image_rgb = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results_hands = hands.process(image_rgb)
            results_face = face_mesh.process(image_rgb)

            # Results 1. Draw face box
            x_min, x_max, y_min, y_max = draw_face_landmarks_box(image_rgb, results_face)

            # Results 2. Draw hands landmarks
            image_bgr, hand_landmarks = draw_hands_landmarks(image_rgb, results_hands)

            # Results 3. Hand gesture recognition
            # gesture_recognizer(image_bgr)
            
            # Results 4. Handle gestures
            if x_min is not None and hand_landmarks is not None:
                gesture_status = check_ok_gesture(image_bgr, hand_landmarks) # num_fingers_on_face = fingers_on_face(image_bgr, hand_landmarks, x_min, x_max, y_min, y_max)

                # Is gesturing OK
                if gesture_status == "OK": # num_fingers_on_face == 2:
                    toggle_keystroke_on_gesture("Victory")
                    debounce_timer = debounce

                # Is gesturing PINCH
                elif gesture_status == "PINCH":
                    check_gesture_ended()

            # Show the image
            cv2.imshow('MediaPipe Hands', image_bgr)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

            debounce_timer -= time.time() - last_time
            last_time = time.time()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
