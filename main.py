import cv2
import sys
import matplotlib.pyplot as plt
import mediapipe as mp
from core import process_and_show_captured_hand_palm

# Initialize Mediapipe for hand detection
hands_model = mp.solutions.hands.Hands(
    static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7
)

s = 0
if len(sys.argv) > 1:
    s = sys.argv[1]

source = cv2.VideoCapture(s)

win_name = "Camera Preview"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

captured_hand_frame = None

while cv2.waitKey(1) != 27:  # Escape
    has_frame, frame = source.read()

    if not has_frame:
        break

    # Flip the frame horizontally for a natural selfie-view display.
    frame = cv2.flip(frame, 1)
    raw_frame = frame.copy()

    # Detect the hand using Mediapipe
    results = hands_model.process(frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            # mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            h, w, _ = frame.shape

            x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * w)
            y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * h)
            x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * w)
            y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * h)

            # Draw a rectangle around the palm area
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
            captured_hand_frame = raw_frame[y_min:y_max, x_min:x_max]

    cv2.imshow(win_name, frame)

source.release()
cv2.destroyWindow(win_name)

process_and_show_captured_hand_palm(captured_hand_frame)
