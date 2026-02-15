import cv2
import mediapipe as mp
import numpy as np
import math

def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def is_pinch(points, threshold=35):
    return distance(points[4], points[8]) < threshold

def is_fist(points):
    fingers = [(8, 6), (12, 10), (16, 14), (20, 18)]
    folded = 0
    for tip, pip in fingers:
        if points[tip][1] > points[pip][1]:
            folded += 1
    return folded == 4

def is_open_hand(points):
    fingers = [(8, 6), (12, 10), (16, 14), (20, 18)]
    extended = 0
    for tip, pip in fingers:
        if points[tip][1] < points[pip][1]:
            extended += 1
    return extended == 4

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

ret, frame = cap.read()
if not ret:
    print("Camera not accessible")
    exit()

frame = cv2.flip(frame, 1)
h, w, _ = frame.shape

canvas = np.zeros((h, w, 3), dtype=np.uint8)

prev_point = None
mode = "IDLE"   

WINDOW_NAME = "Gesture Virtual Whiteboard"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(
    WINDOW_NAME,
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN
)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:

            points = []
            for lm in hand_landmarks.landmark:
                cx = int(lm.x * w)
                cy = int(lm.y * h)
                points.append((cx, cy))

            if is_fist(points):
                mode = "ERASE"

            elif is_pinch(points):
                mode = "DRAW"

            elif is_open_hand(points):
                mode = "IDLE"

            index_tip = points[8]

            if mode == "DRAW":
                if prev_point is not None:
                    cv2.line(canvas, prev_point, index_tip, (255, 255, 255), 5)
                prev_point = index_tip

            elif mode == "ERASE":
                cv2.circle(canvas, index_tip, 40, (0, 0, 0), -1)
                prev_point = None

            else:
                prev_point = None

    frame = cv2.addWeighted(frame, 0.4, canvas, 0.6, 0)

    cv2.putText(frame, f"MODE: {mode}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imshow(WINDOW_NAME, frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
