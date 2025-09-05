# file: app.py
import streamlit as st
import cv2
print("OpenCV version:", cv2.__version__)
import mediapipe as mp
import numpy as np

st.title("🎮 Hand Control Sphere Demo")

# Sidebar tùy chọn
st.sidebar.markdown("### ⚙️ Settings")
run = st.sidebar.checkbox("Run camera", value=False)
radius_min = st.sidebar.slider("Min radius", 30, 100, 50)
radius_max = st.sidebar.slider("Max radius", 150, 300, 200)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Hàm tính khoảng cách
def distance(a, b):
    return np.linalg.norm(np.array([a.x - b.x, a.y - b.y]))

radius = 100
color = (0, 255, 0)

FRAME_WINDOW = st.image([])

cap = None
if run:
    cap = cv2.VideoCapture(0)  # webcam
else:
    st.info("Click 'Run camera' để bật webcam")

while run and cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Mediapipe xử lý
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = handedness.classification[0].label  # Left / Right
            thumb = hand_landmarks.landmark[4]
            index = hand_landmarks.landmark[8]

            if label == "Left":
                d = distance(thumb, index)
                radius = int(np.clip(d * 500, radius_min, radius_max))
            elif label == "Right":
                ix, iy = int(index.x * w), int(index.y * h)
                if (ix - w//2)**2 + (iy - h//2)**2 <= radius**2:
                    color = tuple(np.random.randint(0, 255, 3).tolist())

    # Vẽ hình tròn ở giữa frame
    cv2.circle(frame, (w//2, h//2), radius, color, -1)

    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
