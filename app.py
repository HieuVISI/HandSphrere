# file: app.py
import streamlit as st
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector

st.title("🎮 Hand Control Sphere Demo (cvzone)")

# Sidebar tùy chọn
st.sidebar.markdown("### ⚙️ Settings")
run = st.sidebar.checkbox("Run camera", value=False)
radius_min = st.sidebar.slider("Min radius", 30, 100, 50)
radius_max = st.sidebar.slider("Max radius", 150, 300, 200)

# Detector tay
detector = HandDetector(detectionCon=0.7, maxHands=2)

# Hàm tính khoảng cách giữa 2 điểm (x, y)
def distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

radius = 100
color = (0, 255, 0)

FRAME_WINDOW = st.image([])

cap = None
if run:
    cap = cv2.VideoCapture(0)  # mở webcam
else:
    st.info("Click 'Run camera' để bật webcam")

while run and cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Detect tay
    hands, frame = detector.findHands(frame)

    if hands:
        for hand in hands:
            lmList = hand["lmList"]  # 21 keypoints
            handType = hand["type"]  # Left / Right

            thumb = lmList[4][:2]   # (x, y)
            index = lmList[8][:2]   # (x, y)

            if handType == "Left":
                d = distance(thumb, index)
                radius = int(np.clip(d, radius_min, radius_max))
            elif handType == "Right":
                ix, iy = index
                if (ix - w//2)**2 + (iy - h//2)**2 <= radius**2:
                    color = tuple(np.random.randint(0, 255, 3).tolist())

    # Vẽ hình tròn ở giữa frame
    cv2.circle(frame, (w//2, h//2), radius, color, -1)

    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
