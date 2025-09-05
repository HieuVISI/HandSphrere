# file: app.py
import streamlit as st
import cv2
import numpy as np

st.title("🎮 Hand Control Sphere Demo (No Mediapipe)")

st.sidebar.markdown("### ⚙️ Settings")
uploaded_file = st.sidebar.file_uploader("Upload a video", type=["mp4", "mov", "avi"])
radius_min = st.sidebar.slider("Min radius", 30, 100, 50)
radius_max = st.sidebar.slider("Max radius", 150, 300, 200)

radius = 100
color = (0, 255, 0)

FRAME_WINDOW = st.image([])

if uploaded_file is not None:
    # Lưu tạm file
    with open("temp.mp4", "wb") as f:
        f.write(uploaded_file.read())
    cap = cv2.VideoCapture("temp.mp4")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Lọc màu da
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 30, 60], dtype=np.uint8)
        upper = np.array([20, 150, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            c = max(contours, key=cv2.contourArea)
            (x, y), rad = cv2.minEnclosingCircle(c)

            est_radius = int(np.clip(rad, radius_min, radius_max))
            radius = est_radius

            cx, cy = int(x), int(y)
            if (cx - w//2)**2 + (cy - h//2)**2 <= radius**2:
                color = tuple(np.random.randint(0, 255, 3).tolist())

            cv2.drawContours(frame, [c], -1, (255, 0, 0), 2)

        cv2.circle(frame, (w//2, h//2), radius, color, -1)

        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
else:
    st.info("📤 Upload một video để chạy demo (cloud không hỗ trợ webcam).")
