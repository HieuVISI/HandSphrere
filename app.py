import streamlit as st
import cv2
import mediapipe as mp
import numpy as np

st.title("✋ Hand Tracking Demo (MediaPipe + Streamlit)")

# Khởi tạo mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands()

# Upload ảnh
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # Chuyển sang RGB để mediapipe xử lý
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    # Vẽ landmarks nếu có bàn tay
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Hiển thị ảnh
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Processed Image")
