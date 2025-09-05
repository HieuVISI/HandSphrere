import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from math import acos, degrees
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

st.set_page_config(page_title="Điều khiển khối cầu bằng tay")

st.title("Điều khiển khối cầu bằng tay (WebRTC)")

# Sidebar parameters
touch_dist_threshold = st.sidebar.slider("Ngưỡng chạm (pixel)", 10, 120, 40)
angle_threshold_deg = st.sidebar.slider("Ngưỡng góc phóng to (độ)", 10, 150, 75)
radius_step = st.sidebar.slider("Bước tăng/giảm bán kính", 1, 40, 8)
min_radius = st.sidebar.slider("Bán kính tối thiểu", 10, 200, 30)
max_radius = st.sidebar.slider("Bán kính tối đa", 20, 500, 180)

COLOR_MAP = {
    0: (0, 200, 0),    # xanh
    1: (0, 220, 220),  # vàng
    2: (0, 0, 255)     # đỏ
}
color_names = {0: "Xanh", 1: "Vàng", 2: "Đỏ"}


def calc_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def angle_between_vectors(a, b):
    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    cosang = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    cosang = np.clip(cosang, -1.0, 1.0)
    return degrees(acos(cosang))


class HandSphereProcessor(VideoProcessorBase):
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        )
        self.sphere_radius = 80
        self.color_state = 0
        self.last_touch_frame = 0
        self.frame_count = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape
        self.frame_count += 1

        # position of sphere
        cx, cy = w // 2, h // 2

        # Draw transparent sphere
        overlay = img.copy()
        cv2.circle(overlay, (cx, cy), self.sphere_radius, COLOR_MAP[self.color_state], -1)
        cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)

        # Hand detection
        results = self.hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = handedness.classification[0].label  # "Left" / "Right"

                lm = hand_landmarks.landmark
                def lm_xy(i):
                    return int(lm[i].x * w), int(lm[i].y * h)

                # Right hand → change color if touch sphere
                if label == "Right":
                    for idx in [4, 8, 12]:
                        px, py = lm_xy(idx)
                        if calc_distance((px, py), (cx, cy)) <= self.sphere_radius:
                            if self.frame_count - self.last_touch_frame > 15:
                                self.color_state = (self.color_state + 1) % 3
                                self.last_touch_frame = self.frame_count
                            break

                # Left hand → pinch shrink / spread grow
                if label == "Left":
                    thumb_tip, index_tip = lm_xy(4), lm_xy(8)
                    thumb_mcp, index_mcp = lm_xy(2), lm_xy(5)

                    # pinch
                    if calc_distance(thumb_tip, index_tip) <= touch_dist_threshold:
                        self.sphere_radius = max(min_radius, self.sphere_radius - radius_step)

                    # spread
                    vec_index = (index_tip[0] - index_mcp[0], index_tip[1] - index_mcp[1])
                    vec_thumb = (thumb_tip[0] - thumb_mcp[0], thumb_tip[1] - thumb_mcp[1])
                    ang = angle_between_vectors(vec_index, vec_thumb)
                    if ang >= angle_threshold_deg:
                        self.sphere_radius = min(max_radius, self.sphere_radius + radius_step)

                # Draw landmarks
                mp.solutions.drawing_utils.draw_landmarks(
                    img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )

        # Draw info
        cv2.circle(img, (cx, cy), self.sphere_radius, COLOR_MAP[self.color_state], 3)
        cv2.putText(img, f"Ban kinh: {self.sphere_radius}px", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(img, f"Mau: {color_names[self.color_state]}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        return img


webrtc_streamer(
    key="hand-sphere",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=HandSphereProcessor,
    media_stream_constraints={"video": True, "audio": False},
)
