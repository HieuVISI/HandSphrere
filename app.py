import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time
from math import acos, degrees

st.set_page_config(page_title="Điều khiển khối cầu bằng tay", layout="wide")

st.title("Điều khiển khối cầu bằng tay — Streamlit + Webcam + MediaPipe")
st.markdown("""
- Tay phải: chạm vào quả cầu để **đổi màu** theo chu kỳ **xanh → vàng → đỏ**.  
- Tay trái: **ngón trỏ + ngón cái chạm** → **thu nhỏ**; **góc > 75°** giữa 2 ngón → **phóng to**.
""")

# Sidebar controls
st.sidebar.header("Cài đặt")
min_detection_conf = st.sidebar.slider("Min detection confidence", 0.3, 0.9, 0.7)
min_tracking_conf = st.sidebar.slider("Min tracking confidence", 0.3, 0.9, 0.7)
touch_dist_threshold = st.sidebar.slider("Ngưỡng chạm (pixel, tương đối)", 10, 120, 40)
angle_threshold_deg = st.sidebar.slider("Ngưỡng góc phóng to (độ)", 10, 150, 75)
radius_step = st.sidebar.slider("Bước tăng/giảm bán kính (pixel)", 1, 80, 8)
min_radius = st.sidebar.slider("Bán kính tối thiểu (pixel)", 10, 200, 30)
max_radius = st.sidebar.slider("Bán kính tối đa (pixel)", 20, 500, 180)

# Control buttons
if "running" not in st.session_state:
    st.session_state.running = False

col1, col2 = st.columns([1, 1])
with col1:
    if not st.session_state.running:
        if st.button("Bắt đầu (Start)"):
            st.session_state.running = True
    else:
        if st.button("Dừng (Stop)"):
            st.session_state.running = False

with col2:
    st.write("Trạng thái:", "🟢 Đang chạy" if st.session_state.running else "🔴 Dừng")

# Placeholder for video
frame_placeholder = st.empty()

# initial sphere state
if "sphere_radius" not in st.session_state:
    st.session_state.sphere_radius = 80
if "color_state" not in st.session_state:
    # 0: xanh, 1: vàng, 2: đỏ
    st.session_state.color_state = 0
if "last_touch_time" not in st.session_state:
    st.session_state.last_touch_time = 0.0
# cooldown to avoid multiple rapid color changes
COLOR_CHANGE_COOLDOWN = 0.45  # seconds

# color mapping BGR for OpenCV
COLOR_MAP = {
    0: (0, 200, 0),    # xanh (green)
    1: (0, 220, 220),  # vàng (yellow-like in BGR)
    2: (0, 0, 255)     # đỏ (red)
}

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def calc_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def angle_between_vectors(a, b):
    # a, b: 2D vectors
    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    cosang = np.dot(a, b) / (norm_a * norm_b)
    cosang = np.clip(cosang, -1.0, 1.0)
    return degrees(acos(cosang))

def process_frame(frame, hands_detector):
    h, w, _ = frame.shape
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(image_rgb)

    # sphere position (center)
    cx, cy = w // 2, h // 2
    radius = int(st.session_state.sphere_radius)
    color = COLOR_MAP[st.session_state.color_state]

    # draw semi-transparent background circle (the "sphere")
    overlay = frame.copy()
    cv2.circle(overlay, (cx, cy), radius, color, -1)
    alpha = 0.35
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = handedness.classification[0].label  # "Left" or "Right"
            # convert landmark to pixel coords
            lm = hand_landmarks.landmark
            # fingertip indices: thumb(4), index(8), middle(12), ring(16), pinky(20)
            def lm_xy(i):
                return int(lm[i].x * w), int(lm[i].y * h)

            # Right hand: check touch to change color
            if label == "Right":
                # We'll check several fingertip points for touch: index(8), middle(12), thumb(4)
                fingertip_indices = [4, 8, 12]
                touched = False
                for idx in fingertip_indices:
                    px, py = lm_xy(idx)
                    d = calc_distance((px, py), (cx, cy))
                    if d <= radius:
                        touched = True
                        break
                if touched:
                    now = time.time()
                    if now - st.session_state.last_touch_time > COLOR_CHANGE_COOLDOWN:
                        # cycle color
                        st.session_state.color_state = (st.session_state.color_state + 1) % 3
                        st.session_state.last_touch_time = now

            # Left hand: handle pinch (shrink) and spread-angle (grow)
            if label == "Left":
                # coordinates
                thumb_tip = lm_xy(4)
                index_tip = lm_xy(8)
                thumb_mcp = lm_xy(2)   # thumb MCP-ish
                index_mcp = lm_xy(5)   # index MCP

                # 1) pinch (touch) => shrink
                dist_thumb_index = calc_distance(thumb_tip, index_tip)
                if dist_thumb_index <= touch_dist_threshold:
                    # shrink
                    st.session_state.sphere_radius = max(min_radius, st.session_state.sphere_radius - radius_step)

                # 2) angle between index vector and thumb vector > threshold => enlarge
                vec_index = (index_tip[0] - index_mcp[0], index_tip[1] - index_mcp[1])
                vec_thumb = (thumb_tip[0] - thumb_mcp[0], thumb_tip[1] - thumb_mcp[1])
                ang = angle_between_vectors(vec_index, vec_thumb)
                if ang >= angle_threshold_deg:
                    st.session_state.sphere_radius = min(max_radius, st.session_state.sphere_radius + radius_step)

            # draw landmarks for user feedback
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # draw the circle border and text info
    cv2.circle(frame, (cx, cy), radius, color, 3)
    # label with size and color
    color_names = {0: "XANH", 1: "VANG", 2: "DO"}
    cv2.putText(frame, f"Radius: {st.session_state.sphere_radius}px", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Mau: {color_names[st.session_state.color_state]}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)

    return frame

# Main loop (runs while running True)
if st.session_state.running:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Không bật được webcam. Kiểm tra camera hoặc quyền truy cập.")
        st.session_state.running = False
    else:
        with mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=float(min_detection_conf),
            min_tracking_confidence=float(min_tracking_conf)
        ) as hands:
            try:
                while st.session_state.running:
                    ret, frame = cap.read()
                    if not ret:
                        st.warning("Không đọc được frame từ webcam.")
                        break
                    frame = cv2.flip(frame, 1)  # mirror
                    processed = process_frame(frame, hands)
                    # convert BGR->RGB for streamlit display
                    processed_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(processed_rgb, channels="RGB")
                    # small sleep to reduce CPU (adjust if necessary)
                    time.sleep(0.02)
                    # Streamlit may re-run; check session_state
                    if not st.session_state.running:
                        break
            except Exception as e:
                st.error(f"Lỗi khi xử lý webcam: {e}")
            finally:
                cap.release()
                frame_placeholder.empty()
else:
    st.info("Nhấn **Bắt đầu (Start)** để bật webcam và khởi chạy điều khiển.")
