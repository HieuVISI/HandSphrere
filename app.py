import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

st.title("🎮 Hand Control Sphere Demo (Webcam qua Browser)")

radius = 100
color = (0, 255, 0)

class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.radius = 100
        self.color = (0, 255, 0)

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape

        # Vẽ hình cầu giữa màn hình
        cv2.circle(img, (w//2, h//2), self.radius, self.color, -1)

        return img

webrtc_streamer(
    key="hand-demo",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    rtc_configuration={
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},  # Google STUN server
        ]
    },
)
