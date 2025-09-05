import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase
import cv2
from cvzone.HandTrackingModule import HandDetector

st.title("Điều khiển khối cầu bằng tay (cvzone + streamlit-webrtc)")

class HandSphere(VideoProcessorBase):
    def __init__(self):
        self.detector = HandDetector(detectionCon=0.7, maxHands=2)
        self.radius = 80
        self.color_state = 0
        self.colors = [(0,200,0),(0,220,220),(0,0,255)]

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        hands, img = self.detector.findHands(img)  # phát hiện tay

        h, w, _ = img.shape
        cx, cy = w//2, h//2

        # Vẽ quả cầu
        cv2.circle(img,(cx,cy),self.radius,self.colors[self.color_state],-1)
        cv2.circle(img,(cx,cy),self.radius,(255,255,255),2)

        if hands:
            for hand in hands:
                if hand["type"] == "Right":
                    # nếu ngón trỏ chạm cầu thì đổi màu
                    x,y = hand["lmList"][8][0:2]
                    if (x-cx)**2+(y-cy)**2 <= self.radius**2:
                        self.color_state = (self.color_state+1)%3

                if hand["type"] == "Left":
                    # khoảng cách giữa ngón cái và trỏ
                    dist,_,img = self.detector.findDistance(hand["lmList"][4][0:2], hand["lmList"][8][0:2], img)
                    if dist < 40:
                        self.radius = max(20, self.radius-5)
                    elif dist > 150:
                        self.radius = min(200, self.radius+5)

        return img

webrtc_streamer(
    key="demo",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=HandSphere,
    media_stream_constraints={"video": True, "audio": False},
)
