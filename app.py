import av
import cv2
import streamlit as st

from streamlit_webrtc import (
    webrtc_streamer,
    VideoProcessorBase,
    RTCConfiguration
)

from detector import detect
from analyzer import analyze_face


st.set_page_config(
    page_title="Face Analysis",
    page_icon="😀",
    layout="wide"
)

st.title("😀 Real-Time Face Analysis")
st.markdown("---")


rtc_configuration = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]}
        ]
    }
)


class VideoProcessor(VideoProcessorBase):

    def __init__(self):

        self.frame_count = 0

        self.cache = []


    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        detections = detect(img)

        self.frame_count += 1

        # Analyze every 5th frame
        if self.frame_count % 5 == 0:

            self.cache = []

            for det in detections:

                face = det["face"]

                if face.size == 0:
                    continue

                age, gender, emotion = analyze_face(face)

                self.cache.append({

                    "box": det["box"],

                    "age": age,

                    "gender": gender,

                    "emotion": emotion

                })

        # Draw cached results
        for info in self.cache:

            x1, y1, x2, y2 = info["box"]

            cv2.rectangle(
                img,
                (x1, y1),
                (x2, y2),
                (0,255,0),
                2
            )

            if info["age"] is not None:

                cv2.putText(
                    img,
                    f"Age : {int(info['age'])}",
                    (x1, y1-60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0,255,0),
                    2
                )

            if info["gender"] is not None:

                cv2.putText(
                    img,
                    f"Gender : {info['gender']}",
                    (x1, y1-40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0,255,0),
                    2
                )

            if info["emotion"] is not None:

                cv2.putText(
                    img,
                    f"Emotion : {info['emotion']}",
                    (x1, y1-20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0,255,0),
                    2
                )

        cv2.putText(
            img,
            f"Faces : {len(detections)}",
            (20,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2
        )

        return av.VideoFrame.from_ndarray(
            img,
            format="bgr24"
        )


webrtc_streamer(

    key="face-analysis",

    rtc_configuration=rtc_configuration,

    video_processor_factory=VideoProcessor,

    media_stream_constraints={
        "video": True,
        "audio": False
    },

    async_processing=True

)