import streamlit as st
import cv2
from deepface import DeepFace
import numpy as np
import pandas as pd

# Page configuration
st.set_page_config(page_title="Emotion Detector", page_icon="😊", layout="centered")

# Styling
st.markdown(
    """
    <style>
    .title {
        font-size: 32px;
        font-weight: 700;
        color: #3B82F6;
        text-align: center;
    }
    .emotion {
        font-size: 24px;
        font-weight: 600;
        color: #22C55E;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title
st.markdown('<div class="title">🧠 Real-Time Emotion Detection</div>', unsafe_allow_html=True)
st.markdown("Detect your emotions in real-time using your webcam and DeepFace 💡")

# Streamlit components
FRAME_WINDOW = st.image([])
emotion_text = st.empty()
chart_area = st.empty()
start_btn = st.button("▶️ Start Webcam")
stop_btn = st.empty()

# Emoji mapping
emotion_emoji = {
    "happy": "😊",
    "sad": "😢",
    "angry": "😠",
    "surprise": "😲",
    "fear": "😨",
    "disgust": "🤢",
    "neutral": "😐"
}

# Emotion detection function
def detect_emotion(frame):
    detected_emoji = "😐"
    emotion_scores = {}
    try:
        faces = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False)
        for face in faces:
            x, y, w, h = face['region']['x'], face['region']['y'], face['region']['w'], face['region']['h']
            emotion = face['dominant_emotion']
            emotion_scores = face['emotion']
            detected_emoji = emotion_emoji.get(emotion.lower(), "🙂")
            cv2.rectangle(frame, (x, y), (x + w, y + h), (102, 255, 102), 2)
            cv2.putText(frame, f"{emotion} {detected_emoji}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 102, 255), 2)
    except:
        pass
    return frame, detected_emoji, emotion_scores

# Webcam logic
if start_btn:
    cap = cv2.VideoCapture(0)
    stop_button = stop_btn.button("⏹️ Stop Webcam")
    st.markdown('<div class="emotion">🔍 Detecting Emotions...</div>', unsafe_allow_html=True)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("⚠️ Cannot read from webcam.")
            break

        frame, emoji, scores = detect_emotion(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame)
        emotion_text.markdown(f"<div class='emotion'>Current Emotion: {emoji}</div>", unsafe_allow_html=True)

        if scores:
            df = pd.DataFrame(scores.items(), columns=["Emotion", "Confidence"])
            df["Confidence"] = df["Confidence"].astype(float)
            chart_area.bar_chart(df.set_index("Emotion"))

        if stop_button:
            break

    cap.release()
    FRAME_WINDOW.empty()
    emotion_text.empty()
    chart_area.empty()
    stop_btn.empty()
    st.success("✅ Webcam stopped. Thanks for trying!")
