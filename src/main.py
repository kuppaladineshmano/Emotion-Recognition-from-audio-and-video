import streamlit as st
import os
import gdown
import librosa
import numpy as np
from tensorflow.keras.models import load_model
from fer import FER
import cv2

st.title("Emotion Recognition from Audio and Video")
st.write("Upload a WAV file for audio-based emotion prediction or an MP4 file for video-based emotion prediction.")

# Download audio model
AUDIO_MODEL_PATH = 'models/audio_emotion_model.h5'
AUDIO_MODEL_URL = 'https://drive.google.com/drive/folders/1ohI2trIkypZP7EPT9qRtWcnvHX05dZet?usp=drive_link'
if not os.path.exists(AUDIO_MODEL_PATH):
    os.makedirs('models', exist_ok=True)
    gdown.download(AUDIO_MODEL_URL, AUDIO_MODEL_PATH, quiet=False)

# Load audio model
try:
    audio_model = load_model(AUDIO_MODEL_PATH)
    st.success("Audio model loaded")
except Exception as e:
    st.error(f"Audio model loading failed: {e}")

# Audio emotion prediction
def extract_audio_features(audio_path):
    y, sr = librosa.load(audio_path, sr=22050)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfccs_scaled = np.mean(mfccs.T, axis=0)
    return mfccs_scaled

def predict_audio_emotion(audio_path):
    features = extract_audio_features(audio_path)
    features = features.reshape(1, features.size, 1)
    prediction = audio_model.predict(features)
    emotions = ['Angry', 'Happy', 'Neutral', 'Sad']
    return emotions[np.argmax(prediction)]

# Video emotion prediction
def predict_video_emotion(video_path):
    detector = FER(mtcnn=True)
    cap = cv2.VideoCapture(video_path)
    emotions = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        result = detector.detect_emotions(frame)
        if result:
            emotions.append(result[0]['emotions'])
    cap.release()
    if not emotions:
        return "No emotions detected"
    # Aggregate emotions (e.g., most frequent)
    dominant_emotion = max(set([max(e, key=e.get) for e in emotions]), key=[max(e, key=e.get) for e in emotions].count)
    return dominant_emotion

# Streamlit interface
tab = st.selectbox("Choose Input Type", ["Audio", "Video"])
if tab == "Audio":
    uploaded_audio = st.file_uploader("Upload Audio (WAV)", type=["wav"])
    if uploaded_audio:
        with open("temp_audio.wav", "wb") as f:
            f.write(uploaded_audio.getbuffer())
        try:
            emotion = predict_audio_emotion("temp_audio.wav")
            st.write(f"✅ Predicted Audio Emotion: {emotion}")
        except Exception as e:
            st.error(f"Audio prediction failed: {e}")
elif tab == "Video":
    uploaded_video = st.file_uploader("Upload Video (MP4)", type=["mp4"])
    if uploaded_video:
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_video.getbuffer())
        try:
            emotion = predict_video_emotion("temp_video.mp4")
            st.write(f"✅ Predicted Video Emotion: {emotion}")
        except Exception as e:
            st.error(f"Video prediction failed: {e}")