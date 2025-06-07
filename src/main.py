import streamlit as st
import cv2
import numpy as np
import os
import gdown
from fer import FER
import librosa
from tensorflow.keras.models import load_model

# --- Configuration ---
AUDIO_MODEL_URL = "https://docs.google.com/uc?export=download&id=1g_y14-C5bYV3-d_a-TO2g_yLg-ZgH-qj"
AUDIO_MODEL_PATH = "models/audio_emotion_model.h5"
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# --- Model Loading ---
@st.cache_resource
def download_model():
    """Downloads the audio model from Google Drive if it doesn't exist."""
    if not os.path.exists(AUDIO_MODEL_PATH):
        st.info("Downloading audio emotion recognition model... This may take a moment.")
        os.makedirs(os.path.dirname(AUDIO_MODEL_PATH), exist_ok=True)
        gdown.download(AUDIO_MODEL_URL, AUDIO_MODEL_PATH, quiet=False)
    return load_model(AUDIO_MODEL_PATH)

@st.cache_resource
def get_video_detector():
    """Initializes the FER video emotion detector."""
    return FER(mtcnn=True)

audio_model = download_model()
video_detector = get_video_detector()

# --- Audio Processing ---
def predict_audio_emotion(file_path):
    """Predicts emotion from an audio file."""
    try:
        y, sr = librosa.load(file_path, duration=3, offset=0.5)
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
        mfcc = np.expand_dims(mfcc, axis=-1)
        mfcc = np.expand_dims(mfcc, axis=0)
        prediction = audio_model.predict(mfcc)
        predicted_emotion = EMOTIONS[np.argmax(prediction)]
        return predicted_emotion
    except Exception as e:
        st.error(f"Error processing audio: {e}")
        return None

# --- UI ---
st.set_page_config(page_title="Emotion Recognition System", layout="wide")
st.title("Emotion Recognition from Audio & Video")

st.sidebar.header("Controls")
app_mode = st.sidebar.selectbox("Choose the mode", ["Video Emotion Recognition", "Audio Emotion Recognition"])

if app_mode == "Video Emotion Recognition":
    st.header("Video Emotion Recognition")
    st.info("Position your face in the camera frame to see the detected emotion.")
    
    run_video = st.checkbox("Start Camera", value=True)
    FRAME_WINDOW = st.image([])
    
    cap = cv2.VideoCapture(0)

    if run_video:
        while True:
            ret, frame = cap.read()
            if not ret:
                st.warning("Could not read frame from camera. Please ensure it is not being used by another application.")
                break
            
            # Detect emotions
            result = video_detector.detect_emotions(frame)
            
            # Draw bounding boxes and emotions
            for face in result:
                (x, y, w, h) = face["box"]
                emotions = face["emotions"]
                dominant_emotion = max(emotions, key=emotions.get)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, dominant_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            FRAME_WINDOW.image(frame, channels="BGR")
    else:
        cap.release()
        st.info("Camera is off.")

elif app_mode == "Audio Emotion Recognition":
    st.header("Audio Emotion Recognition")
    st.info("Upload an audio file (.wav, .mp3) to analyze the emotion.")
    
    uploaded_file = st.file_uploader("Choose an audio file...", type=["wav", "mp3"])

    if uploaded_file is not None:
        # Save the uploaded file temporarily
        with open("temp_audio.wav", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.audio(uploaded_file, format='audio/wav')
        
        if st.button("Analyze Emotion"):
            with st.spinner("Analyzing..."):
                emotion = predict_audio_emotion("temp_audio.wav")
                if emotion:
                    st.success(f"Predicted Emotion: **{emotion.capitalize()}**")
            os.remove("temp_audio.wav")