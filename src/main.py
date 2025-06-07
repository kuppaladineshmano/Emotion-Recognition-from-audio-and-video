import streamlit as st
import os
import gdown
import librosa
import numpy as np
from tensorflow.keras.models import load_model
from fer import FER
import cv2
from src.audio_emotion import load_audio_model, predict_emotion_from_audio
from src.video_emotion import initialize_detector

st.title("Emotion Recognition from Audio and Video")
st.write("Upload a WAV file for audio-based emotion prediction or an MP4 file for video-based emotion prediction.")

# Check if PyAudio is available
PYAUDIO_AVAILABLE = True
try:
    import pyaudio
except ImportError:
    PYAUDIO_AVAILABLE = False
    st.warning("⚠️ PyAudio is not available. Audio emotion recognition may not work properly. Video emotion recognition is still available.")

# Download audio model
AUDIO_MODEL_PATH = 'models/audio_emotion_model.h5'
# Direct file ID for Google Drive
AUDIO_MODEL_ID = '1ohI2trIkypZP7EPT9qRtWcnvHX05dZet'
if not os.path.exists(AUDIO_MODEL_PATH):
    os.makedirs('models', exist_ok=True)
    try:
        # Use gdown with file ID instead of URL
        gdown.download(id=AUDIO_MODEL_ID, output=AUDIO_MODEL_PATH, quiet=False)
        st.success("Audio model downloaded successfully")
    except Exception as e:
        st.error(f"Failed to download audio model: {e}")

# Load models
try:
    # Load audio model
    if load_audio_model():
        st.success("Audio model loaded successfully")
    else:
        st.warning("Audio model could not be loaded. Please check if the model file exists.")
        
    # Initialize video detector
    if initialize_detector():
        st.success("Video emotion detector initialized successfully")
    else:
        st.warning("Video emotion detector could not be initialized.")
except Exception as e:
    st.error(f"Model loading failed: {e}")

# Audio emotion prediction
def predict_audio_emotion(audio_path):
    """
    Wrapper function to call the audio emotion prediction from audio_emotion.py
    """
    return predict_emotion_from_audio(audio_path)

# Video emotion prediction
def predict_video_emotion(video_path):
    try:
        from src.video_emotion import detector
        if detector is None:
            if not initialize_detector():
                return "Video emotion detector not initialized"
                
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return "Error opening video file"
            
        emotions = []
        frame_count = 0
        max_frames = 100  # Limit the number of frames to process
        
        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process every 5th frame to speed up analysis
            if frame_count % 5 == 0:
                result = detector.detect_emotions(frame)
                if result:
                    emotions.append(result[0]['emotions'])
                    
            frame_count += 1
            
        cap.release()
        
        if not emotions:
            return "No emotions detected"
            
        # Aggregate emotions (e.g., most frequent)
        try:
            dominant_emotion = max(set([max(e, key=e.get) for e in emotions]),
                                  key=[max(e, key=e.get) for e in emotions].count)
            return dominant_emotion
        except Exception as e:
            st.error(f"Error aggregating emotions: {e}")
            return "Error processing emotions"
            
    except Exception as e:
        st.error(f"Video emotion prediction error: {e}")
        return f"Error: {str(e)}"

# Streamlit interface
tab = st.selectbox("Choose Input Type", ["Audio", "Video"])
if tab == "Audio":
    if not PYAUDIO_AVAILABLE:
        st.error("❌ Audio emotion recognition requires PyAudio, which is not available in this environment. Please use video emotion recognition instead.")
    else:
        uploaded_audio = st.file_uploader("Upload Audio (WAV)", type=["wav"])
        if uploaded_audio:
            # Create a temp directory if it doesn't exist
            os.makedirs("temp", exist_ok=True)
            temp_path = os.path.join("temp", "temp_audio.wav")
            with open(temp_path, "wb") as f:
                f.write(uploaded_audio.getbuffer())
            try:
                emotion = predict_audio_emotion(temp_path)
                st.write(f"✅ Predicted Audio Emotion: {emotion}")
            except Exception as e:
                st.error(f"Audio prediction failed: {e}")
elif tab == "Video":
    uploaded_video = st.file_uploader("Upload Video (MP4)", type=["mp4"])
    if uploaded_video:
        # Create a temp directory if it doesn't exist
        os.makedirs("temp", exist_ok=True)
        temp_path = os.path.join("temp", "temp_video.mp4")
        with open(temp_path, "wb") as f:
            f.write(uploaded_video.getbuffer())
        try:
            emotion = predict_video_emotion(temp_path)
            st.write(f"✅ Predicted Video Emotion: {emotion}")
        except Exception as e:
            st.error(f"Video prediction failed: {e}")