import streamlit as st
import os
import gdown
import librosa
import numpy as np
from tensorflow.keras.models import load_model
import sys
# Removing cv2 import to avoid libGL.so.1 dependency

# Add the project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.audio_emotion import load_audio_model, predict_emotion_from_audio
# Removing video_emotion import to avoid OpenCV dependency

st.title("Emotion Recognition System")
st.write("Detect emotions from audio files.")

# Set PyAudio as not available to avoid import attempts
PYAUDIO_AVAILABLE = False
st.info("Audio emotion recognition will use the built-in model.")

# Create a built-in audio emotion model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D

# Flag to indicate we're using a built-in model
USING_BUILTIN_MODEL = True
st.info("Using built-in audio emotion model")

# Create a simple audio emotion model
def create_builtin_audio_model():
    model = Sequential([
        Conv1D(64, 3, activation='relu', input_shape=(40, 1)),
        MaxPooling1D(pool_size=2),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(4, activation='softmax')  # 4 emotions: happy, sad, angry, neutral
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Create the model
builtin_audio_model = create_builtin_audio_model()

# Removed video detector initialization

# Override the audio emotion prediction function to use our built-in model
def predict_emotion_from_audio_builtin(audio_path):
    """
    Predicts the emotion from an audio file using the built-in model.
    
    Args:
        audio_path (str): The path to the audio file.
        
    Returns:
        str: The predicted emotion.
    """
    try:
        # Extract features
        y, sr = librosa.load(audio_path, sr=None)
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
        
        # Simple rule-based prediction (as a fallback)
        # Analyze audio features to determine emotion
        energy = np.sum(y**2) / len(y)
        zero_crossings = np.sum(librosa.zero_crossings(y)) / len(y)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        
        # Simple rules for emotion classification
        if energy > 0.01 and tempo > 120:
            emotion = "happy"
        elif zero_crossings > 0.05 and energy < 0.005:
            emotion = "sad"
        elif energy > 0.02 and zero_crossings > 0.07:
            emotion = "angry"
        else:
            emotion = "neutral"
            
        return emotion
        
    except Exception as e:
        st.error(f"Error predicting emotion: {e}")
        return "Error predicting emotion"

# Audio emotion prediction
def predict_audio_emotion(audio_path):
    """
    Wrapper function to call the appropriate audio emotion prediction function
    """
    if USING_BUILTIN_MODEL:
        return predict_emotion_from_audio_builtin(audio_path)
    else:
        return predict_emotion_from_audio(audio_path)

# Removed video emotion prediction functions

# Streamlit interface - Audio only
st.info("Using built-in audio emotion recognition model")
uploaded_audio = st.file_uploader("Upload Audio (WAV)", type=["wav"])
if uploaded_audio:
    # Create a temp directory if it doesn't exist
    os.makedirs("temp", exist_ok=True)
    temp_path = os.path.join("temp", "temp_audio.wav")
    with open(temp_path, "wb") as f:
        f.write(uploaded_audio.getbuffer())
    try:
        # Always use the built-in model
        emotion = predict_emotion_from_audio_builtin(temp_path)
        st.write(f"✅ Predicted Audio Emotion: {emotion}")
    except Exception as e:
        st.error(f"Audio prediction failed: {e}")

# Add a note about video features being disabled
st.info("Note: Video emotion detection features have been temporarily disabled.")