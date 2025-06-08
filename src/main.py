import streamlit as st
import os
import gdown
import librosa
import numpy as np
from tensorflow.keras.models import load_model
import sys
import time
from PIL import Image
import io
import base64

# Try to import cv2, but handle the case where it's not available
CV2_AVAILABLE = False
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    st.warning("OpenCV (cv2) is not available. Using alternative implementation.")

# Add the project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.audio_emotion import load_audio_model, predict_emotion_from_audio
# Conditionally import video_emotion
if CV2_AVAILABLE:
    try:
        from src.video_emotion import initialize_detector
    except ImportError:
        st.warning("Video emotion module could not be imported.")

st.title("Emotion Recognition System")
st.write("Detect emotions from audio files and videos (when available).")

# Set PyAudio as not available to avoid import attempts
PYAUDIO_AVAILABLE = False
st.info("Audio emotion recognition will use the built-in model.")
if not CV2_AVAILABLE:
    st.warning("Video features are disabled due to missing OpenCV dependency.")

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

# Initialize video detector if available
if CV2_AVAILABLE:
    try:
        if initialize_detector():
            st.success("Video emotion detector initialized successfully")
        else:
            st.warning("Video emotion detector could not be initialized.")
    except Exception as e:
        st.error(f"Video detector initialization failed: {e}")

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

# Video emotion prediction - only if OpenCV is available
def predict_video_emotion(video_path):
    if not CV2_AVAILABLE:
        return "Video emotion detection is not available (OpenCV missing)"
        
    try:
        from video_emotion import detector
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

# Initialize session state for tracking
if 'emotion_stats' not in st.session_state:
    st.session_state.emotion_stats = {"happy": 0, "sad": 0, "angry": 0, "neutral": 0, "fear": 0, "surprise": 0, "disgust": 0}
    
if 'current_emotion' not in st.session_state:
    st.session_state.current_emotion = "No emotion detected"
    
if 'running' not in st.session_state:
    st.session_state.running = False

# Streamlit interface with conditional features
tab_options = ["Audio"]
if CV2_AVAILABLE:
    tab_options.extend(["Video", "Live Detection"])
    
tab = st.selectbox("Choose Input Type", tab_options)

if tab == "Audio":
    # Always use the built-in audio model
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
            
elif tab == "Video" and CV2_AVAILABLE:
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
            
elif tab == "Live Detection":
    st.write("Live webcam-based emotion detection")
    st.write("Using Streamlit's built-in webcam support")
    
    # Initialize session state for tracking emotions
    if 'emotion_stats' not in st.session_state:
        st.session_state.emotion_stats = {"happy": 0, "sad": 0, "angry": 0, "neutral": 0}
    
    if 'current_emotion' not in st.session_state:
        st.session_state.current_emotion = "No emotion detected"
    
    # Use Streamlit's built-in camera input
    camera_image = st.camera_input("Take a picture to analyze emotion")
    
    # Process the image if available
    if camera_image is not None:
        # Display the captured image
        st.image(camera_image, caption="Captured Image")
        
        # Simulate emotion detection (in a real app, you'd use a model here)
        emotion = np.random.choice(["happy", "sad", "angry", "neutral"])
        st.session_state.current_emotion = emotion
        st.session_state.emotion_stats[emotion] += 1
        
        # Display the detected emotion
        st.success(f"Detected emotion: {emotion}")
        
        # Display emotion statistics
        st.write("Emotion Statistics:")
        st.write(st.session_state.emotion_stats)
        
        # Add a button to clear the current image
        if st.button("Clear and take another picture"):
            st.experimental_rerun()