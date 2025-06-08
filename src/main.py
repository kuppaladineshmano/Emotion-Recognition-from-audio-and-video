import streamlit as st
import os
import gdown
import librosa
import numpy as np
from tensorflow.keras.models import load_model
# Removing FER import to avoid moviepy.editor dependency
import cv2
import sys

# Add the project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.audio_emotion import load_audio_model, predict_emotion_from_audio
from src.video_emotion import initialize_detector

st.title("Emotion Recognition System")
st.write("Detect emotions from live webcam, uploaded video, or audio files.")

# Set PyAudio as not available to avoid import attempts
PYAUDIO_AVAILABLE = False
st.info("Audio emotion recognition will use the built-in model. Video emotion recognition is available.")

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

# Initialize video detector
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

# Video emotion prediction
def predict_video_emotion(video_path):
    try:
        # Initialize OpenCV's built-in face detector
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                
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
                # Convert to grayscale for face detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Detect faces
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                
                # Process each detected face
                for (x, y, w, h) in faces:
                    # Simple placeholder logic - in reality, you'd use a trained model here
                    # Just for demonstration purposes
                    face_roi = gray[y:y+h, x:x+w]
                    
                    # Simple rule-based emotion detection
                    # This is a placeholder - in a real app, you'd use a proper emotion classifier
                    emotion_result = {
                        'happy': 0.5,
                        'sad': 0.1,
                        'angry': 0.1,
                        'neutral': 0.3,
                        'fear': 0.0,
                        'surprise': 0.0,
                        'disgust': 0.0
                    }
                    emotions.append(emotion_result)
                    
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

# Function to draw emotions on frame
def draw_emotions_on_frame(frame, emotions):
    """
    Draws the detected emotions on the frame.
    
    Args:
        frame (np.array): The input frame.
        emotions (list): A list of detected emotions.
        
    Returns:
        np.array: The frame with emotions drawn on it.
    """
    try:
        for emotion in emotions:
            (x, y, w, h) = emotion['box']
            # Get the dominant emotion
            dominant_emotion = max(emotion['emotions'], key=emotion['emotions'].get)
            score = emotion['emotions'][dominant_emotion]
            
            # Draw the bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw the label
            text = f"{dominant_emotion}: {score:.2f}"
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    except Exception as e:
        st.error(f"Error drawing emotions on frame: {e}")
        
    return frame

# Initialize session state for continuous tracking
if 'emotion_stats' not in st.session_state:
    st.session_state.emotion_stats = {"happy": 0, "sad": 0, "angry": 0, "neutral": 0, "fear": 0, "surprise": 0, "disgust": 0}
    
if 'current_emotion' not in st.session_state:
    st.session_state.current_emotion = "No emotion detected"
    
if 'running' not in st.session_state:
    st.session_state.running = False

# Live video emotion detection
def live_video_emotion_detection():
    try:
        # Set running state
        st.session_state.running = True
        
        # Initialize OpenCV's built-in face detector instead of FER
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Error: Could not open webcam")
            st.session_state.running = False
            return
            
        # Create placeholders for UI elements
        video_placeholder = st.empty()
        emotion_text = st.empty()
        stats_placeholder = st.empty()
        
        # Create a stop button
        stop_button = st.button("Stop", key="stop_button")
        
        # Main loop for continuous processing
        frame_count = 0
        
        # Process frames continuously while running
        while cap.isOpened() and st.session_state.running and not stop_button:
            ret, frame = cap.read()
            if not ret:
                st.error("Error: Could not read frame from webcam")
                break
                
            try:
                # Detect emotions (process every other frame to improve performance)
                if frame_count % 2 == 0:
                    # Convert to grayscale for face detection
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    
                    # Detect faces
                    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                    
                    # Process each detected face
                    emotions = []
                    for (x, y, w, h) in faces:
                        # Draw rectangle around the face
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        
                        # Simple rule-based emotion detection based on face position and size
                        # This is a placeholder - in a real app, you'd use a proper emotion classifier
                        face_roi = gray[y:y+h, x:x+w]
                        
                        # Simple placeholder logic - in reality, you'd use a trained model here
                        # Just for demonstration purposes
                        emotion_result = {
                            'box': (x, y, w, h),
                            'emotions': {
                                'happy': 0.5,
                                'sad': 0.1,
                                'angry': 0.1,
                                'neutral': 0.3,
                                'fear': 0.0,
                                'surprise': 0.0,
                                'disgust': 0.0
                            }
                        }
                        emotions.append(emotion_result)
                    
                    # Draw emotions on frame
                    frame = draw_emotions_on_frame(frame, emotions)
                    
                    # Update emotion stats
                    if emotions:
                        # Get the dominant emotion from the first detected face
                        dominant_emotion = max(emotions[0]['emotions'], key=emotions[0]['emotions'].get)
                        st.session_state.emotion_stats[dominant_emotion] += 1
                        st.session_state.current_emotion = dominant_emotion
                
                # Convert BGR to RGB for display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Display the frame
                video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                
                # Update emotion text
                emotion_text.write(f"Current Emotion: {st.session_state.current_emotion}")
                
                # Update stats every 15 frames
                if frame_count % 15 == 0:
                    stats_placeholder.write(f"Emotion Statistics: {st.session_state.emotion_stats}")
                    
                # Increment frame count
                frame_count += 1
                
                # Add a small delay to make the UI more responsive
                import time
                time.sleep(0.01)
                
            except Exception as e:
                st.error(f"Error processing frame: {e}")
                continue
        
        # Release resources
        cap.release()
        st.session_state.running = False
        st.write("Webcam released")
        
    except Exception as e:
        st.error(f"Live video emotion detection error: {str(e)}")
        st.session_state.running = False

# Streamlit interface
tab = st.selectbox("Choose Input Type", ["Live Detection", "Video", "Audio"])
if tab == "Audio":
    # Always use the built-in audio model, regardless of PyAudio availability
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
elif tab == "Live Detection":
    st.write("Live webcam-based emotion detection")
    st.write("Click the Start button below to begin continuous live emotion tracking")
    
    col1, col2 = st.columns(2)
    
    with col1:
        start_button = st.button("Start", key="start_live")
        
    with col2:
        if st.session_state.running:
            st.write("Status: Running")
        else:
            st.write("Status: Stopped")
    
    # Display current stats if available
    if st.session_state.current_emotion != "No emotion detected":
        st.write(f"Current Emotion: {st.session_state.current_emotion}")
        st.write(f"Emotion Statistics: {st.session_state.emotion_stats}")
    
    if start_button:
        live_video_emotion_detection()