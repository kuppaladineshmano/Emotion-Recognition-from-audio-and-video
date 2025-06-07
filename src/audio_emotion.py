import librosa
import numpy as np
from tensorflow.keras.models import load_model
import os
import sys

# Check if PyAudio is available
PYAUDIO_AVAILABLE = True
try:
    import pyaudio
except ImportError:
    PYAUDIO_AVAILABLE = False
    print("Warning: PyAudio is not available. Audio emotion recognition may not work properly.")

# Load the pre-trained model
MODEL_PATH = 'models/audio_emotion_model.h5'
model = None

def load_audio_model():
    """
    Loads the audio emotion recognition model.
    
    Returns:
        bool: True if the model was loaded successfully, False otherwise.
    """
    global model
    try:
        if os.path.exists(MODEL_PATH):
            model = load_model(MODEL_PATH)
            return True
        else:
            print(f"Model file not found at {MODEL_PATH}")
            return False
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

# Define the emotions
EMOTIONS = ["happy", "sad", "angry", "neutral"]

def extract_features(audio_path):
    """
    Extracts features from an audio file.
    
    Args:
        audio_path (str): The path to the audio file.
        
    Returns:
        np.array: The extracted features.
    """
    try:
        y, sr = librosa.load(audio_path, sr=None)
    except Exception as e:
        print(f"Error loading audio file: {e}")
        # Return a default feature vector of zeros
        return np.zeros(40)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfccs

def predict_emotion_from_audio(audio_path):
    """
    Predicts the emotion from an audio file.
    
    Args:
        audio_path (str): The path to the audio file.
        
    Returns:
        str: The predicted emotion.
    """
    if not PYAUDIO_AVAILABLE:
        return "PyAudio is not available. Audio emotion recognition is disabled."
        
    global model
    if model is None:
        if not load_audio_model():
            return "Model not found or could not be loaded"
        
    try:
        features = extract_features(audio_path)
        features = np.expand_dims(features, axis=0)
        features = np.expand_dims(features, axis=2)
        
        prediction = model.predict(features)
        predicted_emotion = EMOTIONS[np.argmax(prediction)]
    except Exception as e:
        print(f"Error predicting emotion: {e}")
        return f"Error predicting emotion: {str(e)}"
    
    return predicted_emotion

# Placeholder for training the model
def train_model():
    """
    This is a placeholder function to train the audio emotion recognition model.
    You will need to implement this function using the RAVDESS dataset.
    """
    pass