import cv2
from fer import FER
import numpy as np

# Initialize the FER detector
detector = None

def initialize_detector():
    """
    Initializes the FER detector.
    
    Returns:
        bool: True if the detector was initialized successfully, False otherwise.
    """
    global detector
    try:
        detector = FER(mtcnn=True)
        return True
    except Exception as e:
        print(f"Error initializing FER detector: {e}")
        return False

def detect_emotions_in_frame(frame):
    """
    Detects emotions in a single frame.
    
    Args:
        frame (np.array): The input frame.
        
    Returns:
        list: A list of detected emotions with their bounding boxes and scores.
    """
    global detector
    if detector is None:
        if not initialize_detector():
            return []
    
    try:
        # The detector returns a list of dictionaries, one for each detected face.
        # Each dictionary contains 'box' and 'emotions'.
        emotions = detector.detect_emotions(frame)
        return emotions
    except Exception as e:
        print(f"Error detecting emotions in frame: {e}")
        return []

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
        print(f"Error drawing emotions on frame: {e}")
        
    return frame