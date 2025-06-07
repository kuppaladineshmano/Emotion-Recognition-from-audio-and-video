import streamlit as st
import cv2
import numpy as np
from video_emotion import detect_emotions_in_frame, draw_emotions_on_frame
from audio_emotion import predict_emotion_from_audio
import pyaudio
import wave
import os

# Audio recording parameters
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
WAVE_OUTPUT_FILENAME = "temp_audio.wav"

def record_audio():
    """
    Records audio from the microphone.
    """
    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    st.write("Recording...")

    frames = []

    for i in range(0, int(RATE / CHUNK * 5)):  # Record for 5 seconds
        data = stream.read(CHUNK)
        frames.append(data)

    st.write("Finished recording.")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

def main():
    st.title("Emotion Recognition System")

    st.header("Video Emotion Recognition")
    run_video = st.checkbox("Run Video Emotion Recognition")

    FRAME_WINDOW = st.image([])
    cap = cv2.VideoCapture(0)

    if run_video:
        while True:
            ret, frame = cap.read()
            if not ret:
                st.write("Failed to grab frame")
                break
            
            emotions = detect_emotions_in_frame(frame)
            frame_with_emotions = draw_emotions_on_frame(frame, emotions)
            FRAME_WINDOW.image(frame_with_emotions, channels="BGR")
            
            if not run_video:
                break
    else:
        st.write("Video emotion recognition is stopped.")

    st.header("Audio Emotion Recognition")
    run_audio = st.button("Record and Analyze Audio")

    if run_audio:
        record_audio()
        emotion = predict_emotion_from_audio(WAVE_OUTPUT_FILENAME)
        st.write(f"Predicted Emotion: {emotion}")
        if os.path.exists(WAVE_OUTPUT_FILENAME):
            os.remove(WAVE_OUTPUT_FILENAME)

if __name__ == "__main__":
    main()