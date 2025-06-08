---
title: Emotion Recognition System
emoji: 😀
colorFrom: blue
colorTo: pink
sdk: streamlit
sdk_version: 1.39.0
app_file: app.py
pinned: false
---

# Emotion Recognition System

This project is an Emotion Recognition System that detects emotions from both live video (facial expressions) and audio (speech) inputs. It is designed with a focus on being beginner-friendly and deployable as a Streamlit web app.

## Features

-   **Real-time Emotion Detection:** Detects emotions from live webcam video and microphone audio.
-   **Facial Emotion Recognition:** Uses a pre-trained model to detect emotions like happy, sad, angry, and neutral from facial expressions.
-   **Speech Emotion Recognition:** Uses a pre-trained model to classify emotions from speech.
-   **Combined Prediction:** Integrates video and audio predictions for a more accurate result.
-   **Streamlit Web App:** A user-friendly web interface to interact with the system.
-   **Model Evaluation:** Includes a Jupyter notebook to evaluate the performance of the models.

## Project Structure

```
emotion_recognition_system/
├── src/
│   ├── main.py
│   ├── video_emotion.py
│   ├── audio_emotion.py
│   └── utils.py
├── data/
├── models/
├── notebooks/
│   └── evaluation.ipynb
├── requirements.txt
└── README.md
```

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/emotion-recognition-system.git
    cd emotion-recognition-system
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download the datasets:**
    -   **RAVDESS (Audio):** Download from [https://zenodo.org/record/1188976](https://zenodo.org/record/1188976) and place it in the `data/` directory.
    -   **fer2013 (Video):** Download from Kaggle and place it in the `data/` directory.

5.  **Run the Streamlit app:**
    ```bash
    streamlit run src/main.py
    ```

## Usage

-   Open the Streamlit app in your browser.
-   Click the "Start Webcam" button to start video emotion recognition.
-   Click the "Start Microphone" button to start audio emotion recognition.
-   The app will display the detected emotions and confidence scores in real-time.

## Evaluation

The `notebooks/evaluation.ipynb` notebook contains the code to evaluate the performance of the models. It includes metrics like accuracy, precision, recall, F1-score, and a confusion matrix.

## Tools and Libraries

-   **Python**
-   **OpenCV**
-   **DeepFace** (or **fer**)
-   **Librosa**
-   **TensorFlow**
-   **Streamlit**
-   **NumPy**
-   **Pandas**
-   **Scikit-learn**
-   **Matplotlib**
