# multilingual emotion detection in voice
Multilingual Emotion Detection in Voice
Overview
This project provides an end-to-end pipeline for multilingual speech emotion recognition using Python. It leverages OpenAI’s Whisper model for speech transcription and language detection, and uses machine learning techniques (SVM) for emotion classification based on audio features. The solution is designed to work with a variety of languages and can process audio from MP4 files.

Features
Automatic Speech Recognition (ASR): Uses Whisper to transcribe speech and detect the spoken language.

Emotion Recognition: Extracts MFCC and pitch features from audio and predicts emotions (happy, sad, angry, neutral) using an SVM classifier.

Multilingual Support: Works with speech in multiple languages.

Audio Conversion: Converts MP4 files to WAV for processing.

Easy Integration: Modular functions for each step in the pipeline.

Installation
Install the required Python packages:

bash
pip install openai-whisper librosa scikit-learn moviepy torch
How It Works
1. Audio Conversion
Converts an MP4 file to WAV format using MoviePy for easier audio processing.

2. Speech Transcription & Language Detection
Transcribes the audio and detects its language using OpenAI’s Whisper model.

3. Feature Extraction
Extracts:

MFCCs (Mel Frequency Cepstral Coefficients)

Pitch features

These are commonly used features for audio-based emotion recognition.

4. Emotion Classification
A Support Vector Machine (SVM) classifier is trained (demo uses synthetic data; in production, use real labeled emotion data) to predict the emotion from the extracted features.

5. Output
Prints the detected emotion, transcription, and language.

Usage
Example:

python
audio_path = "/content/happy voice.mp4"  # Replace with your actual file path
emotion_aware_speech_recognition(audio_path)
Expected Output:

text
MoviePy - Writing audio in /content/temp_audio.wav
MoviePy - Done.
Transcription: Hello how are you? Good morning everyone have a nice day
Detected Emotion: happy
Transcription: Hello how are you? Good morning everyone have a nice day
Language Detected: en
Pipeline Functions
convert_mp4_to_wav(mp4_path, wav_path): Converts MP4 audio to WAV.

transcribe_audio(audio_path): Transcribes speech and detects language using Whisper.

extract_audio_features(audio_path): Extracts MFCC and pitch features.

train_emotion_classifier(): Trains an SVM classifier (demo uses random data).

classify_emotion(features, classifier, le): Predicts emotion from features.

emotion_aware_speech_recognition(mp4_path): Orchestrates the pipeline from input file to output.

Code Snippet
python
import whisper
import librosa
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from moviepy.editor import AudioFileClip

# Load Whisper Model
model = whisper.load_model("base")

def transcribe_audio(audio_path):
    result = model.transcribe(audio_path)
    print("Transcription: ", result["text"])
    return result["text"], result["language"]

def extract_audio_features(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    pitch, _ = librosa.core.piptrack(y=y, sr=sr)
    pitch = np.mean(pitch, axis=1)
    mfcc_features = np.mean(mfcc, axis=1)
    pitch_features = pitch[:13]
    audio_features = np.concatenate([mfcc_features, pitch_features])
    return audio_features

def train_emotion_classifier():
    X = np.random.rand(100, 26)
    y = np.random.choice(['happy', 'sad', 'angry', 'neutral'], size=100)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    classifier = SVC(kernel='linear')
    classifier.fit(X, y_encoded)
    return classifier, le

def classify_emotion(features, classifier, le):
    emotion_idx = classifier.predict([features])[0]
    emotion = le.inverse_transform([emotion_idx])[0]
    return emotion

def convert_mp4_to_wav(mp4_path, wav_path):
    audio_clip = AudioFileClip(mp4_path)
    audio_clip.write_audiofile(wav_path, codec='pcm_s16le')

def emotion_aware_speech_recognition(mp4_path):
    wav_path = "/content/temp_audio.wav"
    convert_mp4_to_wav(mp4_path, wav_path)
    transcription, language = transcribe_audio(wav_path)
    audio_features = extract_audio_features(wav_path)
    emotion = classify_emotion(audio_features, emotion_classifier, label_encoder)
    print(f"Detected Emotion: {emotion}")
    print(f"Transcription: {transcription}")
    print(f"Language Detected: {language}")

# Train the classifier once
emotion_classifier, label_encoder = train_emotion_classifier()

# Example usage
audio_path = "/content/happy voice.mp4"
emotion_aware_speech_recognition(audio_path)
Notes
The demo classifier is trained on random data for illustration. For real applications, train with labeled emotion datasets.

Ensure your audio files are clear for best results.

You can expand emotion classes and language support by retraining the classifier with more diverse data.

References
OpenAI Whisper

Librosa Documentation

scikit-learn SVM

License
MIT License

