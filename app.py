import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

model_files = ["models/logistic_regression_model.pkl", "models/scaler.pkl"]

for model in model_files:
    if not os.path.exists(model):
        st.error(f"Model file missing: {model}. Please train the models first.")
        st.stop()

logreg_model = joblib.load("models/logistic_regression_model.pkl")
scaler = joblib.load("models/scaler.pkl")

def extract_audio_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        if y is None or len(y) == 0:
            return None 

        features = {
            "spectral_centroid_mean": np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),  
            "rms_mean": np.mean(librosa.feature.rms(y=y)),  
            "mfcc1_mean": np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=1)),
            "spectral_bandwidth_mean": np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)),  
            "rolloff_mean": np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)),  
            "zero_crossing_rate_mean": np.mean(librosa.feature.zero_crossing_rate(y)),  
            "tempo": librosa.beat.tempo(y=y, sr=sr)[0] if len(y) > 0 else 0,  
            "chroma_stft_mean": np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
        }

        return np.array(list(features.values())), list(features.keys())
    
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None, None

def predict_genre_from_audio(file_path):
    features, feature_names = extract_audio_features(file_path)

    if features is not None:
        print(f"Extracted Features: {feature_names}")
        try:
            features_df = pd.DataFrame([features], columns=feature_names)

            scaled_features = scaler.transform(features_df)

            predicted_genre = logreg_model.predict(scaled_features)[0]
            return predicted_genre
        except ValueError as e:
            st.error(f"🚨 Feature mismatch error: {e}")
            return None
    return None

def run_streamlit_app():
    st.title("🎵 Music Genre Prediction App")
    st.write("Upload a music file to predict the genre.")

    uploaded_file = st.file_uploader("Upload an audio file (.mp3, .wav)", type=["mp3", "wav"])
    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")

        file_path = f"temp_audio.{uploaded_file.name.split('.')[-1]}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if st.button("Predict Genre from Audio 🎧"):
            genre = predict_genre_from_audio(file_path)
            if genre:
                st.success(f"🎶 Predicted Genre: **{genre}**")
            else:
                st.error("Could not extract features from the audio.")

if __name__ == "__main__":
    run_streamlit_app()