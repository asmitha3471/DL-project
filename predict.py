import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load saved models
logreg = joblib.load("models/logistic_regression_model.pkl")
tree = joblib.load("models/decision_tree_model.pkl")
scaler = joblib.load("models/scaler.pkl")
pca = joblib.load("models/pca.pkl")

def predict_genre(file_path):
    """ Predict the genre of a given audio file """
    # Load extracted features
    df = pd.read_csv(file_path)

    # Drop unnecessary columns
    df = df.drop(columns=["filename", "length", "label"], errors="ignore")  # 🔥 Drop "label"

    # Scale and apply PCA
    X_scaled = scaler.transform(df)
    X_pca = pca.transform(X_scaled)

    # Make predictions
    logreg_pred = logreg.predict(X_pca)[0]
    tree_pred = tree.predict(X_pca)[0]

    return logreg_pred, tree_pred

if __name__ == "__main__":
    file_path = "music_features.csv"  # Change if needed
    logreg_pred, tree_pred = predict_genre(file_path)
    print(f"🎵 Logistic Regression Prediction: {logreg_pred}")
    print(f"🎸 Decision Tree Prediction: {tree_pred}")