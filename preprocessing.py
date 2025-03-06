import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_preprocess():
    # Load dataset
    df = pd.read_csv("features_30_sec.csv")  # Use the correct dataset

    # Drop unnecessary columns
    df = df.drop(columns=["filename", "length"])  # Remove filename and length

    # Extract features and labels
    X = df.drop(columns=["label"])  # Features
    y = df["label"]  # Labels (Genres)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler

# Test script
if __name__ == "__main__":
    X, y, _ = load_and_preprocess()
    print("✅ Preprocessing Completed! Data Shape:", X.shape)