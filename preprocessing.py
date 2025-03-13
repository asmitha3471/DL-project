import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_preprocess():
    df = pd.read_csv("features_30_sec.csv")

    df = df.drop(columns=["filename", "length"])

    X = df.drop(columns=["label"])
    y = df["label"] 

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler

if __name__ == "__main__":
    X, y, _ = load_and_preprocess()
    print("✅ Preprocessing Completed! Data Shape:", X.shape)