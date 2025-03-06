import pandas as pd
import os

# 📌 Check if the dataset exists
csv_file = "features_30_sec.csv"
if not os.path.exists(csv_file):
    print(f"❌ Error: The dataset file '{csv_file}' does not exist. Make sure it's in the correct directory.")
    exit()

# 📌 Load the dataset
print("🔄 Loading dataset...")
df = pd.read_csv(csv_file)

# 📌 Remove unnecessary columns
df = df.drop(columns=["filename", "length"])  # Drop filename & length columns

# 📌 Save processed features as a new CSV
df.to_csv("music_features.csv", index=False)
print("✅ Feature extraction complete! Data saved to 'music_features.csv'.")