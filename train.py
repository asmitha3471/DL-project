import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv("features_30_sec.csv")

expected_features = [
    "spectral_centroid_mean", 
    "rms_mean",
    "mfcc1_mean",
    "spectral_bandwidth_mean",
    "rolloff_mean",
    "zero_crossing_rate_mean",
    "tempo",
    "chroma_stft_mean"
]

missing_features = [feat for feat in expected_features if feat not in df.columns]
if missing_features:
    raise ValueError(f"Missing features in dataset: {missing_features}")

X = df[expected_features]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

tree = DecisionTreeClassifier(random_state=10)
tree.fit(X_train_scaled, y_train)
tree_preds = tree.predict(X_test_scaled)

param_grid = {'C': [0.01, 0.1, 1, 10, 100]}
grid_search = GridSearchCV(LogisticRegression(max_iter=1000, random_state=10), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)

best_logreg = grid_search.best_estimator_
logreg_preds = best_logreg.predict(X_test_scaled)

print("\n🎯 Decision Tree Performance:")
print(confusion_matrix(y_test, tree_preds))
print(classification_report(y_test, tree_preds))

print("\n🎯 Logistic Regression Performance:")
print(confusion_matrix(y_test, logreg_preds))
print(classification_report(y_test, logreg_preds))

joblib.dump(best_logreg, "models/logistic_regression_model.pkl")
joblib.dump(tree, "models/decision_tree_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("\nModels trained and saved successfully!")