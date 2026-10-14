"""Train the educational classifier on generated sample data. Not for clinical use."""
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

ROOT = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(ROOT, "data", "synthetic_medical_data.csv"))
features = ["age", "fever", "cough", "fatigue", "headache", "body_pain", "nausea",
            "sore_throat", "breathlessness", "temperature_c", "oxygen_saturation_pct"]
X, y = df[features], df["demo_label"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=17, stratify=y
)
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", RandomForestClassifier(n_estimators=180, random_state=17, class_weight="balanced"))
])
model.fit(X_train, y_train)
accuracy = accuracy_score(y_test, model.predict(X_test))
os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
joblib.dump({"model": model, "features": features, "accuracy": float(accuracy)},
            os.path.join(ROOT, "models", "disease_demo_model.joblib"))
print(f"Saved model. Synthetic test accuracy: {accuracy:.3f}")
print("Reminder: this is not a clinical model.")
