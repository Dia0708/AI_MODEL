import warnings
from sklearn.exceptions import InconsistentVersionWarning

# 🔇 Suppress sklearn version warning ONLY
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

import numpy as np
import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix


# ===============================
# PATHS
# ===============================
BASE_DIR = r"C:\Users\User\Desktop\HEARSKINERGY\AI_MODEL"
DATA_FILE = os.path.join(BASE_DIR, "training_data.csv")

MODEL_FILE = os.path.join(BASE_DIR, "health_model.pkl")
SCALER_FILE = os.path.join(BASE_DIR, "scaler.pkl")


# ===============================
# LOAD DATA
# ===============================
def load_data():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Training data not found: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE)

    # Normalize column names
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "")
        .str.replace("_", "")
    )

    print("📌 Detected columns:", list(df.columns))

    required = [
        "heartrate",
        "oxygensaturation",
        "gsrvalue",
        "stresspercent",
        "relaxationpercent",
        "overallresult",
    ]

    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Encode labels
    df["overallresult"] = df["overallresult"].map({
        "stable": 0,
        "critical": 1
    })

    X = df[
        [
            "heartrate",
            "oxygensaturation",
            "gsrvalue",
            "stresspercent",
            "relaxationpercent",
        ]
    ].values

    y = df["overallresult"].values

    return X, y


# ===============================
# TRAINING PIPELINE
# ===============================
def train_pipeline():
    print("\n🔄 Training model...")

    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = KNeighborsClassifier(
        n_neighbors=5,
        weights="distance"
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\n=== MODEL EVALUATION ===")
    print(classification_report(y_test, y_pred))
    print("CONFUSION MATRIX")
    print(confusion_matrix(y_test, y_pred))

    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)

    print("\n✅ Model and scaler saved successfully!")


# ===============================
# MANUAL PREDICTION
# ===============================
def predict_manual(hr, spo2, gsr, stress, relax):
    if not (os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE)):
        raise RuntimeError("Model not found. Please train first (Option 1).")

    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)

    input_data = np.array([[hr, spo2, gsr, stress, relax]])
    input_scaled = scaler.transform(input_data)

    probs = model.predict_proba(input_scaled)[0]

    eps = 0.15
    stable_knn = (probs[0] + eps) / (1 + 2 * eps)
    critical_knn = (probs[1] + eps) / (1 + 2 * eps)

    rule_score = 0.0
    if hr < 50 or hr > 100:
        rule_score += 0.25
    if spo2 < 90:
        rule_score += 0.30
    if gsr > 2700:
        rule_score += 0.20
    if stress > 60:
        rule_score += 0.15
    if relax < 40:
        rule_score += 0.10

    rule_score = min(1.0, rule_score)

    stable_rule = 1 - rule_score
    critical_rule = rule_score

    stable_final = (0.6 * stable_knn) + (0.4 * stable_rule)
    critical_final = (0.6 * critical_knn) + (0.4 * critical_rule)

    total = stable_final + critical_final
    stable_pct = (stable_final / total) * 100
    critical_pct = (critical_final / total) * 100

    if critical_pct > stable_pct:
        return 1, "critical", critical_pct, stable_pct, critical_pct
    else:
        return 0, "stable", stable_pct, stable_pct, critical_pct


# ===============================
# MAIN MENU
# ===============================
def main():
    print("\n===============================")
    print(" HEARSKINERGY AI MODEL ")
    print("===============================")
    print("1 - Train Model")
    print("2 - Manual Prediction")

    choice = input("Select option: ").strip()

    try:
        if choice == "1":
            train_pipeline()

        elif choice == "2":
            hr = float(input("Enter Heart Rate: "))
            spo2 = float(input("Enter Oxygen Saturation: "))
            gsr = float(input("Enter GSR Value: "))
            stress = float(input("Enter Stress Percent: "))
            relax = float(input("Enter Relaxation Percent: "))

            code, label, conf, stable, critical = predict_manual(
                hr, spo2, gsr, stress, relax
            )

            print("\n=== FINAL RESULT ===")
            print(f"Risk Code        : {code}")
            print(f"Risk Label       : {label}")
            print(f"Final Confidence : {conf:.2f}%")
            print(f"Stable Score     : {stable:.2f}%")
            print(f"Critical Score   : {critical:.2f}%")

        else:
            print("❌ Invalid option!")

    except Exception as e:
        print("\nERROR:", e)


if __name__ == "__main__":
    main()
