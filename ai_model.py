import joblib
import pandas as pd
import os

model = None

def load_risk_model():
    """Loads the trained KNN model once when server starts."""
    global model

    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "scaler.pkl")

    try:
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            print(f"✅ KNN Model loaded successfully from: {model_path}")
        else:
            print(f"❌ Model file NOT FOUND at: {model_path}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")

def predict_risk(
    heart_rate,
    oxygen_saturation,
    gsr_value,
    stress_percent,
    stress_level,
    relaxation_percent
):
    """Returns: Normal | Abnormal | High Risk"""
    global model

    if model is None:
        print("⚠️ Model not loaded. Returning safe default.")
        return "Normal"

    try:
        # ✅ EXACT FEATURE ORDER used during training
        input_data = pd.DataFrame(
            [[
                heart_rate,
                oxygen_saturation,
                gsr_value,
                stress_percent,
                stress_level,
                relaxation_percent
            ]],
            columns=[
                "Heart Rate",
                "Oxygen Saturation",
                "GSR Value",
                "Stress Percent",
                "Stress Level",
                "Relaxation Percent"
            ]
        )

        prediction = model.predict(input_data)[0]

        print("-" * 45)
        print("🧠 KNN PREDICTION REPORT")
        print(input_data)
        print(f"Raw Output → {prediction}")
        print("-" * 45)

        risk_map = {
            0: "Normal",
            1: "Abnormal",
            2: "High Risk",
            "0": "Normal",
            "1": "Abnormal",
            "2": "High Risk"
        }

        return risk_map.get(prediction, "Normal")

    except Exception as e:
        print(f"❌ Prediction Error: {e}")
        return "Error"

# 🚨 Load model on startup
load_risk_model()

# Debug confirmation
if model is None:
    print("❌ CRITICAL FAILURE: Model is STILL None!")
else:
    print("✅ SUCCESS: KNN Model is READY")

print("-" * 50)
print(f"📂 Working Dir: {os.getcwd()}")
print(f"📄 Script Dir: {os.path.dirname(os.path.abspath(__file__))}")
