import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Load model package
MODEL_PATH = "models/lightgbm_ovr_models.pkl"
model_package = joblib.load(MODEL_PATH)
models = model_package['models']
scaler = model_package['scaler']
feature_columns = model_package['feature_columns']
label_encoders = model_package['label_encoders']

# FastAPI app
app = FastAPI(title="LightGBM Antibiotic Prediction API")

# Input schema for raw CSV-like input
class PatientData(BaseModel):
    culture_description: str
    age: str
    gender: str
    median_heartrate: float
    median_resprate: float
    median_temp: float
    median_sysbp: float
    median_diasbp: float
    median_wbc: float
    median_hgb: float
    median_plt: float
    median_na: float
    median_hco3: float
    median_bun: float
    median_cr: float

@app.get("/")
def root():
    return {"message": "Welcome to the LightGBM Antibiotic Prediction API"}

@app.post("/predict")
def predict(patient: PatientData):
    try:
        # Convert Pydantic model to dict
        patient_dict = patient.dict()

        # ✅ Normalize input fields
        #patient_dict["culture_description"] = patient_dict["culture_description"].strip() # e.g., BLOOD → Blood
        patient_dict["age"] = patient_dict["age"].replace("years", "").strip()  # e.g., 45-54 years → 45-54
        patient_dict["gender"] = patient_dict["gender"].strip().upper()  # e.g., m → M

        # ✅ Apply label encoding using stored encoders
        patient_dict["culture_description_encoded"] = label_encoders["culture_description"].transform([patient_dict["culture_description"]])[0]
        patient_dict["age_encoded"] = label_encoders["age"].transform([patient_dict["age"]])[0]
        patient_dict["gender_encoded"] = label_encoders["gender"].transform([patient_dict["gender"]])[0]

        # Create DataFrame for model
        patient_df = pd.DataFrame([{
            'median_heartrate': patient_dict['median_heartrate'],
            'median_resprate': patient_dict['median_resprate'],
            'median_temp': patient_dict['median_temp'],
            'median_sysbp': patient_dict['median_sysbp'],
            'median_diasbp': patient_dict['median_diasbp'],
            'median_wbc': patient_dict['median_wbc'],
            'median_hgb': patient_dict['median_hgb'],
            'median_plt': patient_dict['median_plt'],
            'median_na': patient_dict['median_na'],
            'median_hco3': patient_dict['median_hco3'],
            'median_bun': patient_dict['median_bun'],
            'median_cr': patient_dict['median_cr'],
            'culture_description_encoded': patient_dict['culture_description_encoded'],
            'age_encoded': patient_dict['age_encoded'],
            'gender_encoded': patient_dict['gender_encoded']
        }])

        # Scale features
        patient_scaled = scaler.transform(patient_df)

        # Predictions
        predictions = {}
        for antibiotic, model_info in models.items():
            model = model_info['model']
            probs = model.predict(patient_scaled)
            prob = float(probs[0]) if probs.ndim == 1 else float(np.max(probs[0]))
            predictions[antibiotic] = prob

        ranked = sorted(predictions.items(), key=lambda x: x[1], reverse=True)

        return {
            "status": "success",
            "top_5_recommendations": [
                {"antibiotic": antibiotic, "probability": round(prob, 4)}
                for antibiotic, prob in ranked[:5]
            ]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")
