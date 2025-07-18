import pandas as pd

def predict_antibiotic(patient_data, model_package):
    models = model_package['models']
    scaler = model_package['scaler']
    feature_columns = model_package['feature_columns']

    patient_features = pd.DataFrame([patient_data])[feature_columns].fillna(0)
    patient_scaled = scaler.transform(patient_features)

    predictions = {}
    for antibiotic, model_info in models.items():
        model = model_info['model']
        prob = model.predict(patient_scaled)[0]
        predictions[antibiotic] = prob

    ranked = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
    return ranked
