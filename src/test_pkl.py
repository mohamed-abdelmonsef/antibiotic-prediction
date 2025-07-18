import joblib
import pandas as pd
from custom_encoders import MultiColumnLabelEncoder  # Needed for pickle to load

# ✅ Load pipeline
pipeline = joblib.load("preprocessing_pipeline.pkl")

# ✅ Example raw data
example_data = pd.DataFrame([{
    'culture_description': 'Blood',
    'antibiotic': 'Amoxicillin/Clavulanic Acid',
    'age': '20-30',
    'gender': 'M',
    'median_heartrate': 95,
    'median_resprate': 22,
    'median_temp': 101.2,
    'median_sysbp': 130,
    'median_diasbp': 85,
    'median_wbc': 16.5,
    'median_hgb': 11.0,
    'median_plt': 200,
    'median_na': 138,
    'median_hco3': 22,
    'median_bun': 28,
    'median_cr': 1.8
}])

# ✅ Apply preprocessing
processed_data = pipeline.transform(example_data)
print("Processed Features:\n", processed_data)
