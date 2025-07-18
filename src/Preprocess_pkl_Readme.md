Here’s an updated **`README.md`** including the **conditions and steps that led to your output**, as well as the **commands in order**:

---

# 🧪 Antibiotic Prediction Project

This project predicts the most suitable antibiotic using patient clinical data. It supports **LightGBM (One-vs-Rest)** and **Random Forest** models, with a consistent **preprocessing pipeline** (`preprocessing_pipeline.pkl`) for categorical encoding, scaling, and imputation.

---

## ✅ **Features**

* **Machine Learning Models**:

  * LightGBM (OVR)
  * Random Forest (OVR)
* **Preprocessing Pipeline**:

  * Handles categorical and numeric features.
  * Uses `LabelEncoder` for categorical variables.
  * Handles missing values with median imputation.
  * Scales numeric features with `StandardScaler`.
* **MLflow Integration**:

  * Tracks experiments for model comparison.
* **FastAPI Deployment**:

  * REST API to predict antibiotics from patient data.
* **Containerization**:

  * Dockerfile for deployment on cloud platforms.

---

## 📂 **Project Structure**

```
.
├── app.py                        # FastAPI app for inference
├── preprocessing_pipeline.py     # Creates and saves preprocessing pipeline
├── custom_encoders.py            # Custom encoder for categorical features
├── preprocessing_pipeline.pkl     # Saved preprocessing pipeline
├── test_pkl.py                   # Test script for preprocessing pipeline
├── lightgbm_main.py              # Train & log LightGBM model
├── train_rf.py                   # Train & log RandomForest model
├── models/                       # Directory for saved models
└── mlruns/                       # MLflow experiment tracking
```

---

## ✅ **Preprocessing Pipeline**

### **`custom_encoders.py`**

```python
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder

class MultiColumnLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns
        self.encoders = {}

    def fit(self, X, y=None):
        for col in self.columns:
            self.encoders[col] = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            self.encoders[col].fit(X[[col]])
        return self

    def transform(self, X):
        X_copy = X.copy()
        for col in self.columns:
            X_copy[col] = self.encoders[col].transform(X_copy[[col]]).astype(int)
        return X_copy
```

---

### **`preprocessing_pipeline.py`**

```python
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from custom_encoders import MultiColumnLabelEncoder

def create_preprocessing_pipeline():
    categorical_columns = ['culture_description', 'age', 'gender']
    numeric_columns = [
        'median_heartrate', 'median_resprate', 'median_temp',
        'median_sysbp', 'median_diasbp',
        'median_wbc', 'median_hgb', 'median_plt',
        'median_na', 'median_hco3', 'median_bun', 'median_cr'
    ]

    cat_pipeline = Pipeline([('label_encoder', MultiColumnLabelEncoder(columns=categorical_columns))])
    num_pipeline = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])

    preprocessor = ColumnTransformer([
        ('categorical', cat_pipeline, categorical_columns),
        ('numerical', num_pipeline, numeric_columns)
    ])

    return preprocessor

if __name__ == "__main__":
    # Fit on actual dataset for real usage
    import pandas as pd
    df = pd.read_csv("complete_microbiology_cultures_data.csv")

    pipeline = create_preprocessing_pipeline()
    pipeline.fit(df[['culture_description','age','gender',
                     'median_heartrate','median_resprate','median_temp',
                     'median_sysbp','median_diasbp','median_wbc','median_hgb',
                     'median_plt','median_na','median_hco3','median_bun','median_cr']])

    joblib.dump(pipeline, 'preprocessing_pipeline.pkl')
    print("✅ Preprocessing pipeline fitted and saved as preprocessing_pipeline.pkl")
```

---

## ✅ **Test the Pipeline**

### `test_pkl.py`:

```python
import joblib
import pandas as pd
from custom_encoders import MultiColumnLabelEncoder  # Required for loading

# Load pipeline
pipeline = joblib.load("preprocessing_pipeline.pkl")

# Example raw input
example_data = pd.DataFrame([{
    'culture_description': 'Blood',
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

processed = pipeline.transform(example_data)
print("Processed Features:\n", processed)
```

---

## ✅ **Correct Way to Get Valid Output**

Run the following commands in order:

```bash
# 1. Fit and save the pipeline on actual data
python preprocessing_pipeline.py

# 2. Test pipeline with example input
python test_pkl.py
```

If fitted correctly:

* You **won't see -1** for known categories like `Blood`, `20-30`, `M`.
* Values will be **encoded and scaled properly**.

---

## ✅ Deployment Steps (Summary)

1. **Train models**:

   ```bash
   python lightgbm_main.py
   python train_rf.py
   ```
2. **Start MLflow UI**:

   ```bash
   mlflow ui --backend-store-uri file:./mlruns
   ```
3. **Run FastAPI app locally**:

   ```bash
   uvicorn app:app --reload
   ```
4. **Test with Postman or Swagger UI**.
5. **Build and push Docker image**:

   ```bash
   docker build -t <dockerhub-username>/lightgbm-api:v1 .
   docker push <dockerhub-username>/lightgbm-api:v1
   ```

---

👉 Do you want me to **include Docker + FastAPI deployment instructions in detail in this same README** and **add example Postman request with sample JSON input**?
