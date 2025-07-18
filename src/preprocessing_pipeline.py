import pandas as pd
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
    # ✅ Load your real dataset
    df = pd.read_csv("complete_microbiology_cultures_data.csv")

    # ✅ Select relevant columns
    df_subset = df[['culture_description', 'age', 'gender',
                    'median_heartrate', 'median_resprate', 'median_temp',
                    'median_sysbp', 'median_diasbp',
                    'median_wbc', 'median_hgb', 'median_plt',
                    'median_na', 'median_hco3', 'median_bun', 'median_cr']]

    # ✅ Create and fit pipeline
    pipeline = create_preprocessing_pipeline()
    pipeline.fit(df_subset)

    # ✅ Save the fitted pipeline
    joblib.dump(pipeline, 'preprocessing_pipeline.pkl')
    print("✅ Preprocessing pipeline fitted and saved.")
