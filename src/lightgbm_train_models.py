import lightgbm as lgb
import numpy as np
import mlflow
import mlflow.lightgbm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

def train_lightgbm_ovr(df, feature_columns, target_column='antibiotic'):
    print("\n🤖 Training LightGBM One-vs-Rest models with MLflow...")
    
    antibiotics = df[target_column].unique()
    models = {}
    performance_metrics = {}

    X = df[feature_columns].fillna(df[feature_columns].median())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Antibiotic_LightGBM_OVR")

    for antibiotic in antibiotics:
        y = (df[target_column] == antibiotic).astype(int)
        
        if y.sum() < 10:
            print(f"⚠️ Skipping {antibiotic}: Not enough samples")
            continue

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )

        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'seed': 42,
            'verbose': -1
        }

        with mlflow.start_run(run_name=f"LGBM_{antibiotic}"):
            train_data = lgb.Dataset(X_train, label=y_train)
            valid_data = lgb.Dataset(X_test, label=y_test)

            model = lgb.train(
                params,
                train_data,
                valid_sets=[valid_data],
                num_boost_round=300,
                callbacks=[lgb.early_stopping(20)]
            )

            y_pred = model.predict(X_test)
            y_pred_binary = (y_pred > 0.5).astype(int)
            accuracy = accuracy_score(y_test, y_pred_binary)
            report = classification_report(y_test, y_pred_binary)

            print(f"✅ {antibiotic}: Accuracy={accuracy:.3f}")

            mlflow.log_params(params)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_text(report, f"classification_report_{antibiotic}.txt")
            mlflow.lightgbm.log_model(model, artifact_path=f"LGBM_model_{antibiotic}")

            models[antibiotic] = {'model': model, 'accuracy': accuracy}
            performance_metrics[antibiotic] = {'accuracy': accuracy, 'report': report}

    return models, scaler, performance_metrics
