import json
import os
import re
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATA_PATH = "data/healthcare_dataset.csv"
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "los_model.pkl")
MODEL_INFO_PATH = os.path.join(MODEL_DIR, "model_info.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")


def clean_text(value):
    """Normalize text fields safely."""
    if pd.isna(value):
        return np.nan
    value = str(value).strip()
    value = re.sub(r"\s+", " ", value)
    return value.title()


def load_data(path):
    df = pd.read_csv(path)

    # Standardize column names just in case
    df.columns = [col.strip() for col in df.columns]

    expected_columns = [
        "Name",
        "Age",
        "Gender",
        "Blood Type",
        "Medical Condition",
        "Date of Admission",
        "Doctor",
        "Hospital",
        "Insurance Provider",
        "Billing Amount",
        "Room Number",
        "Admission Type",
        "Discharge Date",
        "Medication",
        "Test Results",
    ]

    missing_cols = [col for col in expected_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV içinde eksik sütunlar var: {missing_cols}")

    return df


def preprocess_dataframe(df):
    df = df.copy()

    text_columns = [
        "Name",
        "Gender",
        "Blood Type",
        "Medical Condition",
        "Doctor",
        "Hospital",
        "Insurance Provider",
        "Admission Type",
        "Medication",
        "Test Results",
    ]

    for col in text_columns:
        df[col] = df[col].apply(clean_text)

    df["Date of Admission"] = pd.to_datetime(df["Date of Admission"], errors="coerce")
    df["Discharge Date"] = pd.to_datetime(df["Discharge Date"], errors="coerce")

    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df["Billing Amount"] = pd.to_numeric(df["Billing Amount"], errors="coerce")
    df["Room Number"] = pd.to_numeric(df["Room Number"], errors="coerce")

    # Target variable
    df["length_of_stay"] = (df["Discharge Date"] - df["Date of Admission"]).dt.days

    # Remove invalid rows
    df = df.dropna(subset=["Age", "Date of Admission", "Discharge Date", "length_of_stay"])
    df = df[df["length_of_stay"] >= 0].copy()

    return df


def build_model():
    numeric_features = ["Age"]
    categorical_features = [
        "Gender",
        "Blood Type",
        "Medical Condition",
        "Insurance Provider",
        "Admission Type",
        "Medication",
        "Test Results",
    ]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=300,
                    max_depth=None,
                    min_samples_split=4,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    return model, numeric_features, categorical_features


def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return {
        "mae": round(float(mae), 4),
        "rmse": round(float(rmse), 4),
        "r2": round(float(r2), 4),
    }


def baseline_metrics(y_true, baseline_value):
    baseline_pred = np.full(shape=len(y_true), fill_value=baseline_value)
    return evaluate_model(y_true, baseline_pred)


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Veri yükleniyor...")
    df = load_data(DATA_PATH)
    df = preprocess_dataframe(df)

    print(f"Temizlenmiş veri boyutu: {df.shape}")

    feature_columns = [
        "Age",
        "Gender",
        "Blood Type",
        "Medical Condition",
        "Insurance Provider",
        "Admission Type",
        "Medication",
        "Test Results",
    ]
    target_column = "length_of_stay"

    X = df[feature_columns].copy()
    y = df[target_column].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    baseline_value = y_train.mean()
    baseline_result = baseline_metrics(y_test, baseline_value)

    model, numeric_features, categorical_features = build_model()

    print("Model eğitiliyor...")
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_metrics = evaluate_model(y_train, train_pred)
    test_metrics = evaluate_model(y_test, test_pred)

    metrics = {
        "baseline_test": baseline_result,
        "train": train_metrics,
        "test": test_metrics,
        "target_mean_train": round(float(baseline_value), 4),
        "n_rows": int(len(df)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }

    model_info = {
        "feature_columns": feature_columns,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "target_column": target_column,
        "target_description": "Patient length of stay in days",
        "short_stay_threshold": 3,
        "medium_stay_threshold": 7,
    }

    joblib.dump(model, MODEL_PATH)
    joblib.dump(model_info, MODEL_INFO_PATH)

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)

    print("\nEğitim tamamlandı.")
    print("\nBaseline Test Metrics:")
    print(json.dumps(baseline_result, indent=4))

    print("\nTrain Metrics:")
    print(json.dumps(train_metrics, indent=4))

    print("\nTest Metrics:")
    print(json.dumps(test_metrics, indent=4))

    print(f"\nModel kaydedildi: {MODEL_PATH}")
    print(f"Model bilgisi kaydedildi: {MODEL_INFO_PATH}")
    print(f"Metrikler kaydedildi: {METRICS_PATH}")


if __name__ == "__main__":
    main()
