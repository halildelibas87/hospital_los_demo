import json
import os
import re
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# =========================
# PATH CONFIGURATION
# =========================
DATA_PATH = "data/healthcare_dataset.csv"
MODEL_DIR = "model"

LOS_MODEL_PATH = os.path.join(MODEL_DIR, "los_model.pkl")
LOS_MODEL_INFO_PATH = os.path.join(MODEL_DIR, "los_model_info.pkl")
LOS_METRICS_PATH = os.path.join(MODEL_DIR, "los_metrics.json")

BILLING_MODEL_PATH = os.path.join(MODEL_DIR, "billing_model.pkl")
BILLING_MODEL_INFO_PATH = os.path.join(MODEL_DIR, "billing_model_info.pkl")
BILLING_METRICS_PATH = os.path.join(MODEL_DIR, "billing_metrics.json")


# =========================
# FEATURE CONFIGURATION
# =========================
FEATURE_COLUMNS = [
    "Age",
    "Gender",
    "Blood Type",
    "Medical Condition",
    "Insurance Provider",
    "Admission Type",
    "Medication",
    "Test Results",
]

NUMERIC_FEATURES = ["Age"]

CATEGORICAL_FEATURES = [
    "Gender",
    "Blood Type",
    "Medical Condition",
    "Insurance Provider",
    "Admission Type",
    "Medication",
    "Test Results",
]


# =========================
# DATA LOADING & CLEANING
# =========================
def clean_text(value):
    """Standardize text values safely."""
    if pd.isna(value):
        return np.nan
    value = str(value).strip()
    value = re.sub(r"\s+", " ", value)
    return value.title()


def load_data(path):
    """Load raw CSV and validate expected columns."""
    df = pd.read_csv(path)
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

    missing_columns = [col for col in expected_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Eksik sütunlar var: {missing_columns}")

    return df


def preprocess_dataframe(df):
    """
    Clean raw dataframe and create model-ready target columns.
    This function is shared infrastructure for future LOS + Billing models.
    """
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

    # Create target for current project
    df["length_of_stay"] = (
        df["Discharge Date"] - df["Date of Admission"]
    ).dt.days

    # Keep only valid rows
    df = df.dropna(
        subset=[
            "Age",
            "Date of Admission",
            "Discharge Date",
            "length_of_stay",
        ]
    )

    df = df[df["length_of_stay"] >= 0].copy()

    return df


# =========================
# MODEL BUILDING
# =========================
def build_preprocessor():
    """Build shared preprocessing pipeline."""
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
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )

    return preprocessor


def get_candidate_models():
    """
    Candidate regression models to compare.
    We keep the list compact and interpretable for now.
    """
    return {
        "linear_regression": LinearRegression(),
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
        ),
    }


def build_pipeline_with_model(regressor):
    """Build a full preprocessing + model pipeline."""
    preprocessor = build_preprocessor()

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", regressor),
        ]
    )

    return model


# =========================
# EVALUATION
# =========================
def evaluate_regression(y_true, y_pred):
    """Return standard regression metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return {
        "mae": round(float(mae), 4),
        "rmse": round(float(rmse), 4),
        "r2": round(float(r2), 4),
    }

def calculate_overfit_gap(train_metrics, test_metrics):
    """
    Simple overfitting indicator based on R² gap.
    Higher gap means the model fits train much better than test.
    """
    return round(train_metrics["r2"] - test_metrics["r2"], 4)

def evaluate_baseline(y_true, baseline_value):
    """Compare model against simple mean baseline."""
    baseline_predictions = np.full(len(y_true), baseline_value)
    return evaluate_regression(y_true, baseline_predictions)


# =========================
# TRAINING FUNCTIONS
# =========================
def train_length_of_stay_model(df):
    """
    Train and compare multiple candidate models for LOS.
    Select the best model primarily by test MAE, while also inspecting overfit gap.
    """
    X = df[FEATURE_COLUMNS].copy()
    y = df["length_of_stay"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    baseline_value = y_train.mean()
    baseline_metrics = evaluate_baseline(y_test, baseline_value)

    candidate_models = get_candidate_models()
    model_results = {}

    best_model_name = None
    best_model_pipeline = None
    best_test_mae = float("inf")

    for model_name, regressor in candidate_models.items():
        pipeline = build_pipeline_with_model(regressor)
        pipeline.fit(X_train, y_train)

        train_predictions = pipeline.predict(X_train)
        test_predictions = pipeline.predict(X_test)

        train_metrics = evaluate_regression(y_train, train_predictions)
        test_metrics = evaluate_regression(y_test, test_predictions)
        overfit_gap = calculate_overfit_gap(train_metrics, test_metrics)

        model_results[model_name] = {
            "train": train_metrics,
            "test": test_metrics,
            "overfit_gap_r2": overfit_gap,
        }

        if test_metrics["mae"] < best_test_mae:
            best_test_mae = test_metrics["mae"]
            best_model_name = model_name
            best_model_pipeline = pipeline

    metrics = {
        "baseline_test": baseline_metrics,
        "all_models": model_results,
        "selected_model_name": best_model_name,
        "selected_model_train": model_results[best_model_name]["train"],
        "selected_model_test": model_results[best_model_name]["test"],
        "selected_model_overfit_gap_r2": model_results[best_model_name]["overfit_gap_r2"],
        "target_mean_train": round(float(baseline_value), 4),
        "n_rows": int(len(df)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }

    model_info = {
        "problem_type": "regression",
        "target_column": "length_of_stay",
        "target_description": "Patient length of stay in days",
        "feature_columns": FEATURE_COLUMNS,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "short_stay_threshold": 3,
        "medium_stay_threshold": 7,
        "selected_model_name": best_model_name,
    }

    return best_model_pipeline, model_info, metrics

def train_billing_amount_model(df):
    """
    Train and compare multiple candidate models for Billing Amount.
    Select the best model primarily by test MAE, while also inspecting overfit gap.
    """
    billing_df = df.dropna(subset=["Billing Amount"]).copy()

    X = billing_df[FEATURE_COLUMNS].copy()
    y = billing_df["Billing Amount"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    baseline_value = y_train.mean()
    baseline_metrics = evaluate_baseline(y_test, baseline_value)

    candidate_models = get_candidate_models()
    model_results = {}

    best_model_name = None
    best_model_pipeline = None
    best_test_mae = float("inf")

    for model_name, regressor in candidate_models.items():
        pipeline = build_pipeline_with_model(regressor)
        pipeline.fit(X_train, y_train)

        train_predictions = pipeline.predict(X_train)
        test_predictions = pipeline.predict(X_test)

        train_metrics = evaluate_regression(y_train, train_predictions)
        test_metrics = evaluate_regression(y_test, test_predictions)
        overfit_gap = calculate_overfit_gap(train_metrics, test_metrics)

        model_results[model_name] = {
            "train": train_metrics,
            "test": test_metrics,
            "overfit_gap_r2": overfit_gap,
        }

        if test_metrics["mae"] < best_test_mae:
            best_test_mae = test_metrics["mae"]
            best_model_name = model_name
            best_model_pipeline = pipeline

    metrics = {
        "baseline_test": baseline_metrics,
        "all_models": model_results,
        "selected_model_name": best_model_name,
        "selected_model_train": model_results[best_model_name]["train"],
        "selected_model_test": model_results[best_model_name]["test"],
        "selected_model_overfit_gap_r2": model_results[best_model_name]["overfit_gap_r2"],
        "target_mean_train": round(float(baseline_value), 4),
        "n_rows": int(len(billing_df)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }

    model_info = {
        "problem_type": "regression",
        "target_column": "Billing Amount",
        "target_description": "Estimated hospital billing amount",
        "feature_columns": FEATURE_COLUMNS,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "selected_model_name": best_model_name,
    }

    return best_model_pipeline, model_info, metrics


# =========================
# SAVE FUNCTIONS
# =========================
def save_los_artifacts(model, model_info, metrics):
    """Save LOS model outputs to disk."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(model, LOS_MODEL_PATH)
    joblib.dump(model_info, LOS_MODEL_INFO_PATH)

    with open(LOS_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)


def save_billing_artifacts(model, model_info, metrics):
    """Save Billing model outputs to disk."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(model, BILLING_MODEL_PATH)
    joblib.dump(model_info, BILLING_MODEL_INFO_PATH)

    with open(BILLING_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)


# =========================
# MAIN
# =========================
def main():
    print("Veri yükleniyor...")
    df = load_data(DATA_PATH)

    print("Veri temizleniyor ve target oluşturuluyor...")
    df = preprocess_dataframe(df)

    print(f"Temizlenmiş veri boyutu: {df.shape}")

    print("Length of Stay modeli eğitiliyor...")
    los_model, los_model_info, los_metrics = train_length_of_stay_model(df)
    save_los_artifacts(los_model, los_model_info, los_metrics)

    print("\nLOS eğitim tamamlandı.")
    print(f"\nLOS seçilen model: {los_metrics['selected_model_name']}")

    print("\nLOS Selected Model Train Metrics:")
    print(json.dumps(los_metrics["selected_model_train"], indent=4))

    print("\nLOS Selected Model Test Metrics:")
    print(json.dumps(los_metrics["selected_model_test"], indent=4))

    print("\nLOS Selected Model Overfit Gap (R²):")
    print(los_metrics["selected_model_overfit_gap_r2"])

    print("\nLOS Baseline Test Metrics:")
    print(json.dumps(los_metrics["baseline_test"], indent=4))

    print("\nLOS All Model Results:")
    print(json.dumps(los_metrics["all_models"], indent=4))

    print(f"\nLOS model kaydedildi: {LOS_MODEL_PATH}")
    print(f"LOS model bilgisi kaydedildi: {LOS_MODEL_INFO_PATH}")
    print(f"LOS metrikleri kaydedildi: {LOS_METRICS_PATH}")

    print("\nBilling Amount modeli eğitiliyor...")
    billing_model, billing_model_info, billing_metrics = train_billing_amount_model(df)
    save_billing_artifacts(billing_model, billing_model_info, billing_metrics)

    print("\nBilling eğitim tamamlandı.")
    print(f"\nBilling seçilen model: {billing_metrics['selected_model_name']}")

    print("\nBilling Selected Model Train Metrics:")
    print(json.dumps(billing_metrics["selected_model_train"], indent=4))

    print("\nBilling Selected Model Test Metrics:")
    print(json.dumps(billing_metrics["selected_model_test"], indent=4))

    print("\nBilling Selected Model Overfit Gap (R²):")
    print(billing_metrics["selected_model_overfit_gap_r2"])

    print("\nBilling Baseline Test Metrics:")
    print(json.dumps(billing_metrics["baseline_test"], indent=4))

    print("\nBilling All Model Results:")
    print(json.dumps(billing_metrics["all_models"], indent=4))

    print(f"\nBilling model kaydedildi: {BILLING_MODEL_PATH}")
    print(f"Billing model bilgisi kaydedildi: {BILLING_MODEL_INFO_PATH}")
    print(f"Billing metrikleri kaydedildi: {BILLING_METRICS_PATH}")


if __name__ == "__main__":
    main()
