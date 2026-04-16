import json
import joblib
import pandas as pd
import streamlit as st


MODEL_PATH = "model/los_model.pkl"
MODEL_INFO_PATH = "model/model_info.pkl"
METRICS_PATH = "model/metrics.json"


@st.cache_resource
def load_model_assets():
    model = joblib.load(MODEL_PATH)
    model_info = joblib.load(MODEL_INFO_PATH)

    try:
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            metrics = json.load(f)
    except FileNotFoundError:
        metrics = None

    return model, model_info, metrics


def get_stay_band(pred_days, short_threshold=3, medium_threshold=7):
    if pred_days <= short_threshold:
        return "Short Stay", "Expected to stay for a short period."
    elif pred_days <= medium_threshold:
        return "Medium Stay", "Expected to stay for a moderate period."
    return "Long Stay", "Expected to require a longer hospital stay."


def main():
    st.set_page_config(
        page_title="Patient Length of Stay Prediction",
        page_icon="🏥",
        layout="centered",
    )

    st.title("🏥 Patient Length of Stay Prediction")
    st.caption("This demo estimates expected hospital stay duration based on patient admission details.")
    st.info("This tool is for decision support and demo purposes only. It does not replace clinical judgment.")

    try:
        model, model_info, metrics = load_model_assets()
    except Exception as e:
        st.error(f"Model yüklenemedi: {e}")
        st.stop()

    with st.expander("Model Performance"):
        if metrics:
            col1, col2, col3 = st.columns(3)
            col1.metric("Test MAE", metrics["test"]["mae"])
            col2.metric("Test RMSE", metrics["test"]["rmse"])
            col3.metric("Test R²", metrics["test"]["r2"])

            st.write("Baseline test metrics:")
            st.json(metrics["baseline_test"])
        else:
            st.write("Henüz metrik bulunamadı.")

    st.subheader("Patient Information")

    with st.form("prediction_form"):
        age = st.number_input("Age", min_value=0, max_value=120, value=45, step=1)

        gender = st.selectbox("Gender", ["Male", "Female"])
        blood_type = st.selectbox("Blood Type", ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"])
        medical_condition = st.selectbox(
            "Medical Condition",
            ["Arthritis", "Asthma", "Cancer", "Diabetes", "Hypertension", "Obesity"],
        )
        insurance_provider = st.selectbox(
            "Insurance Provider",
            ["Aetna", "Blue Cross", "Cigna", "Medicare", "UnitedHealthcare"],
        )
        admission_type = st.selectbox(
            "Admission Type",
            ["Elective", "Emergency", "Urgent"],
        )
        medication = st.selectbox(
            "Medication",
            ["Aspirin", "Ibuprofen", "Lipitor", "Paracetamol", "Penicillin"],
        )
        test_results = st.selectbox(
            "Test Results",
            ["Normal", "Abnormal", "Inconclusive"],
        )

        submitted = st.form_submit_button("Predict Length of Stay")

    if submitted:
        input_df = pd.DataFrame(
            [
                {
                    "Age": age,
                    "Gender": gender,
                    "Blood Type": blood_type,
                    "Medical Condition": medical_condition,
                    "Insurance Provider": insurance_provider,
                    "Admission Type": admission_type,
                    "Medication": medication,
                    "Test Results": test_results,
                }
            ]
        )

        prediction = model.predict(input_df)[0]
        prediction = max(0, round(float(prediction), 1))

        short_threshold = model_info.get("short_stay_threshold", 3)
        medium_threshold = model_info.get("medium_stay_threshold", 7)
        band, explanation = get_stay_band(prediction, short_threshold, medium_threshold)

        st.success(f"Estimated Length of Stay: {prediction} days")

        if band == "Short Stay":
            st.info(f"Category: {band}\n\n{explanation}")
        elif band == "Medium Stay":
            st.warning(f"Category: {band}\n\n{explanation}")
        else:
            st.error(f"Category: {band}\n\n{explanation}")

        st.subheader("Input Summary")
        st.dataframe(input_df, use_container_width=True)

        st.subheader("Operational Note")
        st.write(
            f"This patient is predicted to stay approximately **{prediction} days**. "
            f"This estimate can support bed planning, staffing, and patient flow management."
        )


if __name__ == "__main__":
    main()
