import joblib
import pandas as pd
import streamlit as st


LOS_MODEL_PATH = "model/los_model.pkl"
LOS_MODEL_INFO_PATH = "model/los_model_info.pkl"

BILLING_MODEL_PATH = "model/billing_model.pkl"
BILLING_MODEL_INFO_PATH = "model/billing_model_info.pkl"


@st.cache_resource
def load_model_assets():
    """
    Load both trained models and their metadata.
    """
    los_model = joblib.load(LOS_MODEL_PATH)
    los_model_info = joblib.load(LOS_MODEL_INFO_PATH)

    billing_model = joblib.load(BILLING_MODEL_PATH)
    billing_model_info = joblib.load(BILLING_MODEL_INFO_PATH)

    return los_model, los_model_info, billing_model, billing_model_info


def get_stay_band(pred_days, short_threshold=3, medium_threshold=7):
    """
    Convert LOS prediction into a simple category.
    """
    if pred_days <= short_threshold:
        return "Short Stay", "Expected to stay for a short period."
    elif pred_days <= medium_threshold:
        return "Medium Stay", "Expected to stay for a moderate period."
    return "Long Stay", "Expected to require a longer hospital stay."


def build_input_dataframe(
    age,
    gender,
    blood_type,
    medical_condition,
    insurance_provider,
    admission_type,
    medication,
    test_results,
):
    """
    Create a single-row dataframe for model inference.
    """
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

    return input_df


def main():
    st.set_page_config(
        page_title="Patient Stay & Billing Estimator",
        page_icon="🏥",
        layout="centered",
    )

    st.title("🏥 Patient Stay & Billing Estimator")
    st.caption(
        "This application estimates hospital length of stay and billing amount using patient admission information."
    )

    st.info(
        "This tool is for demo and decision-support purposes only. It does not replace medical or financial judgment."
    )

    try:
        los_model, los_model_info, billing_model, billing_model_info = load_model_assets()
    except Exception as e:
        st.error(f"Model dosyaları yüklenemedi: {e}")
        st.stop()

    st.subheader("Patient Information")

    with st.form("prediction_form"):
        age = st.number_input("Age", min_value=0, max_value=120, value=45, step=1)

        gender = st.selectbox("Gender", ["Male", "Female"])
        blood_type = st.selectbox(
            "Blood Type",
            ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"],
        )
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

        submitted = st.form_submit_button("Estimate")

    if submitted:
        input_df = build_input_dataframe(
            age=age,
            gender=gender,
            blood_type=blood_type,
            medical_condition=medical_condition,
            insurance_provider=insurance_provider,
            admission_type=admission_type,
            medication=medication,
            test_results=test_results,
        )

        with st.spinner("Calculating estimates..."):
            los_prediction = los_model.predict(input_df)[0]
            billing_prediction = billing_model.predict(input_df)[0]

        los_prediction = max(0, round(float(los_prediction), 1))
        billing_prediction = max(0, round(float(billing_prediction), 2))

        short_threshold = los_model_info.get("short_stay_threshold", 3)
        medium_threshold = los_model_info.get("medium_stay_threshold", 7)

        stay_band, stay_explanation = get_stay_band(
            los_prediction,
            short_threshold=short_threshold,
            medium_threshold=medium_threshold,
        )

        st.success(f"Estimated Length of Stay: {los_prediction} days")
        st.success(f"Estimated Billing Amount: ${billing_prediction:,.2f}")

        if stay_band == "Short Stay":
            st.info(f"Stay Category: {stay_band} | {stay_explanation}")
        elif stay_band == "Medium Stay":
            st.warning(f"Stay Category: {stay_band} | {stay_explanation}")
        else:
            st.error(f"Stay Category: {stay_band} | {stay_explanation}")

        st.subheader("Submitted Information")
        st.dataframe(input_df, use_container_width=True)


if __name__ == "__main__":
    main()
