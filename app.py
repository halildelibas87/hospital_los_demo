import joblib
import pandas as pd
import streamlit as st


LOS_MODEL_PATH = "model/los_model.pkl"
LOS_MODEL_INFO_PATH = "model/los_model_info.pkl"

BILLING_MODEL_PATH = "model/billing_model.pkl"
BILLING_MODEL_INFO_PATH = "model/billing_model_info.pkl"


@st.cache_resource
def load_model_assets():
    los_model = joblib.load(LOS_MODEL_PATH)
    los_model_info = joblib.load(LOS_MODEL_INFO_PATH)

    billing_model = joblib.load(BILLING_MODEL_PATH)
    billing_model_info = joblib.load(BILLING_MODEL_INFO_PATH)

    return los_model, los_model_info, billing_model, billing_model_info


def get_stay_band(pred_days, short_threshold=3, medium_threshold=7):
    if pred_days <= short_threshold:
        return "Short Stay", "Expected to stay for a short period."
    elif pred_days <= medium_threshold:
        return "Medium Stay", "Expected to stay for a moderate period."
    return "Long Stay", "Expected to require a longer hospital stay."


def build_input_dataframe(**kwargs):
    return pd.DataFrame([kwargs])


def main():
    st.set_page_config(
        page_title="Patient Stay & Billing Estimator",
        page_icon="🏥",
        layout="centered",
    )

    st.title("🏥 Patient Stay & Billing Estimator")

    try:
        los_model, los_model_info, billing_model, billing_model_info = load_model_assets()
    except Exception as e:
        st.error(f"Model yüklenemedi: {e}")
        st.stop()

    st.caption(f"Kullanılan model: {los_model_info.get('selected_model_name')}")

    st.subheader("Patient Information")

    # ⚡ Feature list’i modelden al
    features = los_model_info["feature_columns"]

    with st.form("prediction_form"):

        age = st.number_input("Age", min_value=0, max_value=120, value=45)

        gender = st.text_input("Gender", "Male")
        blood_type = st.text_input("Blood Type", "A+")
        medical_condition = st.text_input("Medical Condition", "Diabetes")
        insurance_provider = st.text_input("Insurance Provider", "Aetna")
        admission_type = st.text_input("Admission Type", "Emergency")
        medication = st.text_input("Medication", "Aspirin")
        test_results = st.text_input("Test Results", "Normal")

        submitted = st.form_submit_button("Estimate")

    if submitted:

        input_df = build_input_dataframe(
            Age=age,
            Gender=gender,
            **{
                "Blood Type": blood_type,
                "Medical Condition": medical_condition,
                "Insurance Provider": insurance_provider,
                "Admission Type": admission_type,
                "Medication": medication,
                "Test Results": test_results,
            }
        )

        with st.spinner("Calculating..."):
            los_prediction = los_model.predict(input_df)[0]
            billing_prediction = billing_model.predict(input_df)[0]

        los_prediction = max(0, round(float(los_prediction), 1))
        billing_prediction = max(0, round(float(billing_prediction), 2))

        short_threshold = los_model_info.get("short_stay_threshold", 3)
        medium_threshold = los_model_info.get("medium_stay_threshold", 7)

        stay_band, explanation = get_stay_band(
            los_prediction,
            short_threshold,
            medium_threshold
        )

        st.success(f"LOS: {los_prediction} days")
        st.success(f"Billing: ${billing_prediction:,.2f}")

        st.info(f"{stay_band} → {explanation}")

        st.subheader("Input Data")
        st.dataframe(input_df)


if __name__ == "__main__":
    main()
