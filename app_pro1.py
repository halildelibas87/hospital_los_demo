import os
import joblib
import pandas as pd
import streamlit as st


LOS_MODEL_PATH = "model/los_model.pkl"
LOS_MODEL_INFO_PATH = "model/los_model_info.pkl"
BILLING_MODEL_PATH = "model/billing_model.pkl"
BILLING_MODEL_INFO_PATH = "model/billing_model_info.pkl"
LOGO_PATH = "assets/MedIntel_Logo.jpg"


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
    if pred_days <= medium_threshold:
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
    return pd.DataFrame(
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


def get_default_index(options, fallback_value=None):
    if not options:
        return 0
    if fallback_value in options:
        return options.index(fallback_value)
    return 0


def inject_custom_css():
    st.markdown(
        """
        <style>
        html { scroll-behavior: smooth; }

        header[data-testid="stHeader"],
        div[data-testid="stToolbar"] {
            display: none;
        }

        #MainMenu, footer {
            visibility: hidden;
        }

        .stApp {
            background: linear-gradient(180deg, #f8fbfd 0%, #eef7f8 45%, #f8fbfd 100%);
            color: #0f172a;
        }

        .block-container {
            padding-top: 0.8rem;
            padding-bottom: 2rem;
            max-width: 1200px;
        }

        .anchor-offset {
            display: block;
            position: relative;
            top: -110px;
            visibility: hidden;
        }

        .topbar-shell {
            position: sticky;
            top: 0;
            z-index: 9999;
            background: rgba(255,255,255,0.95);
            border: 1px solid #dbe7ee;
            border-radius: 18px;
            backdrop-filter: blur(12px);
            box-shadow: 0 10px 32px rgba(15, 23, 42, 0.08);
            padding: 0.7rem 1rem;
            margin-bottom: 1.2rem;
        }

        .brand-title {
            font-size: 1.2rem;
            font-weight: 800;
            color: #0f172a;
            letter-spacing: 0.4px;
            line-height: 1.1;
            margin: 0;
        }

        .brand-tagline {
            font-size: 0.9rem;
            color: #64748b;
            font-weight: 500;
            margin-top: 0.18rem;
        }

        .nav-link {
            text-align: center;
            font-size: 0.95rem;
            font-weight: 600;
            margin-top: 1.1rem;
        }

        .nav-link a {
            color: #475569 !important;
            text-decoration: none !important;
        }

        .nav-link a:hover {
            color: #0f766e !important;
        }

        .hero {
            background: linear-gradient(135deg, #0f172a 0%, #164e63 55%, #14b8a6 100%);
            padding: 3rem 2rem;
            border-radius: 28px;
            color: white;
            box-shadow: 0 18px 45px rgba(15, 23, 42, 0.18);
            margin: 1rem 0 2rem 0;
        }

        .hero h1 {
            font-size: 3rem;
            margin-bottom: 0.35rem;
            color: white;
        }

        .hero h2 {
            font-size: 1.05rem;
            margin-top: 0;
            margin-bottom: 1rem;
            color: #cceff0;
            font-weight: 500;
        }

        .hero p {
            font-size: 1.05rem;
            color: #e2f3f3;
            max-width: 780px;
            line-height: 1.7;
        }

        .pill-row {
            display: flex;
            gap: 0.6rem;
            flex-wrap: wrap;
            margin-top: 1.2rem;
        }

        .pill {
            background: rgba(255,255,255,0.14);
            border: 1px solid rgba(255,255,255,0.2);
            color: white;
            padding: 0.55rem 0.9rem;
            border-radius: 999px;
            font-size: 0.92rem;
        }

        .section-title {
            font-size: 1.9rem;
            font-weight: 700;
            color: #0f172a;
            margin-top: 1.2rem;
            margin-bottom: 0.4rem;
        }

        .section-subtitle {
            color: #475569;
            margin-bottom: 1.2rem;
            line-height: 1.7;
        }

        .card, .contact-box {
            background: white;
            border: 1px solid #dde8ef;
            border-radius: 22px;
            padding: 1.25rem;
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.05);
            height: 100%;
        }

        .card h3 {
            margin-top: 0;
            color: #0f172a;
            font-size: 1.15rem;
        }

        .card p {
            color: #475569;
            font-size: 0.95rem;
            line-height: 1.65;
        }

        .metric-card {
            background: linear-gradient(180deg, #ffffff 0%, #f6fbfc 100%);
            border: 1px solid #dcecf0;
            border-radius: 22px;
            padding: 1.25rem;
            box-shadow: 0 12px 30px rgba(15, 23, 42, 0.06);
            text-align: center;
            height: 100%;
        }

        .metric-label {
            color: #475569;
            font-size: 0.95rem;
            margin-bottom: 0.5rem;
        }

        .metric-value {
            color: #0f172a;
            font-size: 2rem;
            font-weight: 800;
        }

        .metric-caption {
            color: #0f766e;
            font-size: 0.9rem;
            margin-top: 0.35rem;
        }

        .footer {
            margin-top: 2rem;
            padding: 1.2rem;
            text-align: center;
            color: #64748b;
            font-size: 0.9rem;
        }

        /* Form container */
        div[data-testid="stForm"] {
            background: white;
            border: 1px solid #dde8ef;
            border-radius: 22px;
            padding: 1.2rem 1.2rem 0.8rem 1.2rem;
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.05);
        }

        div[data-testid="stForm"] h1,
        div[data-testid="stForm"] h2,
        div[data-testid="stForm"] h3,
        div[data-testid="stForm"] h4,
        div[data-testid="stForm"] h5,
        div[data-testid="stForm"] h6 {
            color: #0f172a !important;
        }

        div[data-testid="stForm"] [data-testid="stWidgetLabel"] {
            color: #0f172a !important;
            font-weight: 600 !important;
        }

        /* Inputs */
        div[data-testid="stForm"] div[data-baseweb="input"] > div,
        div[data-testid="stForm"] div[data-baseweb="select"] > div {
            background-color: #ffffff !important;
            border: 1px solid #cbd5e1 !important;
            border-radius: 10px !important;
        }

        div[data-testid="stForm"] div[data-baseweb="input"] input {
            color: #0f172a !important;
            background: #ffffff !important;
            -webkit-text-fill-color: #0f172a !important;
        }

        div[data-testid="stForm"] div[data-baseweb="select"] span,
        div[data-testid="stForm"] div[data-baseweb="select"] div {
            color: #0f172a !important;
            -webkit-text-fill-color: #0f172a !important;
        }

        div[role="listbox"] {
            background: #ffffff !important;
        }

        div[role="option"] {
            color: #0f172a !important;
            background: #ffffff !important;
        }

        div[role="option"]:hover {
            background: #eef7f8 !important;
        }

        /* Number input +/- buttons */
        div[data-testid="stNumberInput"] button {
            background: #14b8a6 !important;
            color: #ffffff !important;
            border: none !important;
        }

        div[data-testid="stNumberInput"] button:hover {
            background: #0f766e !important;
            color: #ffffff !important;
        }

        div[data-testid="stNumberInput"] button svg {
            fill: #ffffff !important;
            color: #ffffff !important;
        }

        /* Form submit button only */
        div[data-testid="stFormSubmitButton"] button {
            background: linear-gradient(135deg, #0f766e 0%, #14b8a6 100%) !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 12px !important;
            font-weight: 700 !important;
            padding: 0.65rem 1.15rem !important;
            box-shadow: 0 8px 18px rgba(20, 184, 166, 0.22);
        }

        div[data-testid="stFormSubmitButton"] button p {
            color: #ffffff !important;
            font-weight: 700 !important;
        }

        div[data-testid="stFormSubmitButton"] button:hover {
            background: linear-gradient(135deg, #115e59 0%, #0f766e 100%) !important;
            color: #ffffff !important;
        }

        /* Portal switch */
        div[role="radiogroup"] {
            display: flex;
            gap: 14px;
            margin: 0.8rem 0 1.4rem 0;
        }

        div[role="radiogroup"] label {
            min-width: 260px;
            background: #ffffff !important;
            border: 1px solid #cbd5e1 !important;
            border-radius: 999px !important;
            padding: 0.75rem 1.25rem !important;
            box-shadow: 0 6px 18px rgba(15, 23, 42, 0.05);
        }

        div[role="radiogroup"] label p {
            color: #0f172a !important;
            font-weight: 700 !important;
            font-size: 0.98rem !important;
        }

        div[role="radiogroup"] label:has(input:checked) {
            background: linear-gradient(135deg, #0f766e 0%, #14b8a6 100%) !important;
            border-color: #0f766e !important;
        }

        div[role="radiogroup"] label:has(input:checked) p {
            color: #ffffff !important;
        }

        div[role="radiogroup"] input[type="radio"] {
            opacity: 0 !important;
            position: absolute !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

def render_topbar():
    st.markdown('<div class="topbar-shell">', unsafe_allow_html=True)

    col_brand, col_home, col_about, col_solutions, col_demo, col_contact = st.columns(
        [3.3, 0.7, 0.8, 1.0, 0.7, 0.9]
    )

    with col_brand:
        brand_left, brand_right = st.columns([0.42, 2.58], vertical_alignment="center")

        with brand_left:
            if os.path.exists(LOGO_PATH):
                st.image(LOGO_PATH, width=95)  # yaklaşık 2.5 cm
            else:
                st.empty()

        with brand_right:
            st.markdown('<div class="brand-title">MEDINTEL</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="brand-tagline">Medical Intelligence Platform</div>',
                unsafe_allow_html=True,
            )

    with col_home:
        st.markdown('<div class="nav-link"><a href="#home-section">Home</a></div>', unsafe_allow_html=True)
    with col_about:
        st.markdown('<div class="nav-link"><a href="#about-section">About</a></div>', unsafe_allow_html=True)
    with col_solutions:
        st.markdown('<div class="nav-link"><a href="#solutions-section">Solutions</a></div>', unsafe_allow_html=True)
    with col_demo:
        st.markdown('<div class="nav-link"><a href="#demo-section">Demo</a></div>', unsafe_allow_html=True)
    with col_contact:
        st.markdown('<div class="nav-link"><a href="#contact-section">Contact</a></div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


def render_hero():
    st.markdown('<span id="home-section" class="anchor-offset"></span>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="hero">
            <h1>MEDINTEL</h1>
            <h2>Medical Intelligence Platform</h2>
            <p>
                MEDINTEL is a healthcare intelligence platform designed for hospitals and patients.
                It helps users estimate patient length of stay and projected billing amount through
                a clean, modern, and institution-ready digital experience.
            </p>
            <div class="pill-row">
                <div class="pill">Hospital Operations</div>
                <div class="pill">Patient Estimates</div>
                <div class="pill">Length of Stay Prediction</div>
                <div class="pill">Billing Forecasting</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_about_section():
    st.markdown('<span id="about-section" class="anchor-offset"></span>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">About Us</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">MEDINTEL is built to support both healthcare institutions and individual patients. Hospitals can use the platform for operational planning, while patients can use it to better understand likely hospitalization duration and estimated billing outcomes before or during care journeys.</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
            <div class="card">
                <h3>For Hospitals</h3>
                <p>Support bed planning, capacity visibility, and operational decision-making with structured prediction outputs.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="card">
                <h3>For Patients</h3>
                <p>Provide individuals with a transparent digital experience to estimate expected stay duration and projected billing levels.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
            <div class="card">
                <h3>Corporate Healthcare Design</h3>
                <p>Designed with a modern visual language that feels credible, professional, and aligned with digital health platforms.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_solutions_section():
    st.markdown('<span id="solutions-section" class="anchor-offset"></span>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Solutions</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">A dual-purpose healthcare platform combining institutional utility with patient-facing accessibility.</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
            <div class="card">
                <h3>Length of Stay Prediction</h3>
                <p>Estimate probable hospitalization duration to support scheduling, planning, and expectation management.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="card">
                <h3>Billing Estimation</h3>
                <p>Generate projected billing visibility to support financial planning and improve communication transparency.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
            <div class="card">
                <h3>Accessible Intelligence</h3>
                <p>Bring machine learning outputs into a clear product interface suitable for both professionals and non-technical users.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_trust_section(los_model_info, billing_model_info):
    st.markdown('<div class="section-title">Why MEDINTEL</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Healthcare-focused positioning, stable model selection, and a cleaner user experience in one platform.</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    cards = [
        ("Dual Audience", "Built for both hospital operations teams and patient-facing estimate experiences."),
        ("Operational Clarity", "Transforms admission details into practical stay duration and cost insights."),
        (
            "Stable Model Choice",
            f"LOS model: {los_model_info.get('selected_model_name', 'N/A')} | Billing model: {billing_model_info.get('selected_model_name', 'N/A')}",
        ),
        ("Scalable Product Design", "Structured as a digital health product experience that can expand into a broader platform."),
    ]

    for col, (title, text) in zip([col1, col2, col3, col4], cards):
        with col:
            st.markdown(
                f"""
                <div class="card">
                    <h3>{title}</h3>
                    <p>{text}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_demo_section(los_model, los_model_info, billing_model):
    st.markdown('<span id="demo-section" class="anchor-offset"></span>', unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">Patient Insight Dashboard</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-subtitle">MEDINTEL provides separate estimation experiences for patients and hospital teams while using the same intelligence engine.</div>',
        unsafe_allow_html=True,
    )

    category_options = los_model_info.get("category_options", {})

    gender_options = category_options.get("Gender", ["Male", "Female"])
    blood_type_options = category_options.get(
        "Blood Type",
        ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"],
    )
    medical_condition_options = category_options.get("Medical Condition", ["Diabetes"])
    insurance_provider_options = category_options.get("Insurance Provider", ["Aetna"])
    admission_type_options = category_options.get("Admission Type", ["Emergency"])
    medication_options = category_options.get("Medication", ["Aspirin"])
    test_results_options = category_options.get("Test Results", ["Normal"])

    def prediction_form(form_key, user_type):
        if user_type == "patient":
            title = "Provide Your Details for Personalized Estimates"
            button_text = "View My Estimate"
            loading_text = "Preparing your personalized estimate..."
        else:
            title = "Enter Admission Details for Operational Planning"
            button_text = "Generate Hospital Estimate"
            loading_text = "Generating operational estimate..."

        with st.form(form_key):
            st.markdown(
                f'<h3 style="color:#0f172a; margin-bottom: 1rem;">{title}</h3>',
                unsafe_allow_html=True,
            )

            c1, c2, c3, c4 = st.columns(4)

            with c1:
                age = st.number_input(
                    "Age",
                    min_value=0,
                    max_value=120,
                    value=45,
                    step=1,
                    key=f"{form_key}_age",
                )

            with c2:
                gender = st.selectbox(
                    "Gender",
                    gender_options,
                    index=get_default_index(gender_options, "Male"),
                    key=f"{form_key}_gender",
                )

            with c3:
                blood_type = st.selectbox(
                    "Blood Type",
                    blood_type_options,
                    index=get_default_index(blood_type_options, "A+"),
                    key=f"{form_key}_blood_type",
                )

            with c4:
                medical_condition = st.selectbox(
                    "Medical Condition",
                    medical_condition_options,
                    index=get_default_index(medical_condition_options, None),
                    key=f"{form_key}_medical_condition",
                )

            c5, c6, c7, c8 = st.columns(4)

            with c5:
                insurance_provider = st.selectbox(
                    "Insurance Provider",
                    insurance_provider_options,
                    index=get_default_index(insurance_provider_options, None),
                    key=f"{form_key}_insurance_provider",
                )

            with c6:
                admission_type = st.selectbox(
                    "Admission Type",
                    admission_type_options,
                    index=get_default_index(admission_type_options, "Emergency"),
                    key=f"{form_key}_admission_type",
                )

            with c7:
                medication = st.selectbox(
                    "Medication",
                    medication_options,
                    index=get_default_index(medication_options, None),
                    key=f"{form_key}_medication",
                )

            with c8:
                test_results = st.selectbox(
                    "Test Results",
                    test_results_options,
                    index=get_default_index(test_results_options, "Normal"),
                    key=f"{form_key}_test_results",
                )

            submitted = st.form_submit_button(button_text)

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

            with st.spinner(loading_text):
                los_prediction = los_model.predict(input_df)[0]
                billing_prediction = billing_model.predict(input_df)[0]

            los_prediction = max(0, round(float(los_prediction), 1))
            billing_prediction = max(0, round(float(billing_prediction), 2))


            st.markdown("<br>", unsafe_allow_html=True)

            result_left, result_right = st.columns([1.2, 1], gap="large")

            with result_left:
                st.markdown(
                    f"""
                    <div class="metric-card" style="
                        background: linear-gradient(135deg, #0f766e 0%, #14b8a6 100%);
                        color: white;
                        border: none;
                        min-height: 180px;
                        display: flex;
                        flex-direction: column;
                        justify-content: center;
                    ">
                        <div class="metric-label" style="color:#e0f7f5;">
                            Estimated Billing Amount
                        </div>
                        <div class="metric-value" style="color:white; font-size:2.8rem;">
                            ${billing_prediction:,.0f}
                        </div>
                        <div class="metric-caption" style="color:#d1faf5;">
                            projected total
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with result_right:
                st.markdown(
                    f"""
                    <div class="metric-card" style="
                        min-height: 180px;
                        display: flex;
                        flex-direction: column;
                        justify-content: center;
                    ">
                        <div class="metric-label">
                            Estimated Length of Stay
                        </div>
                        <div class="metric-value">
                            {los_prediction}
                        </div>
                        <div class="metric-caption">
                            days
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            if user_type == "patient":
                st.markdown(
                    """
                    <div class="card" style="margin-top: 1rem;">
                        <h3>Patient Note</h3>
                        <p>This estimate is designed to help patients understand possible stay duration and projected billing. It does not replace medical or financial advice.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    """
                    <div class="card" style="margin-top: 1rem;">
                        <h3>Hospital Planning Note</h3>
                        <p>This estimate can support bed planning, resource allocation, and operational forecasting. It should be used as a decision-support output, not as a final clinical decision.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with st.expander("View Submitted Information"):
                st.dataframe(input_df, use_container_width=True)

        else:
            if user_type == "patient":
                st.markdown(
                    """
                    <div class="card">
                        <h3>Patient Estimate Panel</h3>
                        <p>Complete the form above to view your estimated hospital stay duration, projected billing amount, and stay category.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    """
                    <div class="card">
                        <h3>Hospital Operations Panel</h3>
                        <p>Complete the form above to generate estimates that can support hospital planning, capacity management, and operational visibility.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # 👇 BURASI ÖNEMLİ: aynı indent
    selected_portal = st.radio(
        label="Select Portal",
        options=["Patient Estimate Portal", "Hospital Operations Panel"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if selected_portal == "Patient Estimate Portal":
        prediction_form("patient_form", "patient")
    else:
        prediction_form("hospital_form", "hospital")




def render_contact_section():
    st.markdown('<span id="contact-section" class="anchor-offset"></span>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Contact</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">A medical platform should look reachable, even if the real world still insists on emails and meetings.</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            <div class="contact-box">
                <h3>Get in Touch</h3>
                <p><strong>Email:</strong> info@medintelplatform.com</p>
                <p><strong>Location:</strong> Ankara, Türkiye</p>
                <p><strong>Audience:</strong> Hospitals, healthcare managers, digital health teams, and patient-facing services</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="contact-box">
                <h3>About This Demo</h3>
                <p>This demo shows how a machine learning prediction engine can be presented as a more realistic corporate healthcare web experience.</p>
                <p>It is intended for demo and decision-support purposes only and does not replace clinical or financial judgment.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def main():
    st.set_page_config(
        page_title="MEDINTEL",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    inject_custom_css()

    try:
        los_model, los_model_info, billing_model, billing_model_info = load_model_assets()
    except Exception as e:
        st.error(f"Model dosyaları yüklenemedi: {e}")
        st.stop()

    render_topbar()
    render_hero()
    render_about_section()
    st.markdown("<br>", unsafe_allow_html=True)
    render_solutions_section()
    st.markdown("<br>", unsafe_allow_html=True)
    render_trust_section(los_model_info, billing_model_info)
    st.markdown("<br>", unsafe_allow_html=True)
    render_demo_section(los_model, los_model_info, billing_model)
    st.markdown("<br>", unsafe_allow_html=True)
    render_contact_section()

    st.markdown(
        """
        <div class="footer">
            © 2026 MEDINTEL. Medical Intelligence Platform for hospitals and patients.
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
