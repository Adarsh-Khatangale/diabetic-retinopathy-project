import os
import sqlite3
import pandas as pd
from datetime import date

import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

from src.model import RetinopathyModel


# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Diabetic Retinopathy Screening System",
    page_icon="👁️",
    layout="wide"
)
# ============================================================
# PROFESSIONAL CSS DESIGN
# ============================================================

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #f7fbff 0%, #eef6ff 100%);
    color: #17324d;
}
.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}
h1 {
    color: #075aaa !important;
    font-size: 2.5rem !important;
    font-weight: 800 !important;
    text-align: center;
}
h2 { color: #0877c9 !important; font-weight: 750 !important; }
h3 { color: #146fa8 !important; font-weight: 700 !important; }
p, label { color: #40566b; }

[data-testid="stWidgetLabel"] label {
    color: #31556f !important;
    font-weight: 600 !important;
}

div[data-baseweb="input"] {
    background: white !important;
    border: 2px solid #c7dceb !important;
    border-radius: 10px !important;
}
div[data-baseweb="input"]:focus-within {
    border-color: #1689d8 !important;
    box-shadow: 0 0 0 3px rgba(22,137,216,.12) !important;
}
div[data-baseweb="input"] input,
div[data-testid="stNumberInput"] input,
div[data-testid="stDateInput"] input {
    color: #172b3d !important;
    -webkit-text-fill-color: #172b3d !important;
    background: white !important;
    font-weight: 500 !important;
}
input::placeholder {
    color: #8a9aaa !important;
    -webkit-text-fill-color: #8a9aaa !important;
    opacity: 1 !important;
}
div[data-baseweb="select"] {
    background: white !important;
    border-radius: 10px !important;
}
div[data-baseweb="select"] * { color: #172b3d !important; }

[data-testid="stFileUploader"] {
    background: white !important;
    border: 2px dashed #9fc4df !important;
    border-radius: 14px !important;
    padding: 1rem !important;
}
[data-testid="stFileUploader"] button {
    background: #087cc1 !important;
    color: white !important;
    -webkit-text-fill-color: white !important;
    border: none !important;
    border-radius: 9px !important;
    font-weight: 600 !important;
}

.stButton > button {
    background: linear-gradient(135deg, #087cc1, #075aa6) !important;
    color: white !important;
    -webkit-text-fill-color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: .65rem 1.4rem !important;
    font-weight: 700 !important;
    box-shadow: 0 4px 10px rgba(7,90,166,.20);
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 7px 16px rgba(7,90,166,.28);
}

[data-testid="stDataFrame"] {
    border-radius: 12px !important;
    overflow: hidden !important;
    border: 1px solid #d4e4ef !important;
    box-shadow: 0 5px 16px rgba(30,80,120,.08);
}
[data-testid="stAlert"] { border-radius: 12px !important; }

.dashboard-card {
    background: white;
    border: 1px solid #d8e7f2;
    border-radius: 16px;
    padding: 18px;
    text-align: center;
    box-shadow: 0 5px 18px rgba(30,80,120,.08);
}
.dashboard-number {
    color: #075aaa;
    font-size: 28px;
    font-weight: 800;
}
.dashboard-label {
    color: #60798c;
    font-size: 14px;
    font-weight: 600;
}
.result-card, .recommendation-card {
    background: white;
    border-radius: 16px;
    padding: 20px;
    border: 1px solid #d8e7f2;
    box-shadow: 0 5px 18px rgba(30,80,120,.08);
}
.result-card { border-left: 6px solid #087cc1; }
.recommendation-card { border-left: 6px solid #f4b400; }

@media (max-width: 768px) {
    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }
    h1 { font-size: 1.9rem !important; }
    .stButton > button { width: 100% !important; }
}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# Class names
# --------------------------------------------------
CLASS_NAMES = {
    0: "No Diabetic Retinopathy",
    1: "Mild Diabetic Retinopathy",
    2: "Moderate Diabetic Retinopathy",
    3: "Severe Diabetic Retinopathy",
    4: "Proliferative Diabetic Retinopathy"
}


# --------------------------------------------------
# Recommendations
# --------------------------------------------------
RECOMMENDATIONS = {
    0: (
        "No diabetic retinopathy was detected by the screening model. "
        "Continue regular diabetes management and routine eye examinations."
    ),
    1: (
        "Mild diabetic retinopathy was detected. "
        "A routine consultation with an eye-care professional is recommended."
    ),
    2: (
        "Moderate diabetic retinopathy was detected. "
        "Please arrange an eye examination with an ophthalmologist."
    ),
    3: (
        "Severe diabetic retinopathy was detected. "
        "Prompt evaluation by an ophthalmologist is recommended."
    ),
    4: (
        "Proliferative diabetic retinopathy was detected. "
        "Prompt specialist ophthalmic evaluation is strongly recommended."
    )
}


# --------------------------------------------------
# Paths
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "best_model.pth"
)

DATABASE_PATH = os.path.join(
    BASE_DIR,
    "patient_records.db"
)


# --------------------------------------------------
# Device
# --------------------------------------------------
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# --------------------------------------------------
# Database
# --------------------------------------------------
def initialize_database():

    connection = sqlite3.connect(DATABASE_PATH)

    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS patient_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_name TEXT NOT NULL,
            age INTEGER NOT NULL,
            examination_date TEXT NOT NULL,
            diagnosis TEXT NOT NULL,
            confidence REAL NOT NULL
        )
    """)

    connection.commit()
    connection.close()


initialize_database()


# --------------------------------------------------
# Dashboard summary
# --------------------------------------------------
def get_dashboard_summary():
    connection = sqlite3.connect(DATABASE_PATH)
    cursor = connection.cursor()

    cursor.execute("SELECT COUNT(*) FROM patient_records")
    total_patients = cursor.fetchone()[0]

    cursor.execute(
        "SELECT COUNT(*) FROM patient_records "
        "WHERE diagnosis != 'No Diabetic Retinopathy'"
    )
    detected_cases = cursor.fetchone()[0]

    cursor.execute(
        "SELECT COUNT(*) FROM patient_records "
        "WHERE diagnosis = 'No Diabetic Retinopathy'"
    )
    no_dr_cases = cursor.fetchone()[0]

    connection.close()
    return total_patients, detected_cases, no_dr_cases


# --------------------------------------------------
# Load model
# --------------------------------------------------
@st.cache_resource
def load_model():

    model = RetinopathyModel(num_classes=5)

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device
        )
    )

    model = model.to(device)
    model.eval()

    return model


model = load_model()


# --------------------------------------------------
# Image preprocessing
# --------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# --------------------------------------------------
# Save patient record
# --------------------------------------------------
def save_patient_record(
    patient_name,
    age,
    examination_date,
    diagnosis,
    confidence
):

    connection = sqlite3.connect(DATABASE_PATH)

    cursor = connection.cursor()

    cursor.execute("""
        INSERT INTO patient_records
        (
            patient_name,
            age,
            examination_date,
            diagnosis,
            confidence
        )
        VALUES (?, ?, ?, ?, ?)
    """, (
        patient_name,
        age,
        examination_date,
        diagnosis,
        confidence
    ))

    connection.commit()
    connection.close()


# --------------------------------------------------
# Application title
# --------------------------------------------------
st.title("👁️ AI-Based Diabetic Retinopathy Screening System")

st.write(
    "AI-assisted retinal image screening and patient record management."
)

st.warning(
    "This system is intended for screening support only and "
    "does not replace professional medical diagnosis."
)


# --------------------------------------------------
# Dashboard
# --------------------------------------------------
total_patients, detected_cases, no_dr_cases = get_dashboard_summary()

dash1, dash2, dash3 = st.columns(3)

with dash1:
    st.markdown(
        f'<div class="dashboard-card"><div class="dashboard-number">'
        f'{total_patients}</div><div class="dashboard-label">'
        f'Total Patients</div></div>',
        unsafe_allow_html=True
    )

with dash2:
    st.markdown(
        f'<div class="dashboard-card"><div class="dashboard-number">'
        f'{detected_cases}</div><div class="dashboard-label">'
        f'DR Detected Cases</div></div>',
        unsafe_allow_html=True
    )

with dash3:
    st.markdown(
        f'<div class="dashboard-card"><div class="dashboard-number">'
        f'{no_dr_cases}</div><div class="dashboard-label">'
        f'No DR Cases</div></div>',
        unsafe_allow_html=True
    )


# --------------------------------------------------
# Patient information
# --------------------------------------------------
st.header("👤 Patient Information")

col1, col2, col3 = st.columns(3)

with col1:

    patient_name = st.text_input(
        "Patient Name"
    )

with col2:

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=30,
        step=1
    )

with col3:

    examination_date = st.date_input(
        "Examination Date",
        value=date.today()
    )


# --------------------------------------------------
# Image upload
# --------------------------------------------------
st.header("📷 Retinal Image")

uploaded_file = st.file_uploader(
    "Upload retinal fundus image",
    type=["jpg", "jpeg", "png"]
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if "prediction_done" not in st.session_state:
    st.session_state.prediction_done = False

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.image(
        image,
        caption="Uploaded Retinal Image",
        width=500
    )

    if st.button(
        "🔍 Analyze Retinal Image",
        type="primary"
    ):

        if not patient_name.strip():

            st.error(
                "Please enter the patient's name."
            )

        else:

            input_tensor = transform(image)

            input_tensor = input_tensor.unsqueeze(0)

            input_tensor = input_tensor.to(device)

            with torch.no_grad():

                outputs = model(input_tensor)

                probabilities = torch.softmax(
                    outputs,
                    dim=1
                )

                confidence, predicted_class = torch.max(
                    probabilities,
                    dim=1
                )

            predicted_class = predicted_class.item()

            confidence = confidence.item() * 100

            diagnosis = CLASS_NAMES[predicted_class]

            recommendation = RECOMMENDATIONS[predicted_class]

            # Save prediction in Streamlit session
            st.session_state.prediction_done = True
            st.session_state.patient_name = patient_name.strip()
            st.session_state.age = int(age)
            st.session_state.examination_date = str(
                examination_date
            )
            st.session_state.diagnosis = diagnosis
            st.session_state.confidence = confidence
            st.session_state.recommendation = recommendation


# --------------------------------------------------
# Display prediction
# --------------------------------------------------

if st.session_state.prediction_done:

    st.header("📊 Screening Result")

    result_col1, result_col2 = st.columns(2)

    with result_col1:
        st.markdown(
            f'<div class="result-card"><h3>🩺 Diagnosis</h3>'
            f'<strong>{st.session_state.diagnosis}</strong></div>',
            unsafe_allow_html=True
        )

    with result_col2:
        st.markdown(
            f'<div class="result-card"><h3>📊 Confidence</h3>'
            f'<strong>{st.session_state.confidence:.2f}%</strong></div>',
            unsafe_allow_html=True
        )


    # --------------------------------------------------
    # Recommendation
    # --------------------------------------------------

    st.header("💡 Recommendation")

    st.markdown(
        f'<div class="recommendation-card">'
        f'{st.session_state.recommendation}</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "Recommendation is for screening support only. "
        "Please consult a qualified healthcare professional "
        "for clinical evaluation."
    )


    # --------------------------------------------------
    # Save patient record
    # --------------------------------------------------

    if st.button("💾 Save Patient Record",key="Save_Patient_Record"):

        save_patient_record(
            st.session_state.patient_name,
            st.session_state.age,
            st.session_state.examination_date,
            st.session_state.diagnosis,
            st.session_state.confidence
        )

        st.success(
            "✅ Patient record saved successfully!"
        )
        st.rerun()


# ==========================================
# PATIENT HISTORY
# ==========================================

st.header("📁 Patient History")

# ------------------------------------------
# SEARCH PATIENT
# ------------------------------------------

st.subheader("🔎 Search Patient")

search_name = st.text_input(
    "Enter patient name",
    placeholder="Type patient name...",
    key="search_patient"
)

if st.button("🔍 Search", key="search_patient_button"):

    connection = sqlite3.connect(DATABASE_PATH)
    cursor = connection.cursor()

    if search_name.strip():
        cursor.execute(
            """
            SELECT id, patient_name, age,
                   examination_date, diagnosis, confidence
            FROM patient_records
            WHERE patient_name LIKE ?
            ORDER BY id DESC
            """,
            ("%" + search_name.strip() + "%",)
        )
    else:
        cursor.execute(
            """
            SELECT id, patient_name, age,
                   examination_date, diagnosis, confidence
            FROM patient_records
            ORDER BY id DESC
            """
        )

    records = cursor.fetchall()
    connection.close()

    if records:
        df = pd.DataFrame(
            records,
            columns=[
                "ID",
                "Patient Name",
                "Age",
                "Examination Date",
                "Diagnosis",
                "Confidence"
            ]
        )

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.warning("No patient found.")


# ------------------------------------------
# DELETE PATIENT
# ------------------------------------------

st.subheader("🗑️ Delete Patient")

delete_id = st.number_input(
    "Enter Patient ID",
    min_value=1,
    step=1,
    key="delete_patient_id"
)

if st.button("🗑️ Delete Patient", key="delete_patient_button"):

    connection = sqlite3.connect(DATABASE_PATH)
    cursor = connection.cursor()

    cursor.execute(
        "SELECT patient_name FROM patient_records WHERE id = ?",
        (delete_id,)
    )

    patient = cursor.fetchone()

    if patient:

        cursor.execute(
            "DELETE FROM patient_records WHERE id = ?",
            (delete_id,)
        )

        connection.commit()
        connection.close()

        st.success(
            f"Patient '{patient[0]}' deleted successfully."
        )

        st.rerun()

    else:
        connection.close()
        st.error("Patient ID not found.")


# ------------------------------------------
# ALL PATIENT HISTORY
# ------------------------------------------

st.subheader("📋 All Patient History")

connection = sqlite3.connect(DATABASE_PATH)
cursor = connection.cursor()

cursor.execute(
    """
    SELECT id, patient_name, age,
           examination_date, diagnosis, confidence
    FROM patient_records
    ORDER BY id DESC
    """
)

records = cursor.fetchall()
connection.close()

if records:

    df = pd.DataFrame(
        records,
        columns=[
            "ID",
            "Patient Name",
            "Age",
            "Examination Date",
            "Diagnosis",
            "Confidence"
        ]
    )

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )

else:

    st.info("No patient records have been saved yet.")