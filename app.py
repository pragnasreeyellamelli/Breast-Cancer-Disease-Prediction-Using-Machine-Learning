import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import io

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListItem, ListFlowable
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="AI Breast Cancer Clinical System",
    page_icon="🩺",
    layout="wide"
)

# ---------------- MEDICAL UI STYLE ---------------- #
st.markdown("""
<style>
.main {
    background-color: #f4f9ff;
}
h1, h2, h3 {
    color: #0a3d62;
}
.stButton>button {
    background-color: #0a3d62;
    color: white;
    height: 3em;
    width: 100%;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ---------------- #
@st.cache_resource
def load_model():
    return joblib.load("cancer_model.pkl")

model = load_model()

# ---------------- STAGE CLASSIFICATION ---------------- #
def get_stage(probability):
    if probability < 0.25:
        return "Stage 1"
    elif probability < 0.50:
        return "Stage 2"
    elif probability < 0.75:
        return "Stage 3"
    else:
        return "Stage 4"

def stage_details(stage):

    details = {
        "Stage 1": {
            "Symptoms": ["Small painless lump", "No lymph node spread", "Minor breast texture change"],
            "Diet": ["Leafy greens", "Berries", "High protein foods", "Whole grains"],
            "Precautions": ["Regular checkups", "Maintain healthy weight", "Avoid alcohol & smoking"]
        },
        "Stage 2": {
            "Symptoms": ["Growing lump", "Swollen lymph nodes", "Skin dimpling"],
            "Diet": ["Broccoli", "Carrots", "Omega-3 foods", "Vitamin C rich fruits"],
            "Precautions": ["Chemotherapy if advised", "Healthy lifestyle", "Light exercise"]
        },
        "Stage 3": {
            "Symptoms": ["Large tumor", "Redness & swelling", "Nipple inversion"],
            "Diet": ["Iron rich foods", "Soft protein meals", "Hydration"],
            "Precautions": ["Aggressive treatment", "Avoid infections", "Proper rest"]
        },
        "Stage 4": {
            "Symptoms": ["Bone pain", "Extreme fatigue", "Weight loss", "Breathing difficulty"],
            "Diet": ["High calorie diet", "Smoothies & soups", "Calcium rich foods"],
            "Precautions": ["Palliative care", "Pain management", "Emotional support"]
        }
    }

    return details[stage]

# ---------------- PDF GENERATOR (IN-MEMORY) ---------------- #
def create_pdf(name, age, gender, stage, probability):

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("MULTI-SPECIALITY HOSPITAL", styles["Heading1"]))
    elements.append(Paragraph("Breast Cancer Clinical Assessment Report", styles["Heading3"]))
    elements.append(Spacer(1, 20))

    elements.append(Paragraph("Patient Information", styles["Heading2"]))
    elements.append(Paragraph(f"Name: {name}", styles["Normal"]))
    elements.append(Paragraph(f"Age: {age}", styles["Normal"]))
    elements.append(Paragraph(f"Gender: {gender}", styles["Normal"]))
    elements.append(Spacer(1, 15))

    elements.append(Paragraph("Cancer Risk Analysis", styles["Heading2"]))
    elements.append(Paragraph(f"Predicted Stage: {stage}", styles["Normal"]))
    elements.append(Paragraph(f"Risk Probability: {probability:.2%}", styles["Normal"]))
    elements.append(Spacer(1, 15))

    info = stage_details(stage)

    elements.append(Paragraph("Clinical Symptoms", styles["Heading2"]))
    elements.append(ListFlowable([ListItem(Paragraph(i, styles["Normal"])) for i in info["Symptoms"]]))
    elements.append(Spacer(1, 15))

    elements.append(Paragraph("Dietary Recommendations", styles["Heading2"]))
    elements.append(ListFlowable([ListItem(Paragraph(i, styles["Normal"])) for i in info["Diet"]]))
    elements.append(Spacer(1, 15))

    elements.append(Paragraph("Medical Precautions", styles["Heading2"]))
    elements.append(ListFlowable([ListItem(Paragraph(i, styles["Normal"])) for i in info["Precautions"]]))
    elements.append(Spacer(1, 20))

    elements.append(Paragraph("Doctor's Clinical Note:", styles["Heading2"]))
    elements.append(Paragraph(
        "This AI-based prediction is for preliminary screening only. "
        "Further diagnostic tests like biopsy and imaging are required "
        "for medical confirmation.",
        styles["Normal"]
    ))

    elements.append(Spacer(1, 30))
    elements.append(Paragraph("Authorized Signature: ____________________", styles["Normal"]))
    elements.append(Paragraph("Date: ____________________", styles["Normal"]))

    doc.build(elements)
    buffer.seek(0)

    return buffer

# ---------------- UI ---------------- #
st.markdown("## 🏥 Breast Cancer Disease Prediction System")
st.markdown("### Advanced Tumor Stage Prediction & Risk Assessment")
st.markdown("---")

# Sidebar Input
st.sidebar.header("Patient Details")
name = st.sidebar.text_input("Patient Name")
age = st.sidebar.number_input("Age", 1, 120, 30)
gender = st.sidebar.selectbox("Gender", ["Female", "Male"])

st.sidebar.header("Tumor Measurements")

features = model.feature_names_in_
input_data = {}

for feature in features:
    input_data[feature] = st.sidebar.number_input(
        feature.replace("_", " ").title(),
        value=0.0
    )

input_df = pd.DataFrame([input_data])

col1, col2 = st.columns(2)

with col1:
    st.subheader("Input Summary")
    st.dataframe(input_df)

with col2:
    st.subheader("Prediction Result")

    if st.button("Predict Cancer Stage") and name.strip() != "":

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]
        stage = get_stage(probability)

        st.progress(float(probability))

        if stage == "Stage 1":
            st.success(f"🟢 {stage} - Early Detection")
        elif stage == "Stage 2":
            st.warning(f"🟡 {stage} - Moderate Risk")
        elif stage == "Stage 3":
            st.error(f"🟠 {stage} - Advanced Local")
        else:
            st.error(f"🔴 {stage} - Metastatic Risk")

        st.write(f"Risk Probability: {probability:.2%}")

        # Pie Chart
        st.subheader("Probability Distribution")
        labels = ["No Cancer", "Cancer"]
        values = [1 - probability, probability]

        fig, ax = plt.subplots()
        ax.pie(values, labels=labels, autopct="%1.1f%%")
        st.pyplot(fig)

        # Feature Importance
        if hasattr(model, "feature_importances_"):
            st.subheader("Top Feature Importance")
            importance_df = pd.DataFrame({
                "Feature": model.feature_names_in_,
                "Importance": model.feature_importances_
            }).sort_values("Importance", ascending=False).head(10)

            st.bar_chart(importance_df.set_index("Feature"))

        info = stage_details(stage)

        st.subheader("Symptoms")
        st.write(info["Symptoms"])

        st.subheader("Diet Recommendation")
        st.write(info["Diet"])

        st.subheader("Precautions")
        st.write(info["Precautions"])

        # Generate PDF
        pdf_buffer = create_pdf(name, age, gender, stage, probability)

        st.download_button(
            label="📥 Download Full Medical Report",
            data=pdf_buffer,
            file_name=f"{name}_Cancer_Report.pdf",
            mime="application/pdf"
        )

    elif name.strip() == "":
        st.warning("Please enter patient name before prediction.")

# Footer
st.markdown("---")
st.caption("Final Year Project | Pragnasree Yellamelli | Machine Learning • Streamlit • Python")