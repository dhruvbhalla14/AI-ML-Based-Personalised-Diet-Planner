import streamlit as st
import joblib
import json
import io
from datetime import datetime
from openai import OpenAI
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

from extractor import extract_text
from Bertgpt import (
    clean_and_segment,
    extract_entities,
    classify_intents,
    build_structured_intent,
    generate_diet_guidelines
)

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="AI Diet Planner", page_icon="🥗", layout="wide")

# ---------------------------------------------------------
# PROFESSIONAL WHITE UI
# ---------------------------------------------------------
st.markdown("""
<style>
.main {
    background-color: #f7f9fc;
}

/* Header */
.header {
    background: linear-gradient(90deg,#4CAF50,#2E7D32);
    padding: 25px;
    border-radius: 14px;
    color: white;
    text-align: center;
    margin-bottom: 25px;
}

/* Cards */
.card {
    background: white;
    padding: 20px;
    border-radius: 14px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.08);
    margin-bottom: 18px;
}

/* Stats */
.stat {
    background: white;
    padding: 18px;
    border-radius: 12px;
    text-align: center;
    box-shadow: 0 2px 10px rgba(0,0,0,0.06);
}

.stat h2 {
    color: #2E7D32;
    margin: 0;
}

/* Day boxes */
.daybox {
    background: #ffffff;
    padding: 16px;
    border-radius: 12px;
    margin-bottom: 12px;
    border-left: 5px solid #4CAF50;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}

/* Buttons */
.stButton>button {
    background-color:#4CAF50;
    color:white;
    border-radius:10px;
    height:42px;
    font-weight:600;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------
st.markdown("""
<div class="header">
<h1>🥗 AI Diet Planner</h1>
<p>Upload your medical report and generate a personalized weekly diet plan</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# LOAD OPENAI + MODEL
# ---------------------------------------------------------
@st.cache_resource
def load_openai():
    return OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

@st.cache_resource
def load_model():
    return joblib.load("best_model.pkl")

client = load_openai()
model = load_model()

# ---------------------------------------------------------
# PDF CREATOR
# ---------------------------------------------------------
def create_pdf(text):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()
    story = []

    for line in text.split("\n"):
        story.append(Paragraph(line, styles["BodyText"]))
        story.append(Spacer(1, 6))

    doc.build(story)
    buffer.seek(0)
    return buffer


# ---------------------------------------------------------
# DIET PARSER
# ---------------------------------------------------------
def parse_days(plan):
    days = []
    current = None

    for line in plan.split("\n"):
        if "day" in line.lower():
            if current:
                days.append(current)
            current = {"title": line, "text": ""}
        elif current:
            current["text"] += line + "\n"

    if current:
        days.append(current)

    return days


# ---------------------------------------------------------
# DIET GENERATOR
# ---------------------------------------------------------
def generate_plan(structured, prediction):

    prompt = f"""
Create a clear 7-day Indian diet plan.

Condition: {structured}
Risk score: {prediction}

For each day include:
Breakfast
Lunch
Dinner
"""

    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return res.choices[0].message.content


# =========================================================
# MAIN UI
# =========================================================

# Upload section
with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "📄 Upload Medical Report (PDF / CSV / TXT)",
        type=["pdf", "csv", "txt"]
    )

    generate_btn = st.button("🚀 Generate Diet Plan")

    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------
# RUN PIPELINE ONLY WHEN BUTTON CLICKED
# ---------------------------------------------------------
if uploaded and generate_btn:

    with st.spinner("Analyzing report and generating diet..."):

        text, numeric_data = extract_text(uploaded)

        # Stats
        col1, col2, col3 = st.columns(3)

        col1.markdown(
            f"<div class='stat'><h2>{len(text)}</h2><p>Characters</p></div>",
            unsafe_allow_html=True
        )

        sentences = clean_and_segment(text)
        entities = extract_entities(sentences)
        intents = classify_intents(sentences)

        structured = build_structured_intent(entities, intents)
        guidelines = generate_diet_guidelines(structured)

        col2.markdown(
            f"<div class='stat'><h2>{len(entities)}</h2><p>Entities</p></div>",
            unsafe_allow_html=True
        )

        if numeric_data:
            prediction = model.predict([list(numeric_data.values())])[0]
        else:
            prediction = "N/A"

        col3.markdown(
            f"<div class='stat'><h2>{prediction}</h2><p>Risk Score</p></div>",
            unsafe_allow_html=True
        )

        plan = generate_plan(guidelines, prediction)

    st.divider()

    # ---------------------------------------------------------
    # SHOW DIET IN PROFESSIONAL BOXES
    # ---------------------------------------------------------
    st.subheader("🥗 Weekly Diet Plan")

    for day in parse_days(plan):
        st.markdown(
            f"<div class='daybox'><h4>{day['title']}</h4><pre>{day['text']}</pre></div>",
            unsafe_allow_html=True
        )

    # ---------------------------------------------------------
    # DOWNLOAD SECTION
    # ---------------------------------------------------------
    st.subheader("⬇️ Download Report")

    col1, col2 = st.columns(2)

    json_data = {
        "generated_date": str(datetime.now()),
        "prediction": str(prediction),
        "diet_plan": plan
    }

    col1.download_button(
        "📄 Download JSON",
        json.dumps(json_data, indent=2),
        file_name="diet_plan.json"
    )

    pdf_buffer = create_pdf(plan)

    col2.download_button(
        "📑 Download PDF",
        pdf_buffer,
        file_name="diet_plan.pdf"
    )

elif uploaded and not generate_btn:
    st.info("Click **Generate Diet Plan** to start analysis.")

else:
    st.info("Upload a medical report to begin.")
