import streamlit as st
import joblib
import json
from datetime import datetime
from openai import OpenAI
import io
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

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------
st.set_page_config(page_title="AI Diet Planner", page_icon="🥗", layout="wide")

# ------------------------------------------------
# MODERN CLEAN UI
# ------------------------------------------------
st.markdown("""
<style>
.main {background:#0e1117;}
.header {
    background:linear-gradient(135deg,#6a11cb,#2575fc);
    padding:18px;border-radius:12px;color:white;text-align:center;
}
.box {
    background:#1c212c;
    padding:14px;
    border-radius:12px;
    margin-bottom:12px;
}
.stat {
    text-align:center;
    background:#161a23;
    padding:10px;
    border-radius:10px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header">
<h1>🥗 AI Diet Planner</h1>
<p>Upload medical report → Get weekly personalized diet</p>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------
# LOAD OPENAI
# ------------------------------------------------
@st.cache_resource
def load_openai():
    return OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

client = load_openai()

# ------------------------------------------------
# LOAD ML MODEL
# ------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("best_model.pkl")

model = load_model()

# ------------------------------------------------
# PDF GENERATOR
# ------------------------------------------------
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


# ------------------------------------------------
# PARSER → boxes
# ------------------------------------------------
def parse_plan(plan):
    days = []
    current = None

    for line in plan.split("\n"):
        l = line.lower().strip()

        if "day" in l:
            if current:
                days.append(current)
            current = {"title": line, "content": ""}

        elif current:
            current["content"] += line + "\n"

    if current:
        days.append(current)

    return days


# ------------------------------------------------
# DIET GENERATOR
# ------------------------------------------------
def generate_plan(structured, prediction):

    prompt = f"""
Create a clear 7 day Indian diet plan.

Condition: {structured}
Health Score: {prediction}

Format:
Day 1
Breakfast:
Lunch:
Dinner:
"""

    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}]
    )

    return res.choices[0].message.content


# ------------------------------------------------
# UPLOAD
# ------------------------------------------------
uploaded = st.file_uploader("Upload PDF / CSV / TXT", type=["pdf","csv","txt"])

# ------------------------------------------------
# MAIN FLOW
# ------------------------------------------------
if uploaded:

    with st.spinner("Analyzing..."):

        text, numeric_data = extract_text(uploaded)

        col1, col2, col3 = st.columns(3)

        col1.markdown(f"<div class='stat'><h3>{len(text)}</h3>Characters</div>", unsafe_allow_html=True)

        sentences = clean_and_segment(text)
        entities = extract_entities(sentences)
        intents = classify_intents(sentences)

        structured = build_structured_intent(entities, intents)
        guidelines = generate_diet_guidelines(structured)

        col2.markdown(f"<div class='stat'><h3>{len(entities)}</h3>Entities</div>", unsafe_allow_html=True)

        if numeric_data:
            prediction = model.predict([list(numeric_data.values())])[0]
        else:
            prediction = "N/A"

        col3.markdown(f"<div class='stat'><h3>{prediction}</h3>Risk</div>", unsafe_allow_html=True)

        plan = generate_plan(guidelines, prediction)

    st.divider()

    # ------------------------------------------------
    # SHOW DIET IN BOXES
    # ------------------------------------------------
    st.subheader("🥗 Weekly Diet Plan")

    for day in parse_plan(plan):
        st.markdown(
            f"<div class='box'><h4>{day['title']}</h4><pre>{day['content']}</pre></div>",
            unsafe_allow_html=True
        )

    # ------------------------------------------------
    # DOWNLOAD BUTTONS
    # ------------------------------------------------
    st.subheader("⬇️ Download Report")

    col1, col2 = st.columns(2)

    # JSON
    json_data = {
        "date": str(datetime.now()),
        "prediction": str(prediction),
        "guidelines": guidelines,
        "diet_plan": plan
    }

    col1.download_button(
        "Download JSON",
        json.dumps(json_data, indent=2),
        file_name="diet_plan.json"
    )

    # PDF
    pdf_file = create_pdf(plan)

    col2.download_button(
        "Download PDF",
        pdf_file,
        file_name="diet_plan.pdf"
    )

else:
    st.info("Upload your medical report to start.")
