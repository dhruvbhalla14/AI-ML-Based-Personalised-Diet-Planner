import streamlit as st
import joblib
import os
import json
from datetime import datetime
from openai import OpenAI
import io

from extractor import extract_text
from Bertgpt import (
    clean_and_segment,
    extract_entities,
    classify_intents,
    build_structured_intent,
    generate_diet_guidelines
)

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="AI Diet Planner",
    page_icon="🥗",
    layout="wide"
)

# -------------------------------------------------
# CLEAN MODERN UI (lighter, compact)
# -------------------------------------------------
st.markdown("""
<style>
.main {
    background: #0f1117;
}

.block-container {
    padding-top: 1.5rem;
}

/* Header */
.header {
    background: linear-gradient(135deg,#6a11cb,#2575fc);
    padding: 1.2rem;
    border-radius: 14px;
    text-align:center;
    color:white;
    margin-bottom: 1.5rem;
}

/* File uploader smaller */
[data-testid="stFileUploader"]{
    padding:0.7rem !important;
    border-radius:12px;
    border:2px dashed #6a11cb;
}

/* Cards */
.card{
    background:#161a23;
    padding:1.2rem;
    border-radius:14px;
    margin-bottom:1rem;
}

/* Stats */
.stat{
    text-align:center;
    background:#1c212c;
    padding:1rem;
    border-radius:12px;
}

.stat h2{
    color:#6a11cb;
    margin:0;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# HEADER
# -------------------------------------------------
st.markdown("""
<div class="header">
<h1>🥗 AI Diet Planner</h1>
<p>Upload your medical report → Get personalized weekly diet</p>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------
# OPENAI (Streamlit secrets only)
# -------------------------------------------------
@st.cache_resource
def init_openai():
    return OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

client = init_openai()

# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("best_model.pkl")

model = load_model()

# -------------------------------------------------
# ROBUST PARSER (FIXED)
# -------------------------------------------------
def parse_diet_plan(plan_text):
    days = []
    current = None

    for line in plan_text.split("\n"):
        line = line.strip()
        lower = line.lower()

        if "day" in lower:
            if current:
                days.append(current)
            current = {
                "title": line,
                "breakfast": "",
                "lunch": "",
                "dinner": "",
                "other": ""
            }

        elif current:
            if "breakfast" in lower:
                current["breakfast"] += line.replace("**", "") + "\n"
            elif "lunch" in lower:
                current["lunch"] += line.replace("**", "") + "\n"
            elif "dinner" in lower:
                current["dinner"] += line.replace("**", "") + "\n"
            else:
                current["other"] += line + "\n"

    if current:
        days.append(current)

    return days


# -------------------------------------------------
# WEEK PLAN GENERATOR
# -------------------------------------------------
def generate_week_plan(structured, prediction):

    prompt = f"""
Create a 7-day Indian personalized diet plan.

Condition: {structured}
Health score: {prediction}

For each day:
Breakfast
Lunch
Dinner
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}]
    )

    return response.choices[0].message.content


# -------------------------------------------------
# FILE UPLOADER
# -------------------------------------------------
st.subheader("📤 Upload Medical Report")

uploaded_file = st.file_uploader(
    "PDF / Image / CSV / TXT",
    type=["pdf","png","jpg","jpeg","csv","txt"]
)

# -------------------------------------------------
# MAIN FLOW
# -------------------------------------------------
if uploaded_file:

    col1, col2, col3 = st.columns(3)

    with col1:
        stat_text = st.empty()
    with col2:
        stat_entities = st.empty()
    with col3:
        stat_pred = st.empty()

    with st.spinner("Processing..."):

        text, numeric_data = extract_text(uploaded_file)

        stat_text.markdown(
            f"<div class='stat'><h2>{len(text)}</h2><p>Characters</p></div>",
            unsafe_allow_html=True
        )

        # NLP
        sentences = clean_and_segment(text)
        entities = extract_entities(sentences)
        intents = classify_intents(sentences)

        structured = build_structured_intent(entities, intents)
        guidelines = generate_diet_guidelines(structured)

        stat_entities.markdown(
            f"<div class='stat'><h2>{len(entities)}</h2><p>Entities</p></div>",
            unsafe_allow_html=True
        )

        # ML
        if numeric_data:
            features = [list(numeric_data.values())]
            prediction = model.predict(features)[0]
        else:
            prediction = "N/A"

        stat_pred.markdown(
            f"<div class='stat'><h2>{prediction}</h2><p>Risk Score</p></div>",
            unsafe_allow_html=True
        )

        plan = generate_week_plan(guidelines, prediction)

    # -------------------------------------------------
    # OUTPUT
    # -------------------------------------------------
    st.markdown("---")
    st.subheader("🥗 Weekly Diet Plan")

    days = parse_diet_plan(plan)

    for day in days:
        with st.container():
            st.markdown(f"### {day['title']}")
            c1, c2, c3 = st.columns(3)

            c1.markdown(f"**🌅 Breakfast**\n{day['breakfast']}")
            c2.markdown(f"**☀️ Lunch**\n{day['lunch']}")
            c3.markdown(f"**🌙 Dinner**\n{day['dinner']}")

            if day["other"]:
                st.info(day["other"])

else:
    st.info("Upload a medical report to start analysis.")

