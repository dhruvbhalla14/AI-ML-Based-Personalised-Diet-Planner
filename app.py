import streamlit as st
import joblib
import json
import io
from datetime import datetime
from openai import OpenAI
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

from extractor import extract_text

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="AI Diet Planner",
    page_icon="🥗",
    layout="wide"
)

# -------------------------------------------------
# WHITE PROFESSIONAL UI
# -------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

* {
    font-family: 'Poppins', sans-serif;
}

/* ------------------------------------------------ */
/* CLEANER BACKGROUND (less noisy, more pro) */
/* ------------------------------------------------ */
.main {
    background: linear-gradient(135deg,#f8fafc,#eef2f7);
}

/* Hide Streamlit branding */
#MainMenu, footer, header {visibility:hidden;}


/* ------------------------------------------------ */
/* HEADER – slightly sharper */
/* ------------------------------------------------ */
.custom-header {
    background: linear-gradient(135deg,#9c27b0,#e91e63);
    padding: 1.6rem;
    border-radius: 18px;
    margin-bottom: 1.6rem;
    box-shadow: 0 8px 30px rgba(156,39,176,.25);
}

.custom-header h1 {
    font-size: 2.6rem;
}


/* ------------------------------------------------ */
/* CARDS – more professional density */
/* ------------------------------------------------ */
.card {
    background: white;
    padding: 1.4rem;
    border-radius: 16px;
    box-shadow: 0 4px 18px rgba(0,0,0,.08);
    transition: .25s;
}

.card:hover {
    transform: translateY(-3px);
}


/* ------------------------------------------------ */
/* STAT CARDS – numbers pop */
/* ------------------------------------------------ */
.stat-card {
    background: white;
    padding: 1.2rem;
    border-radius: 14px;
    text-align: center;
    box-shadow: 0 4px 16px rgba(0,0,0,.06);
    border-top: 5px solid #e91e63;
}

.stat-card h2 {
    font-size: 2.2rem;
    color: #9c27b0;
}


/* ------------------------------------------------ */
/* DAY CARD – premium look */
/* ------------------------------------------------ */
.day-card {
    background: white;
    padding: 1.2rem;
    border-radius: 14px;
    margin-bottom: 1.2rem;
    border-left: 6px solid #9c27b0;
    box-shadow: 0 3px 14px rgba(0,0,0,.06);
    transition: .25s;
}

.day-card:hover {
    transform: translateX(6px);
}

/* FIX TEXT VISIBILITY (IMPORTANT) */
.day-card,
.day-card p,
.day-card div,
.day-card span {
    color: #111 !important;
}


/* ------------------------------------------------ */
/* MEAL BOXES – glass style */
/* ------------------------------------------------ */
.meal-section {
    background: #fafafa;
    padding: .9rem;
    border-radius: 10px;
    border: 1px solid #eee;
    box-shadow: 0 2px 8px rgba(0,0,0,.04);
    transition: .2s;
}

.meal-section:hover {
    background: #fff;
    box-shadow: 0 4px 14px rgba(0,0,0,.08);
}

.meal-section h4 {
    color: #e91e63;
}


/* ------------------------------------------------ */
/* UPLOADER */
/* ------------------------------------------------ */
[data-testid="stFileUploader"] {
    background: white;
    padding: 1.2rem;
    border-radius: 14px;
    border: 2px dashed #9c27b0;
}


/* ------------------------------------------------ */
/* BUTTONS */
/* ------------------------------------------------ */
.stButton>button {
    background: linear-gradient(135deg,#9c27b0,#e91e63);
    border-radius: 12px;
    font-weight: 600;
    height: 42px;
}

.stButton>button:hover {
    transform: scale(1.03);
}


/* ------------------------------------------------ */
/* TABS */
/* ------------------------------------------------ */
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg,#9c27b0,#e91e63);
    color:white;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# HEADER
# -------------------------------------------------
st.markdown("""
<div class="header">
<h1>🥗 AI Diet Planner</h1>
<p>Upload your medical report → Generate personalized weekly diet plan</p>
</div>
""", unsafe_allow_html=True)


# -------------------------------------------------
# OPENAI
# -------------------------------------------------
@st.cache_resource
def load_openai():
    return OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

client = load_openai()


# -------------------------------------------------
# PDF generator
# -------------------------------------------------
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


# -------------------------------------------------
# Diet generator
# -------------------------------------------------
def generate_plan(text):

    prompt = f"""
Create a clean 7-day Indian diet plan based on this medical report:

{text[:4000]}

Format:

Day 1
Breakfast:
Lunch:
Dinner:

Day 2
...
"""

    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}]
    )

    return res.choices[0].message.content


# -------------------------------------------------
# Parser
# -------------------------------------------------
def parse_plan(plan):
    days = []
    current = None

    for line in plan.split("\n"):
        if line.lower().startswith("day"):
            if current:
                days.append(current)
            current = {"title": line, "content": ""}
        elif current:
            current["content"] += line + "\n"

    if current:
        days.append(current)

    return days


# -------------------------------------------------
# UI FLOW
# -------------------------------------------------
st.markdown("### 📤 Upload Medical Report")

uploaded = st.file_uploader(
    "PDF / TXT / CSV",
    type=["pdf","txt","csv"]
)


if uploaded:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    text, numeric_data = extract_text(uploaded)

    st.write(f"📄 Extracted characters: **{len(text)}**")

    generate_btn = st.button("🥗 Generate Diet Plan", use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # -------------------------------------------------
    # GENERATE BUTTON
    # -------------------------------------------------
    if generate_btn:

        with st.spinner("Generating personalized diet plan..."):

            plan = generate_plan(text)

        st.success("Diet Plan Ready ✅")

        # -------------------------------------------------
        # DISPLAY BOXES
        # -------------------------------------------------
        st.markdown("## 🥗 Weekly Diet Plan")

        for day in parse_plan(plan):
            st.markdown(
                f"<div class='daybox'><h4>{day['title']}</h4><pre>{day['content']}</pre></div>",
                unsafe_allow_html=True
            )

        # -------------------------------------------------
        # DOWNLOADS
        # -------------------------------------------------
        st.markdown("## ⬇️ Download Report")

        col1, col2 = st.columns(2)

        json_data = {
            "date": str(datetime.now()),
            "diet_plan": plan
        }

        col1.download_button(
            "Download JSON",
            json.dumps(json_data, indent=2),
            file_name="diet_plan.json"
        )

        col2.download_button(
            "Download PDF",
            create_pdf(plan),
            file_name="diet_plan.pdf"
        )

else:
    st.info("Upload a medical report to begin.")


