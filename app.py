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
.main {
    background: #f6f8fb;
}

/* header */
.header {
    background: linear-gradient(90deg,#4f46e5,#7c3aed);
    padding: 25px;
    border-radius: 14px;
    color: white;
    text-align: center;
    margin-bottom: 25px;
}

/* cards */
.card {
    background: white;
    padding: 18px;
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    margin-bottom: 15px;
}

/* day boxes */
.daybox {
    background: #ffffff;
    padding: 16px;
    border-radius: 12px;
    margin-bottom: 12px;
    border-left: 5px solid #4CAF50;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);

    color: #000000;          /* ✅ ADD THIS */
}

/* force pre text black */
.daybox pre {
    color: #000000;          /* ✅ ADD THIS */
    font-size: 15px;
    white-space: pre-wrap;
}
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

