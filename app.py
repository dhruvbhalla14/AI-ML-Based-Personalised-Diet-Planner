import streamlit as st
import joblib
import os
from dotenv import load_dotenv
from openai import OpenAI

# your files
from extractor import extract_text
from Bertgpt import (
    clean_and_segment,
    extract_entities,
    classify_intents,
    build_structured_intent,
    generate_diet_guidelines
)

# -----------------------
# Setup
# -----------------------
st.set_page_config(page_title="AI Diet Planner", layout="wide")
st.markdown("""
<style>

.main {
    background: linear-gradient(135deg, #ffffff 0%, #ffe6f0 50%, #f3e8ff 100%);
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Card style */
.card {
    background: white;
    padding: 25px;
    border-radius: 20px;
    box-shadow: 0px 8px 25px rgba(0,0,0,0.08);
    margin-bottom: 25px;
}

/* Title */
.title {
    text-align: center;
    font-size: 40px;
    font-weight: 700;
    color: #c77dff;
}

/* Subtitle */
.subtitle {
    text-align: center;
    color: #666;
    margin-bottom: 30px;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(45deg, #ff8fab, #cdb4ff);
    color: white;
    border-radius: 12px;
    height: 3em;
    font-weight: 600;
    border: none;
}

/* Text area */
textarea {
    border-radius: 12px !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: white;
    padding: 20px;
    border-radius: 18px;
    box-shadow: 0px 5px 20px rgba(0,0,0,0.05);
}

</style>
""", unsafe_allow_html=True)

load_dotenv()

# Initialize OpenAI client
@st.cache_resource
def init_openai():
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

try:
    client = init_openai()
    if client:
        st.success('✅ API key loaded', icon="✅")
except Exception as e:
    st.error(f'❌ API key not found: {e}')
    client = None

# -----------------------
# Load model
# -----------------------
@st.cache_resource
def load_model():
    return joblib.load("best_model.pkl")

try:
    model = load_model()
except Exception as e:
    st.error(f"Model loading error: {e}")
    model = None


# -----------------------
# Diet Plan Generator
# -----------------------
def generate_week_plan(structured, prediction):
    if not client:
        return "⚠️ OpenAI client not initialized. Please check your API key."

    prompt = f"""
You are an expert clinical nutritionist.

Patient condition:
{structured}

Health risk score:
{prediction}

Create:
• 7 day Indian diet plan
• breakfast lunch dinner
• foods to avoid
• foods to prefer
• simple home food
• explain reasoning
• medical but easy language

IMPORTANT:
This is only guidance, not medical advice.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ Error generating plan: {e}"


# -----------------------
# UI
# -----------------------

st.markdown('<div class="title">🥗 AI Diet Planner</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload your medical report and get a smart weekly diet plan</div>', unsafe_allow_html=True)
st.caption("⚠️ Not medical advice. Always consult a doctor.")

uploaded_file = st.file_uploader(
    "Upload PDF / Image / CSV / TXT",
    type=["pdf", "png", "jpg", "jpeg", "csv", "txt"]
)

if uploaded_file:

    # Create progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        # -----------------------
        # 1 Extract text
        # -----------------------
        status_text.text("📄 Extracting text from document...")
        progress_bar.progress(20)
        
        text, numeric_data = extract_text(uploaded_file)

        st.subheader("📄 Extracted Text")
        st.text_area("Extracted Text", text, height=200, label_visibility="collapsed")

        # -----------------------
        # 2 NLP pipeline
        # -----------------------
        status_text.text("🧠 Analyzing medical information (this may take a minute on first load)...")
        progress_bar.progress(40)
        
        sentences = clean_and_segment(text)
        
        progress_bar.progress(50)
        entities = extract_entities(sentences)
        
        progress_bar.progress(60)
        intents = classify_intents(sentences)
        
        progress_bar.progress(70)
        structured = build_structured_intent(entities, intents)
        guidelines = generate_diet_guidelines(structured)

        st.subheader("🧠 Detected Medical Info")
        st.json(guidelines)

        # -----------------------
        # 3 ML prediction
        # -----------------------
        status_text.text("📊 Running health assessment...")
        progress_bar.progress(80)
        
        if numeric_data and model:
            features = [list(numeric_data.values())]
            prediction = model.predict(features)[0]
        else:
            prediction = "Not enough numeric data"

        st.subheader("📊 Health Model Output")
        st.success(prediction)

        # -----------------------
        # 4 LLM diet plan
        # -----------------------
        status_text.text("🥗 Generating personalized diet plan...")
        progress_bar.progress(90)
        
        plan = generate_week_plan(guidelines, prediction)

        progress_bar.progress(100)
        status_text.text("✅ Complete!")

        st.subheader("🥗 Your Weekly Diet Plan")
        st.write(plan)

    except Exception as e:
        st.error(f"⚠️ Error processing: {e}")
        st.info("Please try uploading a different file or check the format.")
    
    finally:
        # Clean up progress indicators after a short delay
        import time
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
