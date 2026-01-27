import streamlit as st
import joblib
import os
import json
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.units import inch
import io

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
st.set_page_config(
    page_title="AI Diet Planner by Dhruv",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Fixed visibility issues
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main background */
    .main {
        background: linear-gradient(135deg, #f8fafc 0%, #e0f2fe 50%, #f0f9ff 100%);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom header */
    .custom-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #0891b2 100%);
        padding: 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(30, 58, 138, 0.25);
        text-align: center;
    }
    
    .custom-header h1 {
    color: #ffffff;
    text-shadow: 0 4px 14px rgba(0,0,0,0.55);
    letter-spacing: 1px;
    font-weight: 800;
}
    
    .custom-header p {
        color: #e0f2fe;
        font-size: 1.3rem;
        margin: 0.5rem 0 0 0;
        font-weight: 400;
    }
    
    .custom-header .credit {
        color: #ffffff;
        font-size: 1rem;
        margin-top: 1rem;
        font-weight: 600;
        opacity: 0.95;
        background: rgba(255, 255, 255, 0.1);
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
    }
    
    /* Welcome card - HIGH CONTRAST TEXT */
    .welcome-card {
        background: white;
        padding: 2.5rem;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        margin-bottom: 2rem;
        border: 1px solid #e5e7eb;
    }
    
    .welcome-card h3 {
        color: #1e3a8a !important;
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    
    .welcome-card p {
        color: #1f2937 !important;
        font-size: 1.1rem;
        line-height: 1.8;
        margin-bottom: 1rem;
    }
    
    .welcome-card h4 {
        color: #0891b2 !important;
        font-size: 1.4rem;
        font-weight: 600;
        margin: 1.5rem 0 1rem 0;
    }
    
    .welcome-card ol, .welcome-card ul {
        color: #1f2937 !important;
        font-size: 1.05rem;
        line-height: 2.2;
    }
    
    .welcome-card ol li, .welcome-card ul li {
        color: #1f2937 !important;
    }
    
    .welcome-card strong {
        color: #1e3a8a !important;
        font-weight: 600;
    }
    
    /* Cards */
    /* ===== FIX TEXT VISIBILITY IN WELCOME CARD ===== */
    .card h4,
    .card li,
    .card ol,
    .card ul {
    color: #1f2937 !important;
}
    .card {
        background: white;
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        margin-bottom: 2rem;
        border: 1px solid #e5e7eb;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(30, 58, 138, 0.15);
    }
    
    .card h4 {
        color: #1e3a8a !important;
        font-weight: 600;
    }
    
    .card p, .card div {
        color: #1f2937 !important;
    }
    
    /* Day card for diet plan */
    .day-card {
        background: linear-gradient(135deg, #ffffff 0%, #f0f9ff 100%);
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        border-left: 6px solid #0891b2;
        box-shadow: 0 4px 16px rgba(8, 145, 178, 0.1);
        transition: all 0.3s ease;
    }
    
    .day-card:hover {
        transform: translateX(8px);
        box-shadow: 0 6px 24px rgba(8, 145, 178, 0.2);
    }
    
    .day-card h3 {
        color: #1e3a8a !important;
        font-weight: 700;
        margin-bottom: 1.5rem;
        font-size: 1.8rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .meal-section {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        border: 2px solid #e0f2fe;
        min-height: 120px;
        transition: border-color 0.2s ease;
    }
    
    .meal-section:hover {
        border-color: #0891b2;
    }
    
    .meal-section h4 {
        color: #0891b2 !important;
        font-weight: 700;
        margin-bottom: 0.8rem;
        font-size: 1.2rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .meal-section p {
        color: #1f2937 !important;
        line-height: 1.8;
        margin: 0;
        font-size: 1rem;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background: white;
        padding: 2.5rem;
        border-radius: 16px;
        border: 3px dashed #0891b2;
        box-shadow: 0 4px 16px rgba(8, 145, 178, 0.1);
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #1e3a8a;
        background: #f0f9ff;
    }
    
    [data-testid="stFileUploader"] label {
        color: #1e3a8a !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #1e3a8a 0%, #0891b2 100%);
        color: white !important;
        border-radius: 12px;
        padding: 0.8rem 2.5rem;
        font-weight: 600;
        border: none;
        box-shadow: 0 4px 16px rgba(30, 58, 138, 0.3);
        transition: all 0.3s ease;
        font-size: 1.05rem;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 24px rgba(30, 58, 138, 0.4);
        background: linear-gradient(135deg, #1e40af 0%, #0e7490 100%);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background: white;
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        border-bottom: none;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #1e3a8a !important;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-size: 1rem;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #1e3a8a 0%, #0891b2 100%);
        color: white !important;
    }
    
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 2rem;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #1e3a8a 0%, #0891b2 100%);
    }
    
    /* Text area - FIXED VISIBILITY */
    .stTextArea textarea {
        border-radius: 12px;
        border: 2px solid #cbd5e1;
        color: #1f2937 !important;
        background-color: #f8fafc !important;
        font-size: 0.95rem;
        font-family: 'Courier New', monospace;
    }
    
    .stTextArea textarea:focus {
        border-color: #0891b2;
        box-shadow: 0 0 0 3px rgba(8, 145, 178, 0.1);
        background-color: #ffffff !important;
    }
    
    .stTextArea label {
        color: #1e3a8a !important;
        font-weight: 600;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f0f9ff 0%, #e0f2fe 100%);
    }
    
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: #1e3a8a !important;
    }
    
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] div {
        color: #1f2937 !important;
    }
    
    /* Stats cards */
    .stat-card {
        background: white;
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 16px rgba(8, 145, 178, 0.1);
        border-top: 5px solid #0891b2;
        transition: transform 0.2s ease;
    }
    
    .stat-card:hover {
        transform: translateY(-4px);
    }
    
    .stat-card h2 {
        color: #1e3a8a !important;
        font-size: 3rem;
        margin: 0;
        font-weight: 800;
    }
    
    .stat-card p {
        color: #64748b !important;
        margin: 0.5rem 0 0 0;
        font-size: 1rem;
        font-weight: 500;
    }
    
    /* Section headers */
    .section-header {
        color: #1e3a8a !important;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #0891b2;
    }
    
    /* Success/Error messages */
    .stSuccess {
        background-color: #d1fae5 !important;
        color: #065f46 !important;
    }
    
    .stError {
        background-color: #fee2e2 !important;
        color: #991b1b !important;
    }
    
    .stInfo {
        background-color: #dbeafe !important;
        color: #1e40af !important;
    }
    
    .stWarning {
        background-color: #fef3c7 !important;
        color: #92400e !important;
    }
    
    /* JSON display - HIGH CONTRAST */
    pre {
        background-color: #f8fafc !important;
        color: #1f2937 !important;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #cbd5e1;
    }
    
    code {
        color: #1f2937 !important;
        background-color: #f1f5f9 !important;
    }
    
    /* Markdown text */
    .element-container p {
        color: #1f2937 !important;
    }
    
    .element-container h1, .element-container h2, .element-container h3, 
    .element-container h4, .element-container h5, .element-container h6 {
        color: #1e3a8a !important;
    }
    
    /* Download section */
    .download-section {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        padding: 2rem;
        border-radius: 16px;
        margin-top: 2rem;
        border: 2px solid #0891b2;
    }
    
    .download-section h3 {
        color: #1e3a8a !important;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------
# Header
# -----------------------
st.markdown("""
<div class="custom-header">
    <h1>🥗 AI Diet Planner</h1>
    <p>Your Personal Nutritionist Powered by Artificial Intelligence</p>
    <div class="credit">💻 Developed by Dhruv Bhalla | Making Healthcare Smarter</div>
</div>
""", unsafe_allow_html=True)

# Load environment variables
load_dotenv()

# -----------------------
# Initialize OpenAI
# -----------------------
@st.cache_resource
def init_openai():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets.get("OPENAI_API_KEY")
        except:
            pass
    if api_key:
        return OpenAI(api_key=api_key)
    return None

client = init_openai()

# -----------------------
# Load ML Model
# -----------------------
@st.cache_resource
def load_model():
    try:
        return joblib.load("best_model.pkl")
    except Exception as e:
        st.warning(f"ML Model not loaded: {e}")
        return None

model = load_model()

# -----------------------
# PDF Generator
# -----------------------
def generate_pdf(diet_plan_data, patient_info):
    """Generate a professional PDF of the diet plan"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
    story = []
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1e3a8a'),
        spaceAfter=30,
        alignment=1
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#0891b2'),
        spaceAfter=12,
        spaceBefore=12
    )
    
    story.append(Paragraph("🥗 AI Diet Planner", title_style))
    story.append(Paragraph("Your Personalized Weekly Diet Plan", styles['Normal']))
    story.append(Paragraph(f"Generated on: {datetime.now().strftime('%B %d, %Y')}", styles['Normal']))
    story.append(Spacer(1, 0.3*inch))
    
    if patient_info:
        story.append(Paragraph("Patient Information", heading_style))
        info_data = [[k, str(v)] for k, v in patient_info.items()]
        info_table = Table(info_data, colWidths=[2*inch, 4*inch])
        info_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e0f2fe')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#0891b2'))
        ]))
        story.append(info_table)
        story.append(Spacer(1, 0.3*inch))
    
    story.append(Paragraph("Your Weekly Diet Plan", heading_style))
    
    for line in diet_plan_data.split('\n'):
        if line.strip():
            story.append(Paragraph(line, styles['Normal']))
            story.append(Spacer(1, 0.1*inch))
    
    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph("Developed by Dhruv Bhalla | AI-Powered Healthcare", styles['Italic']))
    story.append(Paragraph("⚠️ This is AI-generated guidance, not medical advice. Consult a doctor.", styles['Italic']))
    
    doc.build(story)
    buffer.seek(0)
    return buffer

# -----------------------
# Parse Diet Plan
# -----------------------
def parse_diet_plan(plan_text):
    """Parse the diet plan text into structured day-by-day format"""
    days = []
    current_day = None
    current_meal = None
    
    lines = plan_text.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line or line in ['---', '***', '___']:
            continue
        
        line_lower = line.lower()
        
        day_keywords = ['day 1', 'day 2', 'day 3', 'day 4', 'day 5', 'day 6', 'day 7',
                        'day one', 'day two', 'day three', 'day four', 'day five', 'day six', 'day seven']
        
        if any(keyword in line_lower for keyword in day_keywords):
            if current_day:
                days.append(current_day)
            current_day = {
                'title': line,
                'breakfast': '',
                'lunch': '',
                'dinner': '',
                'snacks': '',
                'notes': ''
            }
            current_meal = None
            continue
        
        if current_day:
            if 'breakfast' in line_lower or 'morning meal' in line_lower:
                current_meal = 'breakfast'
                continue
            elif 'lunch' in line_lower or 'afternoon meal' in line_lower or 'mid-day' in line_lower:
                current_meal = 'lunch'
                continue
            elif 'dinner' in line_lower or 'evening meal' in line_lower or 'night' in line_lower:
                current_meal = 'dinner'
                continue
            elif 'snack' in line_lower or 'mid-meal' in line_lower:
                current_meal = 'snacks'
                continue
            elif any(word in line_lower for word in ['note', 'tip', 'important', 'remember', 'hydration']):
                current_meal = 'notes'
            
            if current_meal:
                line = line.replace('**', '').replace('*', '').strip()
                if line and not line.startswith('#'):
                    current_day[current_meal] += line + '\n'
    
    if current_day:
        days.append(current_day)
    
    return days

# -----------------------
# Diet Plan Generator
# -----------------------
def generate_week_plan(structured, prediction):
    if not client:
        return "⚠️ OpenAI API not configured. Please add OPENAI_API_KEY."

    prompt = f"""
You are an expert clinical nutritionist creating a personalized Indian diet plan.

Patient Information:
- Condition: {structured.get('condition', 'General')}
- Health Risk Score: {prediction}
- Detected Issues: {', '.join(structured.get('diseases', ['None']))}

Create a DETAILED 7-day Indian diet plan. For EACH day from Day 1 to Day 7, provide:

**Day [Number]**

**Breakfast:**
- Specific items with quantities (e.g., 2 chapatis, 1 cup dal, etc.)

**Lunch:**
- Specific items with quantities

**Dinner:**
- Specific items with quantities

**Snacks (Optional):**
- Healthy between-meal options

**Important Notes:**
- Hydration reminders
- Timing suggestions

At the end, include:
- Foods to COMPLETELY AVOID
- Foods to PREFER and increase
- General nutritional guidelines

Use simple, home-cooked Indian foods. Be very specific about portions.

IMPORTANT: This is dietary guidance, not medical advice.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=2000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ Error generating diet plan: {str(e)}"

# -----------------------
# Sidebar
# -----------------------
with st.sidebar:
    st.markdown("### 🎯 About This App")
    st.info("""
    This AI-powered diet planner analyzes your medical reports and generates personalized weekly diet plans.
    
    **Features:**
    - 📄 Multi-format support
    - 🧠 AI entity extraction
    - 📊 Health risk assessment
    - 🥗 7-day personalized diet
    - 📥 Download as PDF/JSON
    """)
    
    st.markdown("### 👨‍💻 Developer")
    st.success("""
    **Dhruv Bhalla**
    
    AI/ML Enthusiast
    Healthcare Tech Innovator
    """)
    
    st.markdown("### ⚙️ System Status")
    if model:
        st.success("✅ ML Model: Loaded")
    else:
        st.warning("⚠️ ML Model: Not available")
    
    if client:
        st.success("✅ OpenAI API: Connected")
    else:
        st.error("❌ OpenAI API: Not connected")

# -----------------------
# Main App
# -----------------------

st.markdown('<h2 class="section-header">📤 Upload Your Medical Report</h2>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Drag and drop or click to upload (PDF, Image, CSV, TXT)",
    type=["pdf", "png", "jpg", "jpeg", "csv", "txt"],
    help="Upload any medical report, prescription, or health document"
)

if uploaded_file:
    if st.button("🚀 Generate Personalized Diet Plan", type="primary", use_container_width=True):
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 📋 Analysis Progress")
            progress_bar = st.progress(0)
            status_text = st.empty()
        
        with col2:
            st.markdown("### 📊 Quick Stats")
            stat_col1, stat_col2 = st.columns(2)
        
        try:
            status_text.info("📄 Extracting text from document...")
            progress_bar.progress(15)
            
            text, numeric_data = extract_text(uploaded_file)
            
            if not text or len(text.strip()) < 10:
                st.error("❌ Could not extract meaningful text from the file.")
                st.stop()
            
            with stat_col1:
                st.markdown(f"""
                <div class="stat-card">
                    <h2>{len(text)}</h2>
                    <p>Characters Extracted</p>
                </div>
                """, unsafe_allow_html=True)
            
            progress_bar.progress(25)
            
            status_text.info("🧠 Analyzing medical information...")
            progress_bar.progress(35)
            
            sentences = clean_and_segment(text)
            progress_bar.progress(45)
            
            entities = extract_entities(sentences)
            progress_bar.progress(55)
            
            intents = classify_intents(sentences)
            progress_bar.progress(65)
            
            structured = build_structured_intent(entities, intents)
            guidelines = generate_diet_guidelines(structured)
            
            with stat_col2:
                st.markdown(f"""
                <div class="stat-card">
                    <h2>{len(entities)}</h2>
                    <p>Medical Entities Found</p>
                </div>
                """, unsafe_allow_html=True)
            
            progress_bar.progress(75)
            
            status_text.info("📊 Running health assessment...")
            progress_bar.progress(80)
            
            if numeric_data and model:
                try:
                    features = [[float(v) if isinstance(v, (int, float, str)) and str(v).replace('.','',1).replace('-','',1).isdigit() else 0 for v in numeric_data.values()]]
                    prediction = model.predict(features)[0]
                except Exception as e:
                    prediction = f"Could not generate prediction: {str(e)}"
            else:
                prediction = "Insufficient numerical data"
            
            progress_bar.progress(85)
            
            status_text.info("🥗 Generating diet plan...")
            progress_bar.progress(90)
            
            plan = generate_week_plan(guidelines, prediction)
            
            progress_bar.progress(100)
            status_text.success("✅ Complete!")
            
            import time
            time.sleep(1)
            progress_bar.empty()
            status_text.empty()
            
            st.markdown("---")
            
            tab1, tab2, tab3, tab4 = st.tabs(["📄 Extracted Data", "🧠 Medical Analysis", "📊 Health Assessment", "🥗 Your Diet Plan"])
            
            with tab1:
                st.markdown("#### 📄 Extracted Text from Your Document")
                st.text_area("", text, height=300, label_visibility="collapsed", key="extracted_text")
            
            with tab2:
                st.markdown("#### 🧠 AI-Detected Medical Information")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**🔍 Detected Conditions:**")
                    if guidelines.get('condition') and guidelines['condition'] != "General":
                        st.info(guidelines['condition'])
                    else:
                        st.info("No specific conditions detected")
                    
                    st.markdown("**✅ Recommended Foods:**")
                    for food in guidelines.get('allowed_foods', []):
                        st.success(f"• {food}")
                
                with col2:
                    st.markdown("**❌ Foods to Avoid:**")
                    for food in guidelines.get('restricted_foods', []):
                        st.error(f"• {food}")
                    
                    st.markdown("**💡 Lifestyle Advice:**")
                    st.info(guidelines.get('lifestyle_advice', 'Maintain balanced lifestyle'))
            
            with tab3:
                st.markdown("#### 📊 Health Risk Assessment")
                st.success(f"**Assessment Result:** {prediction}")
                st.warning("⚠️ **Important:** This is an AI assessment. Always consult healthcare professionals.")
            
            with tab4:
                st.markdown('<h3 class="section-header">🥗 Your 7-Day Diet Plan</h3>', unsafe_allow_html=True)
                
                days = parse_diet_plan(plan)
                
                if days and len(days) > 0:
                    for day in days:
                        st.markdown(f"""
                        <div class="day-card">
                            <h3>📅 {day['title']}</h3>
                        """, unsafe_allow_html=True)
                        
                        meal_col1, meal_col2, meal_col3 = st.columns(3)
                        
                        with meal_col1:
                            st.markdown(f"""
                            <div class="meal-section">
                                <h4>🌅 Breakfast</h4>
                                <p>{day['breakfast'].strip() if day['breakfast'].strip() else 'Similar to Day 1'}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with meal_col2:
                            st.markdown(f"""
                            <div class="meal-section">
                                <h4>☀️ Lunch</h4>
                                <p>{day['lunch'].strip() if day['lunch'].strip() else 'Similar to Day 1'}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with meal_col3:
                            st.markdown(f"""
                            <div class="meal-section">
                                <h4>🌙 Dinner</h4>
                                <p>{day['dinner'].strip() if day['dinner'].strip() else 'Similar to Day 1'}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        if day['snacks'].strip() or day['notes'].strip():
                            notes_col1, notes_col2 = st.columns(2)
                            
                            if day['snacks'].strip():
                                with notes_col1:
                                    st.markdown(f"""
                                    <div class="meal-section">
                                        <h4>🍎 Snacks</h4>
                                        <p>{day['snacks'].strip()}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                            
                            if day['notes'].strip():
                                with notes_col2:
                                    st.markdown(f"""
                                    <div class="meal-section">
                                        <h4>📝 Notes</h4>
                                        <p>{day['notes'].strip()}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.markdown(plan)
                
                st.markdown('<div class="download-section">', unsafe_allow_html=True)
                st.markdown("### 📥 Download Your Diet Plan")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    diet_data = {
                        "generated_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "patient_info": guidelines,
                        "health_assessment": str(prediction),
                        "diet_plan": plan,
                        "developer": "Dhruv Bhalla"
                    }
                    
                    json_str = json.dumps(diet_data, indent=2)
                    st.download_button(
                        label="📄 Download as JSON",
                        data=json_str,
                        file_name=f"diet_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        use_container_width=True
                    )
                
                with col2:
                    patient_info = {
                        "Condition": guidelines.get('condition', 'General'),
                        "Health Assessment": str(prediction),
                        "Generated Date": datetime.now().strftime("%B %d, %Y")
                    }
                    
                    pdf_buffer = generate_pdf(plan, patient_info)
                    st.download_button(
                        label="📑 Download as PDF",
                        data=pdf_buffer,
                        file_name=f"diet_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                
                st.markdown('</div>', unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"⚠️ Error: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

else:
    st.markdown("""
<div class="card">

<h4>📋 How it works:</h4>
<ol>
    <li><strong>Upload</strong> your medical report (PDF, image, CSV, or text file)</li>
    <li><strong>Click "Generate Diet Plan"</strong> to start the analysis</li>
    <li><strong>Review</strong> your personalized diet plan with day-by-day meal suggestions</li>
    <li><strong>Download</strong> your plan as PDF or JSON for easy reference</li>
</ol>

<h4>✨ What you'll get:</h4>
<ul style="line-height: 2;">
    <li>🔍 AI-powered analysis of your medical data</li>
    <li>📊 Health risk assessment</li>
    <li>🥗 Complete 7-day Indian diet plan</li>
    <li>🍽️ Breakfast, lunch, and dinner suggestions for each day</li>
    <li>✅ Foods to eat and ❌ foods to avoid</li>
    <li>📥 Downloadable PDF and JSON formats</li>
</ul>

<p style="color:#64748b;font-style:italic;margin-top:1.5rem;padding:1rem;background:#f0f9ff;border-radius:8px;border-left:4px solid #0891b2;">
⚠️ <strong>Important:</strong> This tool provides dietary guidance based on AI analysis.
Always consult healthcare professionals.
</p>

</div>
""", unsafe_allow_html=True)






