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

# Custom CSS
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    /* Main background */
    .main {
        background: linear-gradient(135deg, #ffffff 0%, #fce4ec 25%, #f3e5f5 50%, #e1f5fe 75%, #ffffff 100%);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom header */
    .custom-header {
        background: linear-gradient(135deg, #9c27b0 0%, #e91e63 100%);
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(156, 39, 176, 0.3);
        text-align: center;
    }
    
    .custom-header h1 {
        color: white;
        font-size: 3rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .custom-header p {
        color: #fce4ec;
        font-size: 1.2rem;
        margin: 0.5rem 0 0 0;
        font-weight: 300;
    }
    
    .custom-header .credit {
        color: white;
        font-size: 0.9rem;
        margin-top: 1rem;
        font-weight: 500;
        opacity: 0.9;
    }
    
    /* Cards */
    .card {
        background: white;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(156, 39, 176, 0.15);
        margin-bottom: 2rem;
        border: 1px solid rgba(156, 39, 176, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 48px rgba(156, 39, 176, 0.25);
    }
    
    /* Day card for diet plan */
    .day-card {
        background: linear-gradient(135deg, #ffffff 0%, #fce4ec 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 1.5rem;
        border-left: 5px solid #9c27b0;
        box-shadow: 0 4px 16px rgba(156, 39, 176, 0.1);
        transition: all 0.3s ease;
    }
    
    .day-card:hover {
        transform: translateX(10px);
        box-shadow: 0 6px 24px rgba(156, 39, 176, 0.2);
    }
    
    .day-card h3 {
        color: #9c27b0;
        font-weight: 600;
        margin-bottom: 1rem;
        font-size: 1.4rem;
    }
    
    .meal-section {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 0.8rem;
        border: 1px solid rgba(233, 30, 99, 0.2);
    }
    
    .meal-section h4 {
        color: #e91e63;
        font-weight: 600;
        margin-bottom: 0.5rem;
        font-size: 1.1rem;
    }
    
    .meal-section p {
        color: #424242;
        line-height: 1.6;
        margin: 0;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background: white;
        padding: 2rem;
        border-radius: 20px;
        border: 2px dashed #9c27b0;
        box-shadow: 0 4px 16px rgba(156, 39, 176, 0.1);
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #e91e63;
        background: #fce4ec;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #9c27b0 0%, #e91e63 100%);
        color: white;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        border: none;
        box-shadow: 0 4px 16px rgba(156, 39, 176, 0.3);
        transition: all 0.3s ease;
        font-size: 1rem;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 24px rgba(156, 39, 176, 0.4);
        background: linear-gradient(135deg, #7b1fa2 0%, #c2185b 100%);
    }
    
    /* Download buttons */
    .download-btn {
        background: white;
        color: #9c27b0;
        border: 2px solid #9c27b0;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        margin: 0.5rem;
        transition: all 0.3s ease;
        text-decoration: none;
        display: inline-block;
    }
    
    .download-btn:hover {
        background: #9c27b0;
        color: white;
        transform: translateY(-2px);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background: white;
        padding: 1rem;
        border-radius: 15px;
        box-shadow: 0 4px 16px rgba(156, 39, 176, 0.1);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #9c27b0;
        font-weight: 600;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #9c27b0 0%, #e91e63 100%);
        color: white;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #9c27b0 0%, #e91e63 100%);
    }
    
    /* Info boxes */
    .stAlert {
        border-radius: 15px;
        border-left: 5px solid #9c27b0;
    }
    
    /* Text area */
    .stTextArea textarea {
        border-radius: 15px;
        border: 2px solid rgba(156, 39, 176, 0.2);
    }
    
    .stTextArea textarea:focus {
        border-color: #9c27b0;
        box-shadow: 0 0 0 2px rgba(156, 39, 176, 0.1);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f3e5f5 0%, #fce4ec 100%);
    }
    
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: #9c27b0;
    }
    
    /* Success/Error messages */
    .element-container div[data-testid="stMarkdownContainer"] p {
        line-height: 1.6;
    }
    
    /* Info box styling */
    .info-box {
        background: linear-gradient(135deg, #f3e5f5 0%, #fce4ec 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #9c27b0;
        margin: 1rem 0;
    }
    
    .info-box h4 {
        color: #9c27b0;
        margin-bottom: 0.5rem;
    }
    
    /* Stats cards */
    .stat-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 16px rgba(156, 39, 176, 0.1);
        border-top: 4px solid #e91e63;
    }
    
    .stat-card h2 {
        color: #9c27b0;
        font-size: 2.5rem;
        margin: 0;
        font-weight: 700;
    }
    
    .stat-card p {
        color: #757575;
        margin: 0.5rem 0 0 0;
        font-size: 0.9rem;
    }
    
    /* Animations */
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .card, .day-card {
        animation: slideIn 0.5s ease-out;
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
    <div class="credit">Developed by Dhruv Bhalla | Making Healthcare Smarter with AI</div>
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
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#9c27b0'),
        spaceAfter=30,
        alignment=1  # Center
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#e91e63'),
        spaceAfter=12,
        spaceBefore=12
    )
    
    # Title
    story.append(Paragraph("🥗 AI Diet Planner", title_style))
    story.append(Paragraph("Your Personalized Weekly Diet Plan", styles['Normal']))
    story.append(Paragraph(f"Generated on: {datetime.now().strftime('%B %d, %Y')}", styles['Normal']))
    story.append(Spacer(1, 0.3*inch))
    
    # Patient Info
    if patient_info:
        story.append(Paragraph("Patient Information", heading_style))
        info_data = [[k, str(v)] for k, v in patient_info.items()]
        info_table = Table(info_data, colWidths=[2*inch, 4*inch])
        info_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f3e5f5')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#9c27b0'))
        ]))
        story.append(info_table)
        story.append(Spacer(1, 0.3*inch))
    
    # Diet Plan
    story.append(Paragraph("Your Weekly Diet Plan", heading_style))
    story.append(Paragraph(diet_plan_data, styles['Normal']))
    
    # Footer
    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph("Developed by Dhruv Bhalla | AI-Powered Healthcare", styles['Italic']))
    story.append(Paragraph("⚠️ This is AI-generated guidance, not medical advice. Consult a doctor.", styles['Italic']))
    
    doc.build(story)
    buffer.seek(0)
    return buffer

# -----------------------
# Parse Diet Plan into Days
# -----------------------
def parse_diet_plan(plan_text):
    """Parse the diet plan text into structured day-by-day format"""
    days = []
    current_day = None
    
    lines = plan_text.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Detect day headers
        if any(day in line.lower() for day in ['day 1', 'day 2', 'day 3', 'day 4', 'day 5', 'day 6', 'day 7']):
            if current_day:
                days.append(current_day)
            current_day = {
                'title': line,
                'breakfast': '',
                'lunch': '',
                'dinner': '',
                'other': ''
            }
        elif current_day:
            # Categorize meals
            line_lower = line.lower()
            if 'breakfast' in line_lower or 'morning' in line_lower:
                current_day['breakfast'] += line + '\n'
            elif 'lunch' in line_lower or 'afternoon' in line_lower:
                current_day['lunch'] += line + '\n'
            elif 'dinner' in line_lower or 'evening' in line_lower:
                current_day['dinner'] += line + '\n'
            else:
                current_day['other'] += line + '\n'
    
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
You are an expert clinical nutritionist creating a personalized diet plan.

Patient condition: {structured}
Health risk score: {prediction}

Create a detailed 7-day Indian diet plan with:

For EACH DAY (Day 1 through Day 7):
- **Breakfast**: Specific items with approximate portions
- **Lunch**: Specific items with approximate portions  
- **Dinner**: Specific items with approximate portions
- Brief note on timing or preparation tips

Also include:
- Foods to AVOID
- Foods to PREFER
- Key nutritional tips
- Hydration advice

Use simple, home-cooked Indian foods. Be specific about portions and preparations.
Format clearly with day numbers and meal labels.

IMPORTANT: This is guidance only, not medical advice.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
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
    - 📄 Upload any medical document
    - 🧠 AI entity extraction
    - 📊 Health risk assessment
    - 🥗 7-day personalized diet
    - 📥 Download as PDF/JSON
    """)
    
    st.markdown("### 👨‍💻 Developer")
    st.success("""
    **Dhruv Bhalla**
    
    AI/ML Enthusiast | Healthcare Tech Innovator
    
    Making healthcare accessible through artificial intelligence.
    """)
    
    if model:
        st.success("✅ ML Model: Loaded")
    else:
        st.warning("⚠️ ML Model: Not loaded")
    
    if client:
        st.success("✅ AI Engine: Connected")
    else:
        st.warning("⚠️ AI Engine: Disconnected")

# -----------------------
# Main App
# -----------------------

st.markdown("### 📤 Upload Your Medical Report")

uploaded_file = st.file_uploader(
    "Drag and drop or click to upload (PDF, Image, CSV, TXT)",
    type=["pdf", "png", "jpg", "jpeg", "csv", "txt"],
    help="Upload any medical report or prescription"
)

if uploaded_file:
    
    # Create columns for better layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📋 Analysis Progress")
        progress_bar = st.progress(0)
        status_text = st.empty()
    
    with col2:
        st.markdown("### 📊 Quick Stats")
        stat_col1, stat_col2 = st.columns(2)
    
    try:
        # -----------------------
        # Step 1: Extract Text
        # -----------------------
        status_text.info("📄 Extracting text from document...")
        progress_bar.progress(15)
        
        text, numeric_data = extract_text(uploaded_file)
        
        with stat_col1:
            st.markdown(f"""
            <div class="stat-card">
                <h2>{len(text)}</h2>
                <p>Characters Extracted</p>
            </div>
            """, unsafe_allow_html=True)
        
        progress_bar.progress(25)
        
        # -----------------------
        # Step 2: NLP Analysis
        # -----------------------
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
        
        # -----------------------
        # Step 3: ML Prediction
        # -----------------------
        status_text.info("📊 Running health assessment...")
        progress_bar.progress(80)
        
        if numeric_data and model:
            features = [list(numeric_data.values())]
            prediction = model.predict(features)[0]
        else:
            prediction = "Insufficient data for numerical prediction"
        
        progress_bar.progress(85)
        
        # -----------------------
        # Step 4: Generate Diet Plan
        # -----------------------
        status_text.info("🥗 Generating personalized diet plan...")
        progress_bar.progress(90)
        
        plan = generate_week_plan(guidelines, prediction)
        
        progress_bar.progress(100)
        status_text.success("✅ Analysis Complete!")
        
        # Clear progress indicators after a moment
        import time
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        # -----------------------
        # Display Results in Tabs
        # -----------------------
        st.markdown("---")
        
        tab1, tab2, tab3, tab4 = st.tabs(["📄 Extracted Data", "🧠 Medical Analysis", "📊 Health Assessment", "🥗 Diet Plan"])
        
        with tab1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("#### 📄 Extracted Text from Document")
            st.text_area("", text, height=300, label_visibility="collapsed")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
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
                st.info(guidelines.get('lifestyle_advice', 'No specific advice'))
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab3:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("#### 📊 Health Risk Assessment")
            
            if isinstance(prediction, str):
                st.info(f"**Result:** {prediction}")
            else:
                st.success(f"**Health Risk Score:** {prediction}")
            
            st.warning("⚠️ **Important:** This is an AI assessment, not a medical diagnosis. Always consult with healthcare professionals.")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab4:
            st.markdown("#### 🥗 Your Personalized Weekly Diet Plan")
            
            # Parse diet plan into days
            days = parse_diet_plan(plan)
            
            if days:
                # Display in card format
                for day in days:
                    st.markdown(f"""
                    <div class="day-card">
                        <h3>📅 {day['title']}</h3>
                    """, unsafe_allow_html=True)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if day['breakfast'].strip():
                            st.markdown(f"""
                            <div class="meal-section">
                                <h4>🌅 Breakfast</h4>
                                <p>{day['breakfast']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    with col2:
                        if day['lunch'].strip():
                            st.markdown(f"""
                            <div class="meal-section">
                                <h4>☀️ Lunch</h4>
                                <p>{day['lunch']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    with col3:
                        if day['dinner'].strip():
                            st.markdown(f"""
                            <div class="meal-section">
                                <h4>🌙 Dinner</h4>
                                <p>{day['dinner']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    if day['other'].strip():
                        st.markdown(f"""
                        <div class="meal-section">
                            <h4>📝 Additional Notes</h4>
                            <p>{day['other']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                # Fallback to regular display
                st.markdown(f'<div class="card">{plan}</div>', unsafe_allow_html=True)
            
            # Download buttons
            st.markdown("---")
            st.markdown("### 📥 Download Your Diet Plan")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # JSON download
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
                    file_name=f"diet_plan_{datetime.now().strftime('%Y%m%d')}.json",
                    mime="application/json",
                    use_container_width=True
                )
            
            with col2:
                # PDF download
                patient_info = {
                    "Condition": guidelines.get('condition', 'General'),
                    "Health Assessment": str(prediction),
                    "Generated Date": datetime.now().strftime("%B %d, %Y")
                }
                
                pdf_buffer = generate_pdf(plan, patient_info)
                st.download_button(
                    label="📑 Download as PDF",
                    data=pdf_buffer,
                    file_name=f"diet_plan_{datetime.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
        
    except Exception as e:
        st.error(f"⚠️ An error occurred: {str(e)}")
        st.info("Please try uploading a different file or contact support.")

else:
    # Show welcome message when no file is uploaded
    st.markdown("""
    <div class="card">
        <h3 style="color: #9c27b0;">👋 Welcome to AI Diet Planner!</h3>
        <p style="font-size: 1.1rem; line-height: 1.8;">
            Upload your medical report to get started. Our AI will analyze your health data 
            and create a personalized 7-day diet plan tailored to your needs.
        </p>
        <br>
        <h4 style="color: #e91e63;">How it works:</h4>
        <ol style="line-height: 2;">
            <li><strong>Upload</strong> - Add your medical report (PDF, image, or text)</li>
            <li><strong>Analyze</strong> - AI extracts medical entities and assesses health</li>
            <li><strong>Generate</strong> - Get a personalized weekly diet plan</li>
            <li><strong>Download</strong> - Save your plan as PDF or JSON</li>
        </ol>
        <br>
        <p style="color: #757575; font-style: italic;">
            ⚠️ This tool provides dietary guidance based on AI analysis. Always consult 
            with healthcare professionals for medical advice.
        </p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; color: #757575;">
    <p style="font-size: 0.9rem;">
        Developed with ❤️ by <strong style="color: #9c27b0;">Dhruv Bhalla</strong> | 
        Powered by AI & Machine Learning
    </p>
    <p style="font-size: 0.8rem;">
        Making healthcare smarter, one diet plan at a time 🥗
    </p>
</div>
""", unsafe_allow_html=True)
