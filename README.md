# 🥗 AI Personalized Diet Planner

An AI-powered health assistant that analyzes **medical reports (PDF / Images / CSV)** and automatically generates a **personalized weekly diet plan** using OCR, BERT, Machine Learning, and LLMs.

Built with ❤️ using Python + Streamlit.

---

## 🚀 Live Demo

👉 Add your deployed Streamlit link here (optional)
Example: [https://ai-diet-planner.streamlit.app](https://ai-diet-planner.streamlit.app)

---

## ✨ Features

✅ Upload medical report (PDF / Image / CSV)
✅ OCR text extraction using Tesseract
✅ Biomedical Named Entity Recognition (BERT)
✅ Disease & diet intent detection
✅ Health risk prediction using trained ML model
✅ GPT-powered personalized weekly diet plan
✅ Beautiful modern Streamlit UI
✅ Works locally or deployable to cloud

---

## 🧠 How it Works (Pipeline)

Medical Report
↓
OCR / Text Extraction
↓
BERT Biomedical NER
↓
Intent Classification
↓
ML Health Risk Prediction
↓
LLM generates Personalized Diet Plan

---

## 🛠 Tech Stack

* Python
* Streamlit
* Transformers (HuggingFace BERT)
* OpenAI / GPT API
* Scikit-learn / Joblib
* Tesseract OCR
* Pandas

---

## 📂 Project Structure

```
app.py              → Streamlit frontend
extractor.py        → OCR & text extraction
bertgpt.py          → NER + intent classification
best_model.pkl      → trained ML model
requirements.txt    → dependencies
.env                → API key (not uploaded)
```

---

## ⚙️ Installation

### 1. Clone repo

```
git clone https://github.com/your-username/ai-diet-planner.git
cd ai-diet-planner
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Install Tesseract OCR (Windows)

Download:
[https://github.com/UB-Mannheim/tesseract/wiki](https://github.com/UB-Mannheim/tesseract/wiki)

---

## 🔐 Setup API Key

Create `.env` file:

```
OPENAI_API_KEY=your_key_here
```

---

## ▶ Run App

```
streamlit run app.py
```

---

## 📸 Screenshots

(Add screenshots here later)

---

## ⚠ Disclaimer

This project provides AI-generated dietary guidance only and **is not medical advice**. Always consult a healthcare professional.

---

## 👨‍💻 Author

Dhruv Bhalla
BCA Student | AI/ML Enthusiast
LinkedIn: [http://linkedin.com/in/dhruv-bhalla-65b870280](http://linkedin.com/in/dhruv-bhalla-65b870280)

---

## ⭐ If you like this project

Give it a star ⭐ on GitHub

