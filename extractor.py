import pdfplumber
import pandas as pd


def extract_text(uploaded_file):
    """
    Cloud-safe extractor
    NO OCR
    NO tesseract
    """

    text = ""
    numeric_data = None

    name = getattr(uploaded_file, "name", "")
    file_type = name.split(".")[-1].lower()

    # -------------------------
    # PDF (text only)
    # -------------------------
    if file_type == "pdf":
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

    # -------------------------
    # TXT
    # -------------------------
    elif file_type == "txt":
        text = uploaded_file.read().decode("utf-8")

    # -------------------------
    # CSV
    # -------------------------
    elif file_type == "csv":
        df = pd.read_csv(uploaded_file)
        text = df.astype(str).to_string()
        numeric_data = df.iloc[0].to_dict()

    return text.strip(), numeric_data
