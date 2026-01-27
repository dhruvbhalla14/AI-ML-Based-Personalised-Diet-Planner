import pdfplumber
import pytesseract
from PIL import Image
import pandas as pd


def extract_text(uploaded_file):
    text = ""
    numeric_data = None

    filename = getattr(uploaded_file, "name", "")
    file_type = filename.split(".")[-1].lower()

    uploaded_file.seek(0)

    # -------------------------
    # PDF (OCR fallback)
    # -------------------------
    if file_type == "pdf":
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()

                if page_text and page_text.strip():
                    text += page_text + "\n"
                else:
                    img = page.to_image(resolution=300).original
                    text += pytesseract.image_to_string(img) + "\n"

    # -------------------------
    # Images
    # -------------------------
    elif file_type in ["png", "jpg", "jpeg"]:
        image = Image.open(uploaded_file)
        text = pytesseract.image_to_string(image)

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
        text = df.iloc[0].astype(str).str.cat(sep=" ")
        numeric_data = df.iloc[0].to_dict()

    return text.strip(), numeric_data
