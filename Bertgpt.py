import re
import spacy
from transformers import pipeline

# Load models once
nlp = spacy.load("en_core_web_sm")
ner_model = pipeline(
    "ner",
    model="d4data/biomedical-ner-all",
    aggregation_strategy="simple"
)
classifier = pipeline(
    "zero-shot-classification",
    model="typeform/distilbert-base-uncased-mnli"
)

LABELS = [
    "diagnosis",
    "diet advice",
    "medication",
    "lifestyle advice"
]

def clean_and_segment(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9., ]", "", text)
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 5]

def extract_entities(sentences):
    entities = []
    for s in sentences:
        entities.extend(ner_model(s))
    return entities

def classify_intents(sentences):
    results = []
    for s in sentences:
        out = classifier(s, LABELS)
        results.append({
            "sentence": s,
            "intent": out["labels"][0]
        })
    return results

def build_structured_intent(entities, intents):
    structured = {
        "diseases": [],
        "diet_advice": [],
        "lifestyle_advice": []
    }

    for e in entities:
        label = e["entity_group"].lower()
        if "disease" in label:
            structured["diseases"].append(e["word"])

    for i in intents:
        intent_label = i["intent"].lower()
        if "diet" in intent_label:
            structured["diet_advice"].append(i["sentence"])
        if "lifestyle" in intent_label:
            structured["lifestyle_advice"].append(i["sentence"])

    return structured

def generate_diet_guidelines(intent):
    disease = ", ".join(intent["diseases"]) if intent["diseases"] else "General"
    diet = ", ".join(intent["diet_advice"]) if intent["diet_advice"] else "No specific advice"
    lifestyle = ", ".join(intent["lifestyle_advice"]) if intent["lifestyle_advice"] else "No specific advice"

    return {
        "condition": disease,
        "allowed_foods": ["vegetables", "whole grains", "fruits"],
        "restricted_foods": ["sugar", "fried food", "junk food"],
        "diet_plan": f"Follow a balanced diet. {diet}",
        "lifestyle_advice": lifestyle
    }