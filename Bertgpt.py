import re
import spacy
from transformers import pipeline

# Don't load models at import time - load them lazily
_nlp = None
_ner_model = None
_classifier = None

LABELS = [
    "diagnosis",
    "diet advice",
    "medication",
    "lifestyle advice"
]

def get_nlp():
    """Lazy load spacy model"""
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp

def get_ner_model():
    """Lazy load NER model"""
    global _ner_model
    if _ner_model is None:
        _ner_model = pipeline(
            "ner",
            model="d4data/biomedical-ner-all",
            aggregation_strategy="simple"
        )
    return _ner_model

def get_classifier():
    """Lazy load classifier"""
    global _classifier
    if _classifier is None:
        _classifier = pipeline(
            "zero-shot-classification",
            model="typeform/distilbert-base-uncased-mnli"
        )
    return _classifier

def clean_and_segment(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9., ]", "", text)
    nlp = get_nlp()
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 5]

def extract_entities(sentences):
    ner_model = get_ner_model()
    entities = []
    for s in sentences:
        entities.extend(ner_model(s))
    return entities

def classify_intents(sentences):
    classifier = get_classifier()
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
