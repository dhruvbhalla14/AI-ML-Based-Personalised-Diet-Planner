from transformers import pipeline

ner = pipeline(
    "ner",
    model="dslim/bert-base-NER",
    aggregation_strategy="simple"
)

def extract_entities(text):

    results = ner(text)

    entities = []

    for r in results:
        entities.append({
            "entity": r["word"].lower(),
            "value": r["score"]
        })

    return entities