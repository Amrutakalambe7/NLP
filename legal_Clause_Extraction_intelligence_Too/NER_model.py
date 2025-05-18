# ner_model.py

import spacy

def run_ner(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]
