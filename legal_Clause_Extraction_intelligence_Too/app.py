import streamlit as st
import pandas as pd
from main import process_contract
from clause_splitter import extract_clauses_from_csv
from train_classifier import train_classifier
from PyPDF2 import PdfReader  # For PDF parsing

@st.cache
def load_model():
    return train_classifier()

def extract_text_from_pdf(uploaded_file):
    pdf = PdfReader(uploaded_file)
    return "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])

def get_text_from_file(uploaded_file):
    if uploaded_file.name.endswith('.txt'):
        return uploaded_file.read().decode("utf-8")
    elif uploaded_file.name.endswith('.pdf'):
        return extract_text_from_pdf(uploaded_file)
    else:
        return None

st.set_page_config(page_title="Clause Extraction Tool", layout="wide")
st.title("📄 Contract Clause Extraction & Analysis Tool")
st.markdown("Upload a contract file (.txt, .csv, .pdf)")

uploaded_file = st.file_uploader("Upload file", type=["txt", "csv", "pdf"])

if uploaded_file:
    model = load_model()
    st.success("✅ File uploaded. Processing...")

    if uploaded_file.name.endswith('.csv'):
        clauses = extract_clauses_from_csv(uploaded_file)
        results = []
        for clause in clauses:
            label = model.predict([clause])[0]
            from NER_model import run_ner
            entities = run_ner(clause)
            results.append({
                "clause": clause,
                "category": label,
                "entities": entities
            })

    else:
        raw_text = get_text_from_file(uploaded_file)
        if raw_text:
            results = process_contract(raw_text, model)
        else:
            st.error("❌ Could not read the uploaded file.")
            results = []

    st.info(f"🔍 Total clauses extracted: {len(results)}")

    for i, res in enumerate(results, 1):
        with st.expander(f"Clause {i}: {res['category']}"):
            st.text_area("Clause Text", res['clause'], height=150)
            if res['entities']:
                st.markdown("**Entities found:**")
                for ent, label in res['entities']:
                    st.markdown(f"- `{ent}` : *{label}*")
            else:
                st.markdown("No named entities detected.")
