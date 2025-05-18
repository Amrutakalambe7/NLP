import streamlit as st
import pandas as pd
from main import process_contract
from clause_splitter import extract_clauses_from_csv
from train_classifier import train_classifier
from PyPDF2 import PdfReader
from NER_model import run_ner

# --- Styling ---
st.set_page_config(page_title="🧾 Legal Clause Intelligence", layout="wide")
st.markdown("""
    <style>
        .main {
            background-color: #0B0C10;
            color: #C5C6C7;
        }
        h1, h2, h3 {
            color: #66FCF1;
        }
        .stButton>button {
            color: white;
            background: #45A29E;
        }
        .stTextArea, .stTextInput, .stSelectbox {
            background-color: #1F2833;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# --- Header ---
st.title("🧾 Contract Clause Extraction & Analysis Tool")
st.markdown("Upload a contract file (`.txt`, `.csv`, `.pdf`) to extract, classify, and analyze legal clauses with NER insights.")

uploaded_file = st.file_uploader("📤 Upload file", type=["txt", "csv", "pdf"])

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
    return None

if uploaded_file:
    st.success("✅ File uploaded. Processing started...")

    model = load_model()
    results = []

    if uploaded_file.name.endswith('.csv'):
        clauses = extract_clauses_from_csv(uploaded_file)
        for clause in clauses:
            label = model.predict([clause])[0]
            entities = run_ner(clause)
            results.append({"clause": clause, "category": label, "entities": entities})
    else:
        text = get_text_from_file(uploaded_file)
        if text:
            results = process_contract(text, model)
        else:
            st.error("❌ Failed to extract text from file.")

    # --- Results Section ---
    if results:
        st.markdown(f"### 📌 Total Clauses Extracted: `{len(results)}`")

        # Clause category filter
        categories = sorted(set(res['category'] for res in results))
        selected_categories = st.multiselect("🔍 Filter by Clause Type:", categories, default=categories)

        for i, res in enumerate(results, 1):
            if res['category'] in selected_categories:
                with st.expander(f"📎 Clause {i} - {res['category']}", expanded=False):
                    st.text_area("📝 Clause Text", res['clause'], height=150)
                    if res['entities']:
                        st.markdown("**📍 Named Entities:**")
                        for ent, label in res['entities']:
                            st.markdown(f"- `{ent}` : *{label}*")
                    else:
                        st.markdown("No entities detected.")
    else:
        st.warning("⚠️ No clauses extracted. Please check the file format.")
