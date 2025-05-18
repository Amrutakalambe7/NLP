import streamlit as st
import pandas as pd
from main import process_contract
from clause_splitter import extract_clauses_from_csv
from train_classifier import train_classifier
from PyPDF2 import PdfReader
from NER_model import run_ner
from fpdf import FPDF
import io

# --- Branding ---
st.set_page_config(page_title="Legal Clause Intelligence", layout="wide")

# --- Logo and Title ---
st.image("logo.png", width=100)
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

def export_to_csv(results):
    rows = []
    for r in results:
        ents = "; ".join([f"{text} ({label})" for text, label in r['entities']])
        rows.append({'Clause': r['clause'], 'Category': r['category'], 'Entities': ents})
    return pd.DataFrame(rows)

def sanitize_text(text):
    return (
        text.replace("’", "'")
            .replace("‘", "'")
            .replace("“", '"')
            .replace("”", '"')
            .replace("–", "-")
            .replace("—", "-")
    )

def export_to_pdf(results):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    for i, r in enumerate(results, 1):
        clause_text = sanitize_text(r['clause'])
        entities = ", ".join([f"{sanitize_text(t)} ({sanitize_text(l)})" for t, l in r['entities']])
        clause_block = f"Clause {i}: {r['category']}\n\n{clause_text}\n\nEntities: {entities}\n\n---\n"
        
        for line in clause_block.splitlines():
            pdf.multi_cell(0, 10, line)
    
    pdf_bytes = pdf.output(dest='S').encode('latin1')  # <-- fix here
    return pdf_bytes


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

    if results:
        st.markdown(f"### 📌 Total Clauses Extracted: `{len(results)}`")

        # Filters
        categories = sorted(set(r['category'] for r in results))
        selected_categories = st.multiselect("📂 Filter by Category:", categories, default=categories)

        search_term = st.text_input("🔍 Search Clause Text (optional):").lower()

        # Filter & display
        filtered = [
            r for r in results
            if r['category'] in selected_categories and search_term in r['clause'].lower()
        ]

        for i, res in enumerate(filtered, 1):
            with st.expander(f"📎 Clause {i} - {res['category']}", expanded=False):
                st.text_area("📝 Clause Text", res['clause'], height=150)
                if res['entities']:
                    st.markdown("**📍 Named Entities:**")
                    for ent, label in res['entities']:
                        st.markdown(f"- `{ent}` : *{label}*")
                else:
                    st.markdown("No entities detected.")

        # --- Export Buttons ---
        st.markdown("---")
        st.subheader("📤 Export Extracted Data")

        col1, col2 = st.columns(2)
        with col1:
            csv_df = export_to_csv(filtered)
            st.download_button("⬇️ Download as CSV", csv_df.to_csv(index=False).encode('utf-8'), file_name="clauses.csv", mime="text/csv")

        with col2:
            pdf_bytes = export_to_pdf(filtered)
            st.download_button("⬇️ Download as PDF", pdf_bytes, file_name="clauses.pdf", mime="application/pdf")

    else:
        st.warning("⚠️ No clauses extracted. Please check the file format.")
