import re
import pandas as pd

def extract_clauses(text):
    pattern = r'\n?\s*(\d+)\.\s+(.*?)(?=\n\s*\d+\.|\Z)'  # Match '1. Clause' style
    matches = re.findall(pattern, text, re.DOTALL)
    clauses = [clause.strip() for _, clause in matches]
    return clauses

def extract_clauses_from_csv(csv_file):
    df = pd.read_csv(csv_file)
    return df['clause_text'].dropna().tolist()

