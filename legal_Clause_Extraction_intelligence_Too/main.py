from clause_splitter import extract_clauses
from NER_model import run_ner

def process_contract(text, model):
    clauses = extract_clauses(text)
    results = []

    for clause in clauses:
        if clause.strip():
            label = model.predict([clause])[0]
            entities = run_ner(clause)
            results.append({
                "clause": clause,
                "category": label,
                "entities": entities
            })
    return results
