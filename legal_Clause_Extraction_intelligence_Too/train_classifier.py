# clause_clasifier.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import pandas as pd

def train_classifier():
    df = pd.read_csv("contracts.csv")  # columns: clause_text, category
    pipe = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', LogisticRegression())
    ])
    pipe.fit(df['clause_text'], df['category'])
    return pipe
