import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Load data and preprocess only once
@st.cache(allow_output_mutation=True)
def load_data_and_train_models():
    dataset = pd.read_csv("Restaurant_Reviews.tsv", delimiter='\t', quoting=3)

    nltk.download('stopwords')
    corpus = []
    ps = PorterStemmer()
    for i in range(0, 1000):
        review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
        review = review.lower().split()
        review = [ps.stem(word) for word in review if word not in set(stopwords.words('english'))]
        corpus.append(' '.join(review))

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus).toarray()
    y = dataset.iloc[:, 1].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    models = {
        "Logistic Regression": LogisticRegression(),
        "Naive Bayes": GaussianNB(),
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC(probability=True),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        "LightGBM": LGBMClassifier()
    }

    trained_models = {}
    for name, model in models.items():
        if name == "Naive Bayes":
            model.fit(X_train.astype(np.float64), y_train)
        else:
            model.fit(X_train, y_train)
        trained_models[name] = model

    return vectorizer, trained_models

# Prediction helper
def preprocess_input(text, vectorizer):
    ps = PorterStemmer()
    text = re.sub('[^a-zA-Z]', ' ', text).lower().split()
    text = [ps.stem(word) for word in text if word not in set(stopwords.words('english'))]
    final = ' '.join(text)
    return vectorizer.transform([final]).toarray()

# Streamlit App
st.set_page_config(page_title="Customer Sentiment Analysis", layout="centered")
st.title("Customer Sentiment Analysis App")
st.markdown("Analyze the sentiment (positive/negative) of a customer review using various ML classifiers.")

review_input = st.text_area("Enter customer review:", "")

classifier_choice = st.selectbox(
    "Choose a classifier:",
    ["Logistic Regression", "Naive Bayes", "KNN", "Decision Tree", "Random Forest", "SVM", "XGBoost", "LightGBM"]
)

if st.button("Predict Sentiment"):
    if review_input.strip() == "":
        st.warning("Please enter a review.")
    else:
        vectorizer, trained_models = load_data_and_train_models()
        model = trained_models[classifier_choice]
        processed_input = preprocess_input(review_input, vectorizer)

        if classifier_choice == "Naive Bayes":
            prediction = model.predict(processed_input.astype(np.float64))[0]
            proba = model.predict_proba(processed_input.astype(np.float64))[0][1]
        else:
            prediction = model.predict(processed_input)[0]
            proba = model.predict_proba(processed_input)[0][1]

        sentiment = "👍 Positive" if prediction == 1 else "👎 Negative"
        st.subheader(f"Prediction: {sentiment}")
        st.write(f"Confidence: {proba:.2f}")
