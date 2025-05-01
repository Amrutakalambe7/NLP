import streamlit as st
import nltk
import re
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Set of English stopwords
stop_words = set(stopwords.words('english'))

# Streamlit App
st.title("NLP Text Analyzer with WordCloud")

# Text input
user_input = st.text_area("Enter your text here:", height=200)

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Tokenization
        words = word_tokenize(user_input)
        sentences = sent_tokenize(user_input)

        # Stopwords
        stopword_tokens = [word for word in words if word.lower() in stop_words]

        # Blank space count (2 or more spaces)
        blank_space_matches = re.findall(r'  +', user_input)
        blank_spaces_count = len(blank_space_matches)

        # Whitespace character count (spaces, tabs, newlines, etc.)
        whitespace_matches = re.findall(r'\s', user_input)
        whitespace_chars_count = len(whitespace_matches)

        # Display results
        st.subheader("Tokenized Words")
        st.write(words)

        st.subheader("Tokenized Sentences")
        st.write(sentences)

        st.subheader("Stopword Tokens")
        st.write(stopword_tokens)

        st.subheader("Blank Spaces (2 or more spaces together)")
        st.write(f"Found {blank_spaces_count} blank space sequences:")
        st.code(blank_space_matches if blank_space_matches else "None")

        st.subheader("Whitespace Characters (spaces, tabs, newlines)")
        st.write(f"Found {whitespace_chars_count} whitespace characters:")
        st.code(whitespace_matches if whitespace_matches else "None")

        # WordCloud
        st.subheader("Word Cloud")
        wordcloud = WordCloud(width=800, height=400, background_color='white',
                              stopwords=stop_words).generate(user_input)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        st.pyplot(plt)
