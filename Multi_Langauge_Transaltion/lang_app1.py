import streamlit as st
from mtranslate import translate
import pandas as pd
import os
from gtts import gTTS
import base64

# App Config
st.set_page_config(page_title="Language Translator", layout="wide")

# Custom CSS for UI improvements
st.markdown("""
    <style>
        .main {background-color: #0e1117;}
        .stApp {background-color: #0e1117; color: white;}
        h1, h2, h3, .css-1v3fvcr, .css-h5rgaw {color: #f1f1f1;}
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .css-1d391kg {
            background-color: #1e1e1e;
            border-radius: 10px;
            padding: 10px;
        }
        .stTextArea label {
            font-weight: bold;
            color: #e2e2e2;
        }
        .stSidebar {
            background-color: #111;
        }
    </style>
""", unsafe_allow_html=True)

# Load language data
df = pd.read_csv(r'E:\VS code\NLP\Text_Summarization\language.csv')
df.dropna(inplace=True)
lang = df['name'].to_list()
langlist = tuple(lang)
langcode = df['iso'].to_list()
lang_array = {lang[i]: langcode[i] for i in range(len(langcode))}

# Title
st.markdown("<h1 style='text-align: center;'>🌍 Language Translator App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Instantly translate and listen to your text in multiple languages</p>", unsafe_allow_html=True)

# Input text
inputtext = st.text_area("✍️ Enter the text to translate", height=100, placeholder="Type your sentence here...")

# Sidebar language choice
choice = st.sidebar.radio("🌐 Choose Output Language", langlist)

# Audio supported languages
speech_langs = {
    "af": "Afrikaans", "ar": "Arabic", "bg": "Bulgarian", "bn": "Bengali",
    "bs": "Bosnian", "ca": "Catalan", "cs": "Czech", "cy": "Welsh", "da": "Danish",
    "de": "German", "el": "Greek", "en": "English", "eo": "Esperanto", "es": "Spanish",
    "et": "Estonian", "fi": "Finnish", "fr": "French", "gu": "Gujarati", "od": "odia",
    "hi": "Hindi", "hr": "Croatian", "hu": "Hungarian", "hy": "Armenian", "id": "Indonesian",
    "is": "Icelandic", "it": "Italian", "ja": "Japanese", "jw": "Javanese", "km": "Khmer",
    "kn": "Kannada", "ko": "Korean", "la": "Latin", "lv": "Latvian", "mk": "Macedonian",
    "ml": "Malayalam", "mr": "Marathi", "my": "Myanmar (Burmese)", "ne": "Nepali",
    "nl": "Dutch", "no": "Norwegian", "pl": "Polish", "pt": "Portuguese", "ro": "Romanian",
    "ru": "Russian", "si": "Sinhala", "sk": "Slovak", "sq": "Albanian", "sr": "Serbian",
    "su": "Sundanese", "sv": "Swedish", "sw": "Swahili", "ta": "Tamil", "te": "Telugu",
    "th": "Thai", "tl": "Filipino", "tr": "Turkish", "uk": "Ukrainian", "ur": "Urdu",
    "vi": "Vietnamese", "zh-CN": "Chinese"
}

# Function to create audio download link
def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    return f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">📥 Download {file_label}</a>'

# Processing translation
if len(inputtext.strip()) > 0:
    try:
        translated_text = translate(inputtext, lang_array[choice])
        st.markdown("### ✅ Translated Text")
        st.text_area("", translated_text, height=200)

        if lang_array[choice] in speech_langs:
            audio_file = gTTS(text=translated_text, lang=lang_array[choice], slow=False)
            audio_file.save("lang.mp3")
            with open("lang.mp3", "rb") as f:
                audio_data = f.read()
                st.audio(audio_data, format="audio/mp3")
                st.markdown(get_binary_file_downloader_html("lang.mp3", "Audio File"), unsafe_allow_html=True)

    except Exception as e:
        st.error(f"🚨 Translation failed: {str(e)}")
