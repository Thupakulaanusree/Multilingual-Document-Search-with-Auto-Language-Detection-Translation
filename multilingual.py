# src/multilingual.py
# multilingual.py

from langdetect import detect
from deep_translator import GoogleTranslator

def detect_language(text):
    try:
        return detect(text)
    except:
        return "unknown"

def translate_to_english(text, src_lang):
    if src_lang == "en":
        return text  # no need to translate

    try:
        translated = GoogleTranslator(source=src_lang, target="en").translate(text)
        return translated
    except:
        return text  # fallback
