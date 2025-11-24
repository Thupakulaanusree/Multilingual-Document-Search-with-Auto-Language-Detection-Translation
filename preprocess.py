# src/preprocess.py
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# English stopwords only (we don't want to remove Telugu/Hindi words)
STOPWORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

def normalize_text(text: str, lemmatize: bool = True):
    if not text:
        return []

    # 1️⃣ Remove punctuation but KEEP ALL UNICODE letters (Telugu, Hindi etc.)
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)

    # 2️⃣ Lowercase (safe for Indian languages)
    text = text.lower()

    # 3️⃣ Split by whitespace (universal tokenizer)
    tokens = text.split()

    # 4️⃣ Remove ONLY English stopwords (don't break Telugu/Hindi)
    tokens = [t for t in tokens if t not in STOPWORDS]

    # 5️⃣ Lemmatize English ONLY (Unicode safe)
    if lemmatize:
        tokens = [LEMMATIZER.lemmatize(t) for t in tokens]

    return tokens


def tokens_to_string(tokens):
    """Turn tokens into a space-separated string."""
    return " ".join(tokens)
