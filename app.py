# src/app.py
import os
import uuid
from flask import Flask, request, render_template, flash
from werkzeug.utils import secure_filename

from src.multilingual import detect_language
from deep_translator import GoogleTranslator

from src.pdf_reader import extract_text_from_pdf
from src.search_engine import SuperSearchEngine

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"pdf"}

app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), "templates"))
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.secret_key = "replace-with-a-secure-key"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Search engine instance
ENGINE = SuperSearchEngine(use_bm25=True, use_faiss=True)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# -----------------------------
# MULTI-LANGUAGE TRANSLATOR
# -----------------------------
def translate_text(text, target_lang):
    try:
        return GoogleTranslator(source="auto", target=target_lang).translate(text)
    except:
        return text


# -----------------------------
# MAIN ROUTE
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    results = []
    mode = request.args.get("mode", "tfidf")
    combine = request.args.get("combine", "0") == "1"

    if request.method == "POST":

        # Clear previous index (single-upload mode)
        ENGINE.doc_texts.clear()
        ENGINE.doc_titles.clear()
        ENGINE.doc_ids.clear()
        ENGINE.corpus_tokens.clear()
        ENGINE.corpus_strings.clear()
        ENGINE.doc_meta = {}

        files = request.files.getlist("pdfs")
        uploaded_any = False

        # -----------------------------
        # PROCESS PDF UPLOAD
        # -----------------------------
        for f in files:
            if f and allowed_file(f.filename):
                uploaded_any = True

                filename = secure_filename(f.filename)
                unique_name = f"{uuid.uuid4().hex}_{filename}"

                path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
                f.save(path)

                # Extract text
                raw_text = extract_text_from_pdf(path)

                # Detect original language
                detected_lang = detect_language(raw_text)

                # Translate to English (for indexing)
                english_text = translate_text(raw_text, "en")

                # Add document to search engine
                ENGINE.add_document(unique_name, english_text, title=filename)

                # Save metadata
                ENGINE.doc_meta[unique_name] = {
                    "original_text": raw_text,
                    "language": detected_lang,
                    "filename": filename,
                    "path": path
                }

        if uploaded_any:
            ENGINE.build_index()
            flash(f"Uploaded {len(files)} file(s) | Multilingual index rebuilt.", "success")

        # -----------------------------
        # SEARCH PART
        # -----------------------------
        q = request.form.get("query", "").strip()
        mode = request.form.get("mode", mode)
        combine = request.form.get("combine", "0") == "1"

        # NEW → selected output language  
        target_lang = request.form.get("translate_to", "en")

        if q:
            raw_results = ENGINE.search(q, mode=mode, top_k=20, combine_bm25_tfidf=combine)

            # Prepare final output
            results = []
            for r in raw_results:
                doc_id = r["id"]
                meta = ENGINE.doc_meta.get(doc_id, {})

                original_text = meta.get("original_text", "")
                snippet = r["snippet"]

                # Translate snippet to target language
                full_translated_text = translate_text(original_text, target_lang)


                r["language"] = meta.get("language", "unknown")
                r["original_text"] = original_text
                r["translated_snippet"] = translate_text(snippet, target_lang)
                r["full_translated_text"] = full_translated_text
                r["target_lang"] = target_lang

                results.append(r)

    # List of indexed documents
    indexed = [{"id": d, "title": ENGINE.doc_titles.get(d, d)} for d in ENGINE.doc_ids]

    return render_template(
        "index.html",
        results=results,
        indexed=indexed,
        mode=mode,
        combine=combine,
        languages=LANG_CHOICES
    )


# -----------------------------
# AVAILABLE LANGUAGES
# -----------------------------
LANG_CHOICES = {
    "en": "English",
    "te": "Telugu",
    "hi": "Hindi",
    "ta": "Tamil",
    "kn": "Kannada",
    "ml": "Malayalam",
    "fr": "French",
    "es": "Spanish",
    "de": "German",
    "zh-CN": "Chinese",
    "ja": "Japanese",
    "ar": "Arabic",
    "ru": "Russian"
}

if __name__ == "__main__":
    app.run(debug=True)
