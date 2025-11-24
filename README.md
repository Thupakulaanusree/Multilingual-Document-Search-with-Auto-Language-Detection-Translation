 Multilingual Intelligent PDF Search Engine

 *BM25 + TF-IDF + FAISS | Auto Language Detection | Snippet Translation | PDF Search*

This project is a **Flask-based multilingual search engine** that allows users to:

✔ Upload **multiple PDF files**
✔ Automatically **extract text**
✔ Automatically **detect PDF language** (Hindi, Telugu, English, etc.)
✔ Search using **BM25**, **TF-IDF**, or **FAISS**
✔ Highlight matching keywords in snippets
✔ Translate snippets or full documents to **any target language**
✔ Handle mixed-language PDFs safely
✔ Provide ranked results with scores

This makes it a **mini Google-like search engine for multilingual PDFs**.

---

**Features**

**1. Automatic Language Detection**

* Detects the language of PDF text using `langdetect`
* Supports Telugu / Hindi / English / Tamil / Kannada / Malayalam / etc.
* Bypasses wrong detections using improved chunk-based detection logic

---

 **2. Intelligent Search**

You can choose ranking:

| Mode                     | Description                    |
| ------------------------ | ------------------------------ |
| **BM25**                 | Best for keyword-heavy queries |
| **TF-IDF**               | Classic cosine similarity      |
| **FAISS**                | Fast vector similarity search  |
| **BM25 + TF-IDF Fusion** | Combined weighted scoring      |

All results display:

✔ file name
✔ detected language
✔ score
✔ highlighted snippet

---

 **3. Translation Support**

Using `deep-translator` + Google Translate API internally.

Supports translating:

* Snippets
* Full PDF text

Into target languages like:

`English, Telugu, Hindi, Tamil, Kannada, Malayalam, French, Japanese, Chinese, German, Arabic, Russian`

---

**4. PDF Processing**

* Extracts text using PDFMiner
* Handles Unicode scripts correctly (Indian languages)
* Prevents empty-vocabulary crashes from scikit-learn

---

**5. Unicode-Aware Preprocessing**

* Custom tokenizer using Unicode regex
* Works with Indian scripts (Telugu/Hindi/etc.)
* No data loss even for complex glyphs

---

**6. Error Handling**

* Stops crashes when PDF text is empty
* Prevents TF-IDF empty-vocabulary error
* Falls back to raw text when tokenization fails
* Handles translation rate limits safely

---

**Tech Stack**

| Component          | Library             |
| ------------------ | ------------------- |
| Backend            | Flask               |
| Search Algorithms  | BM25, TF-IDF, FAISS |
| Language Detection | langdetect          |
| Translation        | deep-translator     |
| PDF Reader         | pdfminer.six        |
| ML Utilities       | scikit-learn        |
| Frontend Templates | HTML + Bootstrap    |

---

**Project Structure**

```
mini-search-engine/
│
├── src/
│   ├── app.py                 # Flask app
│   ├── search_engine.py       # Search Engine (BM25 / TF-IDF / FAISS)
│   ├── preprocess.py          # Text cleaning + tokenizing
│   ├── multilingual.py        # Language detection + translation
│   ├── pdf_reader.py          # PDF text extractor
│   └── templates/
│       └── index.html         # UI
│
├── uploads/                   # Uploaded PDFs
├── README.md
├── requirements.txt
└── venv/ (optional)
```

---

 **Installation**

**1. Clone the Repo**

```
git clone https://github.com/<your-username>/<repo-name>.git
cd mini-search-engine
```
 **2. Create Virtual Environment**

```
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

 **3. Install Dependencies**

```
pip install -r requirements.txt
```

---

 **How to Run the App**

Run the Flask app:

```
python -m src.app
```

Then open in browser:

```
http://127.0.0.1:5000
```

---

 **How to Use**

 **1️⃣ Upload PDFs**

You can upload one or multiple PDF files.

**2️⃣ Search**

Enter a query → select algorithm:

* BM25
* TF-IDF
* BM25+TF-IDF Fusion
* FAISS

 **3️⃣ View Results**

You will see:

* File name
* Detected language
* Snippet with <mark>highlighting</mark>
* Score
* Buttons to translate snippet or full text

**4️⃣ Translate**

Choose a target language → snippet and full document will be translated.

---

 **Supported Languages**

| Code  | Language  |
| ----- | --------- |
| en    | English   |
| te    | Telugu    |
| hi    | Hindi     |
| kn    | Kannada   |
| ml    | Malayalam |
| ta    | Tamil     |
| fr    | French    |
| es    | Spanish   |
| ja    | Japanese  |
| zh-CN | Chinese   |
| ar    | Arabic    |
| de    | German    |
| ru    | Russian   |

---

 **Use Cases**

* Academic paper search
* Multilingual government document search
* Legal/medical PDF retrieval
* Searching across Hindi/Telugu regional PDFs
* Resume or file search database
* Company document indexing

---

 **Known Issues & Fixes**

1️⃣ **Mixed-language PDFs**

**Problem:** langdetect gets confused
**Fix:** extract largest text block → detect from that only

 2️⃣ **Empty vocabulary error**

**Problem:** PDFs with stopwords-only cause TF-IDF crash
**Fix:** fallback to raw unicode tokens

3️⃣ **Wrong language detection**

**Problem:** short Hindi/Telugu text sometimes detected as `sv`
**Fix:** improved detection logic using 1000+ char window

 4️⃣ **Translation rate limits**

**Problem:** Google limiter
**Fix:** auto-retry + fallback text

---

 **License**

MIT License — Do not use, modify, and share.

---

**Contribute**

Pull requests are welcome.
Issues are welcome.
Feature suggestions are welcome.

---


