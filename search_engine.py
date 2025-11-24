# src/search_engine.py
import os
import math
import numpy as np
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as sk_normalize
from rank_bm25 import BM25Okapi
import faiss

from src.preprocess import normalize_text, tokens_to_string


class SuperSearchEngine:
    def __init__(self, use_bm25=True, use_faiss=True):
        self.use_bm25 = use_bm25
        self.use_faiss = use_faiss

        # document storage
        self.doc_texts: Dict[str, str] = {}
        self.doc_titles: Dict[str, str] = {}
        self.doc_ids: List[str] = []

        # corpus
        self.corpus_tokens: List[List[str]] = []
        self.corpus_strings: List[str] = []

        # TF-IDF
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None

        # BM25
        self.bm25: BM25Okapi = None

        # FAISS
        self.faiss_index = None
        self.faiss_dim = 0

    # ---------------------------------------------------------
    # DOCUMENT ADD (NO LANGUAGE DETECTION HERE)
    # ---------------------------------------------------------
    def add_document(self, doc_id: str, text: str, title: str = None):
        self.doc_texts[doc_id] = text
        self.doc_titles[doc_id] = title or doc_id

        if doc_id not in self.doc_ids:
            self.doc_ids.append(doc_id)

    # ---------------------------------------------------------
    # BUILD INDEX
    # ---------------------------------------------------------
    def build_index(self, lemmatize=True, ngram_range=(1, 1), max_features=None):
        self.corpus_tokens = []
        self.corpus_strings = []

        for doc_id in self.doc_ids:
            txt = self.doc_texts.get(doc_id, "")
            tokens = normalize_text(txt, lemmatize)
            self.corpus_tokens.append(tokens)
            self.corpus_strings.append(tokens_to_string(tokens))

        # TF-IDF
        self.tfidf_vectorizer = TfidfVectorizer(
            preprocessor=lambda s: s,
            tokenizer=lambda s: s.split(),
            ngram_range=ngram_range,
            max_features=max_features
        )

        if len(self.corpus_strings) == 0:
            self.tfidf_matrix = None
        else:
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.corpus_strings)
            self.tfidf_matrix = sk_normalize(self.tfidf_matrix, norm='l2', axis=1)

        # BM25
        if self.use_bm25 and len(self.corpus_tokens) > 0:
            self.bm25 = BM25Okapi(self.corpus_tokens)

        # FAISS
        if self.use_faiss and self.tfidf_matrix is not None:
            X = self.tfidf_matrix.toarray().astype('float32')
            n_docs, dim = X.shape
            self.faiss_dim = dim

            if dim == 0:
                self.faiss_index = None
            else:
                index = faiss.IndexFlatIP(dim)
                if n_docs > 0:
                    index.add(X)
                self.faiss_index = index

    # ---------------------------------------------------------
    # TF-IDF
    # ---------------------------------------------------------
    def _query_tfidf(self, query, top_k=10):
        if self.tfidf_vectorizer is None or self.tfidf_matrix is None:
            return []

        q_tokens = normalize_text(query)
        q_str = tokens_to_string(q_tokens)

        q_vec = self.tfidf_vectorizer.transform([q_str])
        q_vec = sk_normalize(q_vec, norm="l2", axis=1)

        sims = (q_vec @ self.tfidf_matrix.T).toarray().flatten()

        result = [(self.doc_ids[i], float(sims[i])) for i in range(len(self.doc_ids))]
        result.sort(key=lambda x: x[1], reverse=True)
        return result[:top_k]

    # ---------------------------------------------------------
    # BM25
    # ---------------------------------------------------------
    def _query_bm25(self, query, top_k=10):
        if not self.bm25:
            return []
        q_tokens = normalize_text(query)
        scores = self.bm25.get_scores(q_tokens)
        result = [(self.doc_ids[i], float(scores[i])) for i in range(len(scores))]
        result.sort(key=lambda x: x[1], reverse=True)
        return result[:top_k]

    # ---------------------------------------------------------
    # FAISS
    # ---------------------------------------------------------
    def _query_faiss(self, query, top_k=10):
        if self.faiss_index is None:
            return []
        q_tokens = normalize_text(query)
        q_str = tokens_to_string(q_tokens)

        q_vec = self.tfidf_vectorizer.transform([q_str])
        q_vec = sk_normalize(q_vec, norm="l2", axis=1).toarray().astype("float32")

        D, I = self.faiss_index.search(q_vec, top_k)

        result = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(self.doc_ids):
                continue
            result.append((self.doc_ids[idx], float(score)))
        return result

    # ---------------------------------------------------------
    # SEARCH
    # ---------------------------------------------------------
    def search(self, query, mode="tfidf", top_k=10, combine_bm25_tfidf=False):

        if mode == "tfidf":
            scored = self._query_tfidf(query, top_k if not combine_bm25_tfidf else len(self.doc_ids))
        elif mode == "bm25":
            scored = self._query_bm25(query, top_k if not combine_bm25_tfidf else len(self.doc_ids))
        elif mode == "faiss":
            scored = self._query_faiss(query, top_k)
        else:
            raise ValueError("Invalid search mode")

        # BM25 + TF-IDF fusion
        if combine_bm25_tfidf and self.bm25 and mode in ("tfidf", "bm25"):
            tfidf_scores = dict(self._query_tfidf(query, len(self.doc_ids)))
            bm25_scores = dict(self._query_bm25(query, len(self.doc_ids)))

            alpha = 0.6
            combined_list = []
            for doc_id in self.doc_ids:
                combined = alpha * tfidf_scores.get(doc_id, 0) + (1 - alpha) * bm25_scores.get(doc_id, 0)
                combined_list.append((doc_id, combined))

            combined_list.sort(key=lambda x: x[1], reverse=True)
            scored = combined_list[:top_k]

        # Prepare results
        final = []
        for doc_id, score in scored[:top_k]:
            text = self.doc_texts.get(doc_id, "")
            snippet = self._make_snippet(text, query)

            final.append({
                "id": doc_id,
                "title": self.doc_titles.get(doc_id, doc_id),
                "score": float(score),
                "snippet": snippet
            })

        return final

    # ---------------------------------------------------------
    # SNIPPET MAKER
    # ---------------------------------------------------------
    def _make_snippet(self, text: str, query: str, snippet_len=300):

        tokens = normalize_text(query)
        if not tokens:
            snippet = text[:snippet_len]
            return snippet + ("..." if len(text) > snippet_len else "")

        lower = text.lower()
        idx = -1

        # locate any matching keyword
        for t in tokens:
            pos = lower.find(t.lower())
            if pos != -1:
                idx = pos
                break

        # Case 1: No match
        if idx == -1:
            start = 0
            end = min(len(text), snippet_len)
            snippet = text[start:end]

        # Case 2: Found
        else:
            start = max(0, idx - 80)
            end = min(len(text), idx + snippet_len - 80)
            snippet = text[start:end]

        # Highlight tokens
        import re
        for t in set(tokens):
            pattern = re.compile(re.escape(t), re.IGNORECASE)
            snippet = pattern.sub(lambda m: f"<mark>{m.group(0)}</mark>", snippet)

        if end < len(text):
            snippet += "..."

        return snippet
