import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import pickle


class FaissIndexer:
    def __init__(self, index_path="data/faiss_index.bin", map_path="data/faiss_map.pkl"):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        self.index_path = index_path
        self.map_path = map_path

        # Mapping: vector_id → (pdf_name, chunk_text)
        self.doc_map = []

        # Load index if exists
        if os.path.exists(index_path) and os.path.exists(map_path):
            self._load()
        else:
            self.index = faiss.IndexFlatIP(384)  # Inner product = cosine sim with normalized vectors

    # -------------------------
    # Normalize vectors
    # -------------------------
    def _normalize(self, vecs):
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / norms

    # -------------------------
    # Add text chunks to FAISS
    # -------------------------
    def add_documents(self, pdf_name, chunks):
        """
        chunks = list of text segments extracted from PDF
        """
        embeddings = self.model.encode(chunks, convert_to_numpy=True)
        embeddings = self._normalize(embeddings)

        # Add to FAISS
        self.index.add(embeddings)

        # Update doc map
        for chunk in chunks:
            self.doc_map.append((pdf_name, chunk))

        # Save after update
        self._save()

    # -------------------------
    # Search top-k matches
    # -------------------------
    def search(self, query, top_k=5):
        query_vec = self.model.encode([query], convert_to_numpy=True)
        query_vec = self._normalize(query_vec)

        scores, indices = self.index.search(query_vec, top_k)
        scores = scores[0]
        indices = indices[0]

        results = []
        for score, idx in zip(scores, indices):
            if idx == -1:
                continue

            pdf_name, text = self.doc_map[idx]
            results.append({
                "pdf": pdf_name,
                "text": text,
                "score": float(score)
            })

        # Sort results by similarity
        results = sorted(results, key=lambda x: x["score"], reverse=True)
        return results

    # -------------------------
    # Save index & mapping
    # -------------------------
    def _save(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.map_path, "wb") as f:
            pickle.dump(self.doc_map, f)

    # -------------------------
    # Load index & mapping
    # -------------------------
    def _load(self):
        self.index = faiss.read_index(self.index_path)
        with open(self.map_path, "rb") as f:
            self.doc_map = pickle.load(f)
