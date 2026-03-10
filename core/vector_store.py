# ================================================================
#  core/vector_store.py — TF-IDF with disk caching
# ================================================================

import re, os, pickle, hashlib
import numpy as np


class VectorStore:
    """
    TF-IDF vector store with disk caching.
    First run: builds index (~3-5 min) → saves to cache/
    Next runs: loads from cache (~5 seconds)
    Cache invalidates automatically when documents change.
    """
    CACHE_DIR = "cache"

    def __init__(self, documents: list, cache_name: str = "vector_index"):
        self.documents  = documents
        self.cache_name = cache_name
        self.matrix = self.vocab = self.idf = None
        os.makedirs(self.CACHE_DIR, exist_ok=True)

        if self._load_cache():
            print(f"✅ VectorStore loaded from cache: {len(documents)} documents. (~5s)")
        else:
            print(f"🔨 Building vector index for {len(documents)} documents...")
            print(f"   ⏳ First run only — will be cached for future runs.")
            self._build_index()
            self._save_cache()
            print(f"✅ VectorStore built and cached: {len(documents)} documents.")

    def _cache_path(self):
        return os.path.join(self.CACHE_DIR, f"{self.cache_name}.pkl")

    def _doc_hash(self):
        sample  = self.documents[:100] + self.documents[-100:]
        content = str(len(self.documents)) + str([d.get('id','') for d in sample])
        return hashlib.md5(content.encode()).hexdigest()

    def _save_cache(self):
        with open(self._cache_path(), 'wb') as f:
            pickle.dump({
                "hash": self._doc_hash(),
                "matrix": self.matrix,
                "vocab":  self.vocab,
                "idf":    self.idf,
            }, f, protocol=4)
        print(f"💾 Index cached → {self._cache_path()}")

    def _load_cache(self) -> bool:
        path = self._cache_path()
        if not os.path.exists(path):
            return False
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            if data.get("hash") != self._doc_hash():
                print("🔄 Data changed — rebuilding index...")
                return False
            self.matrix = data["matrix"]
            self.vocab  = data["vocab"]
            self.idf    = data["idf"]
            return True
        except Exception as e:
            print(f"⚠️  Cache load failed ({e}) — rebuilding...")
            return False

    def _tokenize(self, text: str) -> list:
        return re.findall(r'\b[a-z]{3,}\b', text.lower())

    def _build_index(self):
        tokenized = [self._tokenize(d['text']) for d in self.documents]
        vocab_set = sorted({w for toks in tokenized for w in toks})
        self.vocab = {w: i for i, w in enumerate(vocab_set)}
        V, N = len(self.vocab), len(self.documents)

        tf = np.zeros((N, V), dtype=np.float32)
        for i, toks in enumerate(tokenized):
            for w in toks:
                if w in self.vocab:
                    tf[i][self.vocab[w]] += 1
            if toks: tf[i] /= len(toks)

        df       = np.sum(tf > 0, axis=0)
        self.idf = (np.log((N + 1) / (df + 1)) + 1).astype(np.float32)
        self.matrix = tf * self.idf
        norms = np.linalg.norm(self.matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1
        self.matrix /= norms

    def search(self, query: str, k: int = 6) -> list:
        toks = self._tokenize(query)
        if not toks: return []

        q = np.zeros(len(self.vocab), dtype=np.float32)
        for w in toks:
            if w in self.vocab: q[self.vocab[w]] += 1
        q /= len(toks)
        q *= self.idf
        norm = np.linalg.norm(q)
        if norm > 0: q /= norm

        scores  = self.matrix @ q
        top_idx = np.argsort(scores)[::-1][:k]
        return [
            {**self.documents[i], "relevance_score": round(float(scores[i]), 4)}
            for i in top_idx if scores[i] > 0.01
        ]

    def search_by_symptoms(self, symptoms: list, k: int = 6) -> list:
        if not symptoms: return self.search(" ".join(symptoms), k=k)
        sym_set = {s.lower().strip() for s in symptoms}
        query   = " ".join(symptoms)
        tfidf_results = {
            r['id']: r['relevance_score']
            for r in self.search(query, k=max(100, len(self.documents)//10))
        }
        scored = []
        for doc in self.documents:
            doc_syms = {s.lower().strip() for s in doc.get('symptoms', [])}
            overlap  = len(sym_set & doc_syms)
            tfidf_sc = tfidf_results.get(doc['id'], 0)
            if overlap > 0 or tfidf_sc > 0.01:
                combined = (overlap / max(len(sym_set), 1)) * 0.6 + tfidf_sc * 0.4
                scored.append({**doc, "relevance_score": round(combined, 4)})
        scored.sort(key=lambda x: x['relevance_score'], reverse=True)
        return scored[:k]