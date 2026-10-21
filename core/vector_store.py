# ================================================================
#  core/vector_store.py — TF-IDF with disk caching
#
#  [P3] Source Locking upgrades:
#  - RELEVANCE_THRESHOLD raised 0.01 → 0.10
#  - CATEGORY_BLOCKLIST: hard-excludes irrelevant categories
#    (Oncology, Genetics, Endocrinology, etc.) unless query
#    contains explicit override keywords
#  - QUERY_CATEGORY_ROUTES: routes SYMPTOM/MEDICATION/etc. queries
#    to only the relevant source categories
#  - Both search() and search_by_symptoms() honour all three layers
# ================================================================

import re, os, sys, pickle, hashlib
from collections import Counter
import numpy as np
import scipy.sparse as sp

# Ensure standard output and error support Unicode on Windows terminals
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

# ── Text-level hard filter for congenital/neonatal conditions ──────────────
# Catches ICD-10 Q/P chunks even if their category label slips past the blocklist.
_CONGENITAL_RE = re.compile(
    r"\b(cleft palate|cleft lip|congenital|neonatal|newborn|chromosom"
    r"|down syndrome|spina bifida|hirschsprung|phenylketonuria"
    r"|cystic fibrosis|turner syndrome|klinefelter|trisomy)\b",
    re.IGNORECASE,
)


class VectorStore:
    """
    TF-IDF sparse vector store with versioned disk caching.
    Uses SciPy CSR sparse representation to eliminate multi-gigabyte dense matrices.
    First run: builds sparse index in a few seconds → saves versioned cache.
    Next runs: loads from versioned cache in sub-second time.
    Cache invalidates automatically when documents change or format version mismatches.
    """
    CACHE_DIR = "cache"
    CACHE_VERSION = 2
    DEFAULT_CACHE_NAME = "vector_index_v2"

    def __init__(self, documents: list, cache_name: str = DEFAULT_CACHE_NAME):
        self.documents  = documents
        self.cache_name = cache_name
        self.matrix = self.vocab = self.idf = None
        os.makedirs(self.CACHE_DIR, exist_ok=True)

        if self._load_cache():
            print(f"✅ VectorStore loaded from cache (v{self.CACHE_VERSION} sparse): {len(documents)} documents.")
        else:
            print(f"🔨 Building sparse vector index for {len(documents)} documents...")
            self._build_index()
            if self.documents or self.cache_name != self.DEFAULT_CACHE_NAME:
                self._save_cache()
            print(f"✅ VectorStore built and cached (v{self.CACHE_VERSION} sparse): {len(documents)} documents.")

    def _cache_path(self):
        return os.path.join(self.CACHE_DIR, f"{self.cache_name}.pkl")

    def _doc_hash(self):
        sample  = self.documents[:100] + self.documents[-100:]
        content = str(len(self.documents)) + str([d.get('id','') for d in sample])
        return hashlib.md5(content.encode()).hexdigest()

    def _save_cache(self):
        with open(self._cache_path(), 'wb') as f:
            pickle.dump({
                "version": self.CACHE_VERSION,
                "format":  "scipy_csr_v2",
                "hash":    self._doc_hash(),
                "matrix":  self.matrix,
                "vocab":   self.vocab,
                "idf":     self.idf,
            }, f, protocol=4)
        print(f"💾 Index cached → {self._cache_path()}")

    def _load_cache(self) -> bool:
        path = self._cache_path()
        if not os.path.exists(path):
            return False
        # Safety guard: legacy dense cache files are several gigabytes.
        # Reject immediately without deserializing to prevent Out-Of-Memory.
        file_size = os.path.getsize(path)
        if file_size > 500 * 1024 * 1024:
            print(f"⚠️  Legacy multi-gigabyte cache file detected ({file_size / (1024*1024):.0f} MB) — rejecting to prevent OOM and rebuilding sparse index...")
            return False
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            # Version check — reject incompatible or legacy cache format
            if data.get("version") != self.CACHE_VERSION:
                print(f"⚠️  Incompatible or legacy cache format detected (expected v{self.CACHE_VERSION}) — rebuilding sparse index...")
                return False
            if data.get("hash") != self._doc_hash():
                print("🔄 Data changed — rebuilding index...")
                return False
            mat = data.get("matrix")
            if not isinstance(mat, sp.csr_matrix):
                print("⚠️  Invalid matrix type in cache (expected scipy.sparse.csr_matrix) — rebuilding...")
                return False
            self.matrix = mat
            self.vocab  = data["vocab"]
            self.idf    = data["idf"]
            return True
        except Exception as e:
            print(f"⚠️  Cache load failed ({e}) — rebuilding...")
            return False

    def _tokenize(self, text: str) -> list:
        return re.findall(r'\b[a-z]{3,}\b', text.lower())

    def _build_index(self):
        tokenized = [self._tokenize(d.get('text', '')) for d in self.documents]
        df_counts = Counter()
        for toks in tokenized:
            df_counts.update(set(toks))

        vocab_list = sorted(df_counts.keys())
        self.vocab = {w: i for i, w in enumerate(vocab_list)}
        V, N = len(self.vocab), len(self.documents)

        df_arr = np.array([df_counts[w] for w in vocab_list], dtype=np.float32)
        self.idf = (np.log((N + 1) / (df_arr + 1)) + 1).astype(np.float32)

        indptr = [0]
        indices = []
        data = []

        for toks in tokenized:
            if not toks:
                indptr.append(len(indices))
                continue
            tok_counts = Counter(toks)
            items = sorted((self.vocab[w], cnt) for w, cnt in tok_counts.items() if w in self.vocab)
            if not items:
                indptr.append(len(indices))
                continue

            doc_len = float(len(toks))
            row_cols = [col for col, _ in items]
            row_vals = np.array([(cnt / doc_len) * self.idf[col] for col, cnt in items], dtype=np.float32)
            row_norm = float(np.linalg.norm(row_vals))
            if row_norm > 0:
                row_vals /= row_norm

            indices.extend(row_cols)
            data.extend(row_vals)
            indptr.append(len(indices))

        self.matrix = sp.csr_matrix(
            (np.array(data, dtype=np.float32),
             np.array(indices, dtype=np.int32),
             np.array(indptr, dtype=np.int32)),
            shape=(N, V)
        )

    # ── [P3 — Source Locking] ─────────────────────────────────────────────

    # 0.10 — the blocklist now handles category-level exclusions.
    # The threshold only needs to cut truly zero-signal chunks.
    # 0.15 was cutting valid General Medicine chunks on short queries.
    RELEVANCE_THRESHOLD = 0.10

    # Max chunks from the same category per retrieval call.
    MAX_PER_CATEGORY = 2

    # [P3-BLOCKLIST] Hard-exclude these categories unless the query contains
    # an explicit override keyword.  Prevents "air" → Air Pollution,
    # "growth" → Endocrinology, "rare" → Genetics, etc.
    CATEGORY_BLOCKLIST = {
        "Oncology",
        "Genetics / Rare Diseases",
        "Endocrinology",
        "Poisoning, Toxicology, Environmental Health",
        "Older Adults",
        "Pediatrics",
        "Complementary and Alternative Medicine",
        "Fluid and Electrolyte Disorders",
        # ICD-10 prefix groups that produce irrelevant results
        "Congenital",          # ICD-10 Q-codes: cleft palate, Down syndrome, etc.
        "Pediatrics",          # ICD-10 P-codes: newborn/neonatal conditions
        "Poisoning / Toxicology",  # ICD-10 T-codes variant label
        "ENT",                 # Block ENT unless query is explicitly ear/nose/throat
        "Ophthalmology",       # Block eye specialty unless query mentions eyes
        "Dermatology",         # Block skin specialty unless query mentions skin
    }

    # Keywords that unlock a blocklisted category when present in the query.
    BLOCKLIST_OVERRIDES = {
        "Oncology": [
            "cancer", "tumour", "tumor", "malignant", "carcinoma",
            "lymphoma", "leukaemia", "leukemia", "oncology",
            "chemotherapy", "biopsy",
        ],
        "Genetics / Rare Diseases": [
            "genetic", "hereditary", "chromosome",
            "inherited", "rare disease", "congenital",
        ],
        "Endocrinology": [
            "thyroid", "diabetes", "insulin", "hormone",
            "cortisol", "adrenal", "pituitary", "endocrine",
        ],
        "Poisoning, Toxicology, Environmental Health": [
            "poison", "toxic", "overdose", "chemical",
            "carbon monoxide", "lead poisoning",
        ],
        "ENT": [
            "ear", "nose", "throat", "sinus", "hearing", "tonsil",
            "nasal", "rhinitis", "otitis", "laryngitis", "pharyngitis",
            "snoring", "tinnitus", "vertigo", "hoarse", "adenoid",
        ],
        "Ophthalmology": [
            "eye", "vision", "sight", "blind", "retina", "cornea",
            "glaucoma", "cataract", "conjunctivitis", "pupil",
        ],
        "Dermatology": [
            "skin", "rash", "itch", "acne", "eczema", "psoriasis",
            "hives", "lesion", "mole", "dermatitis", "wound",
        ],
        "Older Adults": [
            "elderly", "geriatric", "nursing home",
            "dementia", "alzheimer",
        ],
    }

    # [P3-ROUTING] Maps query_category (from REASONING_PROMPT) to the
    # set of allowed source categories for that query type.
    # None = no restriction (open search).
    # [P3-ROUTING] For SYMPTOM queries the blocklist alone is sufficient —
    # whitelisting specific category strings is fragile because MedQuAD uses
    # inconsistent category labels across CSVs.  Setting SYMPTOM → None means
    # "allow everything not on the blocklist", which is the correct behaviour.
    # Narrow whitelists are only useful for MEDICATION / WELLNESS where we
    # want to actively exclude clinical specialty chunks.
    QUERY_CATEGORY_ROUTES = {
        "SYMPTOM":      None,    # open — blocklist handles exclusions
        "MEDICATION":   None,    # open — drug info spans many categories
        "CONDITION":    None,    # open — condition queries span all
        "WELLNESS":     None,    # open
        "MENTAL_HEALTH": None,   # open
        "EMERGENCY":    None,    # open — emergency needs all sources
        "UNKNOWN":      None,    # open fallback
    }

    # ── Internal helpers ──────────────────────────────────────────────────

    def _is_blocklisted(self, category: str, query_lower: str) -> bool:
        """Return True if this category is blocked for this query."""
        if category not in self.CATEGORY_BLOCKLIST:
            return False
        overrides = self.BLOCKLIST_OVERRIDES.get(category, [])
        return not any(kw in query_lower for kw in overrides)

    def _allowed_by_route(self, category: str, allowed_set) -> bool:
        """Return True if category is within the routed allowed set."""
        if allowed_set is None:
            return True
        return category in allowed_set

    def _filter_doc(self, doc: dict, query_lower: str, allowed_set) -> bool:
        """Return True if this doc should be kept (passes blocklist + route)."""
        category = doc.get('category', 'General')
        if self._is_blocklisted(category, query_lower):
            return False
        if not self._allowed_by_route(category, allowed_set):
            return False
        # Text-level safety net: block congenital/neonatal chunks regardless of category label
        if _CONGENITAL_RE.search(doc.get('text', '')[:300]):
            return False
        return True

    # ── PUBLIC SEARCH METHODS ─────────────────────────────────────────────

    def search(self, query: str, k: int = 6,
               query_category: str = None) -> list:
        """
        TF-IDF cosine search with three-layer source locking:
          1. RELEVANCE_THRESHOLD — score floor (0.15)
          2. CATEGORY_BLOCKLIST  — hard-exclude irrelevant categories
          3. QUERY_CATEGORY_ROUTES — only return categories relevant to
             the query type (SYMPTOM / MEDICATION / CONDITION / etc.)

        query_category: pass the value from REASONING_PROMPT for routing.
        If None, routing is skipped (only blocklist + threshold apply).
        """
        toks = self._tokenize(query)
        if not toks:
            return []

        query_lower = query.lower()
        allowed_set = self.QUERY_CATEGORY_ROUTES.get(
            query_category or "UNKNOWN"
        )  # None = open

        q = np.zeros(len(self.vocab), dtype=np.float32)
        for w in toks:
            if w in self.vocab:
                q[self.vocab[w]] += 1
        q /= len(toks)
        q *= self.idf
        norm = np.linalg.norm(q)
        if norm > 0:
            q /= norm

        scores = self.matrix.dot(q)

        # Pull k*8 candidates — diversity filtering will shrink this down
        candidate_idx = np.argsort(scores)[::-1][:k * 8]

        results: list = []
        category_counts: dict = {}

        for i in candidate_idx:
            score = float(scores[i])

            # Hard score threshold — break because scores are sorted descending
            if score < self.RELEVANCE_THRESHOLD:
                break

            doc = self.documents[i]

            # Blocklist + route filter
            if not self._filter_doc(doc, query_lower, allowed_set):
                continue

            category = doc.get('category', 'General')

            # Diversity cap
            if category_counts.get(category, 0) >= self.MAX_PER_CATEGORY:
                continue

            category_counts[category] = category_counts.get(category, 0) + 1
            results.append({**doc, "relevance_score": round(score, 4)})

            if len(results) == k:
                break

        # ── Graceful degradation ──────────────────────────────────────────
        # If strict pass yields < 3 results, relax score floor but keep
        # blocklist + routing.  Prevents empty context on obscure queries.
        if len(results) < 3:
            results = []
            category_counts = {}
            for i in np.argsort(scores)[::-1][:k * 6]:
                score = float(scores[i])
                if score <= 0:
                    break
                doc = self.documents[i]
                if not self._filter_doc(doc, query_lower, allowed_set):
                    continue
                category = doc.get('category', 'General')
                if category_counts.get(category, 0) >= self.MAX_PER_CATEGORY:
                    continue
                category_counts[category] = category_counts.get(category, 0) + 1
                results.append({**doc, "relevance_score": round(score, 4)})
                if len(results) == k:
                    break

        return results

    def search_by_symptoms(self, symptoms: list, k: int = 6,
                           query_category: str = None) -> list:
        """
        Symptom-overlap + TF-IDF hybrid search with source locking.
        Passes query_category through to search() for routing.
        """
        # Guard: if symptoms list is genuinely empty, fall back to plain TF-IDF
        # using the full conversation text passed in (empty join would return nothing)
        if not symptoms:
            return self.search("general symptom assessment", k=k,
                               query_category=query_category or "SYMPTOM")

        sym_set     = {s.lower().strip() for s in symptoms}
        query       = " ".join(symptoms)
        query_lower = query.lower()
        allowed_set = self.QUERY_CATEGORY_ROUTES.get(
            query_category or "SYMPTOM"  # default SYMPTOM for symptom searches
        )

        tfidf_results = {
            r.get('id', r.get('text','')[:40]): r['relevance_score']
            for r in self.search(query, k=max(100, len(self.documents) // 10),
                                 query_category=query_category)
        }

        scored = []
        for doc in self.documents:
            # Blocklist + route filter applied before scoring
            if not self._filter_doc(doc, query_lower, allowed_set):
                continue

            doc_syms = {s.lower().strip() for s in doc.get('symptoms', [])}
            overlap  = len(sym_set & doc_syms)
            tfidf_sc = tfidf_results.get(doc.get('id', doc.get('text','')[:40]), 0)

            if overlap > 0 or tfidf_sc > self.RELEVANCE_THRESHOLD:
                # Normalize by BOTH query and doc symptom counts.
                # A 1/10 query coverage no longer scores the same as 1/1.
                n_query   = max(len(sym_set), 1)
                n_doc     = max(len(doc_syms), 1) if doc_syms else 1
                coverage  = overlap / n_query          # recall-side
                precision = overlap / n_doc            # precision-side
                sym_score = (coverage + precision) / 2 # F1-inspired
                combined  = sym_score * 0.6 + tfidf_sc * 0.4
                scored.append({**doc, "relevance_score": round(combined, 4)})

        scored.sort(key=lambda x: x['relevance_score'], reverse=True)

        # Diversity cap
        results: list = []
        category_counts: dict = {}
        for doc in scored:
            category = doc.get('category', 'General')
            if category_counts.get(category, 0) >= self.MAX_PER_CATEGORY:
                continue
            category_counts[category] = category_counts.get(category, 0) + 1
            results.append(doc)
            if len(results) == k:
                break

        return results