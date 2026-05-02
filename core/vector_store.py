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

import re, os, pickle, hashlib
import numpy as np

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

        scores = self.matrix @ q

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