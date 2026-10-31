<div align="center">

# 🏥 Care-AI

### Multilingual Healthcare Information Assistant with RAG, Explainable AI, and Multi-Provider LLM Routing

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.12-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.42+-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Architecture](https://img.shields.io/badge/Architecture-Sparse%20RAG-008080?style=flat-square)](https://scipy.org/)
[![XAI](https://img.shields.io/badge/XAI-Explainable%20Clinical%20Reasoning-6B46C1?style=flat-square)](#core-features)
[![Deployment](https://img.shields.io/badge/Deployment-Streamlit%20Cloud-2563EB?style=flat-square&logo=streamlit&logoColor=white)](https://care-ai-health.streamlit.app/)
[![License](https://img.shields.io/badge/Copyright-All%20Rights%20Reserved-gray?style=flat-square)](#copyright-and-licensing)

<p align="center">
  <a href="https://care-ai-health.streamlit.app/"><strong>Launch Live Application</strong></a> •
  <a href="https://github.com/vinayjoseph003/care-ai"><strong>GitHub Repository</strong></a> •
  <a href="#quick-start"><strong>Quick Start</strong></a> •
  <a href="#system-architecture"><strong>Architecture</strong></a> •
  <a href="#empirical-evaluation"><strong>Evaluation</strong></a>
</p>

</div>

---

> [!IMPORTANT]
> **Academic Research Prototype & Healthcare Safety Notice**<br>
> Care-AI is an experimental academic research prototype developed as a final-year engineering project. It is designed to evaluate retrieval-augmented language models, multilingual communication, and explainable AI in healthcare triage workflows. **Care-AI is not a certified medical device, does not provide clinical diagnoses or treatment plans, and is not a substitute for professional medical judgment.** Never disregard or delay professional clinical evaluation because of information provided by this system. For life-threatening emergencies, immediately contact local emergency medical services.

---

## Table of Contents

- [Overview](#overview)
- [Live Application](#live-application)
- [Core Features](#core-features)
- [System Architecture](#system-architecture)
- [Multi-Provider LLM Routing & Resilience](#multi-provider-llm-routing--resilience)
- [Knowledge Base & Ingestion Pipeline](#knowledge-base--ingestion-pipeline)
- [Engineering Improvements](#engineering-improvements)
- [Evaluation](#evaluation)
- [Workflow Latency Characteristics](#workflow-latency-characteristics)
- [Quick Start](#quick-start)
- [CLI Reference](#cli-reference)
- [Verification & Automated Testing](#verification--automated-testing)
- [Project Structure](#project-structure)
- [Cache Maintenance](#cache-maintenance)
- [Privacy and Limitations](#privacy-and-limitations)
- [Copyright and Licensing](#copyright-and-licensing)
- [Authors & Acknowledgments](#authors--acknowledgments)

---

## Overview

**Care-AI** is an academic healthcare conversational AI prototype engineered to investigate how **Retrieval-Augmented Generation (RAG)**, **Explainable AI (XAI)**, and **resilient multi-provider large language model (LLM) routing** can deliver grounded, traceable, and language-adaptive symptom triage and healthcare information.

Modern healthcare information assistants often struggle with hallucinated medical claims, opaque reasoning chains, language translation barriers, and cloud service disruptions. Care-AI addresses these challenges through a modular 10-pass pipeline:

1. **Multilingual Ingestion & Response**: Automatically detects the user's language (including English, Telugu, Hindi, and other regional dialects) and ensures cultural and linguistic coherence throughout the diagnostic intake.
2. **Deterministic & Semantic Grounding**: Pairs structured symptom datasets with a 37,233-chunk curated clinical corpus using a lightweight SciPy CSR sparse TF-IDF retrieval index.
3. **Structured Clinical Intake (SOCRATES)**: Dynamically gathers symptom history (site, onset, character, radiation, associations, timing, exacerbating factors, severity) before formulating clinical assessments.
4. **Explainable AI (XAI) & Fact-Checking**: Employs claim-level verification and a *cause → clinical reasoning → actionable recommendation* format to trace each conclusion back to indexed medical literature.
5. **Multi-Tier Fault Tolerance**: Routes requests dynamically across leading cloud and local LLM engines (Google Gemini Flash Lite, OpenRouter Llama 3.3 70B, Groq, and local Ollama) with instantaneous failover upon rate limits or provider downtime.

Care-AI is strictly an academic research milestone intended to demonstrate engineering feasibility, software resilience, and verifiable inference for informational triage.

---

## Live Application

The hosted demonstration of Care-AI is accessible on Streamlit Community Cloud:

👉 **[https://care-ai-health.streamlit.app/](https://care-ai-health.streamlit.app/)**

### Hosted Environment Notes
- **Infrastructure**: Hosted on Streamlit Community Cloud containerized runtime.
- **Provider Staging**: Configured with primary routing to Google Gemini (`gemini-3.5-flash-lite`) and secondary automated fallback to OpenRouter (`meta-llama/llama-3.3-70b-instruct`).
- **Scope & Security**: Deployment serves as an interactive demonstration of technical accessibility and user experience. It is not an enterprise-hardened medical portal and must not be used to enter Protected Health Information (PHI) or real patient identifiers.

---

## Core Features

| Feature | Technical Implementation | Verified Capability |
|---|---|---|
| **Multilingual Interaction** | Language identification pass with strict cross-prompt target-language constraints | Automatically recognizes input languages (English, Telugu, Hindi, and others) and maintains consistent target-language output across all triage passes. |
| **Sparse Vector RAG** | SciPy CSR sparse TF-IDF index over 37,233 chunks with category blocklists and query routing | Performs sparse vector retrieval across 5 authoritative medical sources, enforcing relevance filtering (`RELEVANCE_THRESHOLD = 0.10`). |
| **Explainable AI (XAI)** | Dual-pass structured generation: *Cause → Reasoning → Recommendation* | Grounds each response in referenced literature (`MedQuAD`, `MedlinePlus`, `ICD-10-CM`, `NHS`), displaying explicit source provenance for transparent inspection. |
| **Four-Tier Symptom Triage** | Weighted mathematical severity aggregation + differential LLM consensus | Stratifies risk into `SELF_CARE`, `SEE_DOCTOR`, `URGENT_CARE`, or `EMERGENCY`. Implements deterministic red-flag escalation and conservative triage floors to minimize under-triage risk for emergency presentations. |
| **SOCRATES Intake Engine** | Confidence-threshold conversational state machine (`ConvState`) | Replaces rigid turn limits with adaptive follow-up questions; stops when sufficient clinical detail is collected or when vague user input is identified. |
| **Doctor & Specialist Finder** | Geolocation + OpenStreetMap (Nominatim/Overpass) specialist mapping | Automatically recommends relevant medical specialties (e.g., Cardiologist, Pulmonologist, Neurologist) and queries nearby medical facilities. |
| **Multi-Provider LLM Router** | Sticky priority router across Gemini, OpenRouter, Groq, and local Ollama | Bounded timeouts, non-blocking failover on 429/503 errors, safe response parsing, and native Server-Sent Events (SSE) token streaming. |
| **Per-Session State Isolation** | Strict separation between `@st.cache_resource` and `st.session_state` | Heavy knowledge assets and indexes are loaded once globally into shared memory, while conversation history, user language, and triage state remain strictly sandboxed per session. |
| **Defensive Dynamic Rendering** | Content sanitization via `html.escape` and controlled error formatting | Strips untrusted markup and script tags from dynamic chat cards, preventing HTML injection and obfuscating raw API stack traces. |
| **Session Logging & Export** | Structured JSON logging of diagnostic passes and XAI justifications | Enables full audit trails of diagnostic decisions, severity weights, and retrieval provenance in both UI and CLI interfaces. |

---

## System Architecture

Care-AI is engineered around an **Inquiry-First, Verify-Later (IFVL)** 10-pass pipeline that decouples shared immutable assets from per-session mutable conversation context.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             USER INTERACTION                                │
│       Streamlit Cloud Browser UI  │  Terminal CLI (main.py)                 │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          CAREAI INTAKE CONTROLLER                           │
│  [Pass 1] Language Detection & Normalization                                │
│  [Pass 2] SOCRATES Clinical History Gathering (Confidence-Threshold Gated)  │
│  [Pass 2.5] Local Query Pre-classification (Zero-Inference Fast Path)       │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                   RETRIEVAL & CLINICAL REASONING ENGINE                     │
│  [Pass 3] Sparse TF-IDF Vector Retrieval (SciPy CSR, 37,233 chunks)         │
│  [Pass 4] Quantitative Symptom Severity Aggregation (Weights 1–7)           │
│  [Pass 5] Differential Diagnosis Formulation (Two-Pass Self-Consistency)    │
│  [Pass 6] Streamed Generation (Strict 80–120 Word Target-Language Cap)      │
│  [Pass 7] Sentence-Level Claim Fact-Checking (Hallucination Control)        │
│  [Pass 8] Structured XAI Explanation (Cause → Reasoning → Recommendation)   │
│  [Pass 9] Non-Pharmacological Treatment & Precaution Guidance               │
│  [Pass 10] Emergency Red-Flag Tripping & Mental Health Check-In             │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MULTI-PROVIDER LLM ROUTER                           │
│  Primary: Gemini Flash Lite  │  Fallback: OpenRouter Llama 3.3 70B          │
│  Tertiary: Groq Fast API     │  Local Fallback: Ollama (Bounded Probe)       │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Architectural Diagram
The complete component relationships across the ingestion, retrieval, and runtime layers are illustrated below:

<div align="center">
  <img width="900" alt="Care-AI High-Level Architecture" src="https://github.com/user-attachments/assets/d354f1f1-3ef8-447c-baa9-47223dc2361f" />
</div>

---

## Multi-Provider LLM Routing & Resilience

High availability is essential for healthcare assistants. Care-AI incorporates an intelligent, sticky routing engine (`core/llm_router.py`) that minimizes latency, avoids blocking sleeps, and maintains session continuity during upstream provider degradation.

<div align="center">
  <img width="900" alt="Care-AI LLM Fallback Chain" src="https://github.com/user-attachments/assets/c36eea7c-8298-468a-a3a4-b57d81b24b5b" />
</div>

### Routing Strategy
1. **Configurable Primary Selection**: Provider priority is configured via `LLM_PRIMARY_PROVIDER` (defaults to `gemini` for low turn latency, or `openrouter`).
2. **Sticky Session Reuse**: When a provider succeeds, subsequent calls within that conversation session reuse that provider to minimize connection overhead.
3. **Instant Failover on Depletion**: If a provider returns a rate limit (HTTP `429`), upstream capacity error (`503`), or authentication fault (`401`/`403`), the router immediately switches to the next healthy provider in the chain without blocking sleeps.
4. **Bounded Local Ollama Probing**: Local Ollama connectivity is probed using a bounded 0.8-second HTTP health check. In hosted environments lacking a local daemon, Ollama is marked offline instantly, preventing connection hangs.
5. **Secure Authentication**: Gemini API keys are passed strictly via the `x-goog-api-key` HTTP header rather than query string parameters, avoiding leakage in proxy access logs and system monitors.

### Provider Chain Configuration

| Priority | Provider | Verified Model Candidate | Primary Use Case | Failover Trigger |
|:---:|---|---|---|---|
| **1** | **Google Gemini** | `gemini-3.5-flash-lite` | Primary generation; multi-pass turn generation | HTTP 429, daily quota exhaustion, API downtime |
| **2** | **OpenRouter** | `meta-llama/llama-3.3-70b-instruct` | High-fidelity secondary fallback; deep clinical reasoning | Rate limiting, token cap exhaustion, network timeout |
| **3** | **Groq Cloud** | `llama-3.3-70b-versatile` | Ultra-fast tertiary cloud fallback | Quota depletion, provider error |
| **4** | **Ollama** | `qwen2.5:7b` (or configured) | Offline / on-premise local fallback (`localhost:11434`) | Offline daemon, port unreachable (0.8s timeout) |

> [!NOTE]
> Model identifiers are dynamically configurable via environment variables (`GEMINI_MODEL`, `OPENROUTER_MODEL`, `GROQ_MODEL`, `OLLAMA_MODEL`) and auto-migrate deprecated model tags. Cloud failover is conducted on a best-effort basis; no distributed system can guarantee absolute zero downtime under total upstream network failure.

---

## Knowledge Base & Ingestion Pipeline

Care-AI's RAG system aggregates **37,233 preprocessed clinical chunks** across five medical corpora, indexed using a versioned sparse vector representation (`cache/vector_index_v2.pkl`).

| Corpus | Primary Source | Content & Structure | Indexed Units |
|---|---|---|:---:|
| **Disease-Symptom Dataset** | Mendeley Data / Kaggle | Structured records linking 42 disease profiles to 132 unique symptoms, clinical severity weights (1–7), and preventive actions. | **42** chunks + 132 weights |
| **MedQuAD** | NIH / National Library of Medicine | Curated medical question-answering pairs across 10 clinical domains (Cancer, Diabetes, Kidney Disorders, Genetics, Heart & Lung, etc.). | **32,810** Q&A chunks |
| **MedlinePlus Health Topics** | U.S. National Institutes of Health (NIH) | Validated XML summaries describing diseases, syndromes, diagnostic tests, and wellness guidelines (`mplus_topics_2026-02-26.xml`). | **2,029** topic chunks |
| **ICD-10-CM 2026** | Centers for Medicare & Medicaid Services (CMS) | Standardized diagnostic classifications aggregated into prefix-grouped clinical code families for semantic retrieval. | **1,704** prefix groups |
| **NHS Conditions** | UK National Health Service (NHS.uk) | Real-world clinical guidance detailing conditions, acute warning signs, symptoms, and self-care protocols. | **648** condition chunks |
| **Total Corpus** | *Aggregated Multi-Source* | *Comprehensive multimodal medical knowledge index* | **37,233 Chunks** |

> [!NOTE]
> Third-party datasets and medical content remain the intellectual property of their respective creators and agencies. They are incorporated for academic evaluation and benchmarking purposes. Care-AI makes no representation that external datasets reflect the latest medical discoveries or current clinical guidelines.

---

## Engineering Improvements

The Care-AI codebase has undergone extensive structural and operational hardening to resolve memory bottlenecks, improve response stability, and protect execution state:

### 1. SciPy CSR Sparse Vector Store Optimization
- **The Problem**: Early iterations stored the TF-IDF representation as an uncompressed dense NumPy matrix, generating a **5.08 GB** index file (`cache/vector_index.pkl`) that exceeded container memory limits and caused frequent Out-Of-Memory (OOM) termination on Streamlit Cloud.
- **The Engineering Solution**: Re-engineered `core/vector_store.py` to construct and serialize compressed sparse rows using `scipy.sparse.csr_matrix` (`cache/vector_index_v2.pkl`).
- **Storage & Memory Efficiency**:
  - **Disk Footprint**: Serialized sparse CSR format occupies **26.14 MiB (27.41 MB)** on disk (`cache/vector_index_v2.pkl`), eliminating the uncompressed dense matrix format that caused container out-of-memory crashes.
  - **Sparse Matrix Representation**: Directly queries 37,233 document vectors using SciPy CSR matrix multiplication without expanding into dense memory arrays.
  - **OOM Defense**: Implemented an automated file-size safeguard that immediately rejects legacy cache files over 500 MB without deserialization.

### 2. Multi-Provider Router Hardening
- **Immediate Failover**: Removed blocking linear backoff sleeps when an alternate healthy provider is ready, preventing UI freezes during high-load periods.
- **Robust Schema Parsing**: Hardened JSON extraction against truncated provider outputs, missing Gemini `candidates` structures, and unexpected list element types (e.g., numeric confidence scores).
- **Native SSE Token Streaming**: Implemented native chunk-by-chunk generator streaming for both Gemini and OpenRouter, delivering immediate perceptual response times to end users.

### 3. Session Isolation & Concurrency Safety
- **State Sandboxing**: Decoupled expensive immutable resources (loaded once globally via `@st.cache_resource`) from mutable session entities.
- **Per-User State Management**: Conversation history, SOCRATES state counters, detected language flags, and triage ratings are stored exclusively within `st.session_state.bot`. Multiple concurrent users cannot see or mutate each other's dialogue context.

### 4. Defensive Sanitization & Environment Hardening
- **HTML Injection Defense**: User inquiries, specialist cards, and model explanations rendered within custom UI containers are escaped via `sanitize_html()` (`html.escape`), preventing malicious script execution in shared browser sessions.
- **Hosted Secrets Support**: Seamlessly reconciles local `.env` variables with Streamlit Community Cloud's native `st.secrets` store.
- **Light Professional Medical Theme**: Styled using IBM Plex Mono and Source Serif 4 typography, clear semantic color indicators (Green for Self-Care, Yellow for Clinic, Red for Urgent/Emergency), and responsive card grids.

---

## Evaluation

Test inputs reconstructed from case descriptions; author-written; expected labels author-assigned; not clinician-reviewed; 30 cases. Scripted turns were executed with the conversational intake engine. If no triage assessment was produced after scripted turns ended, the fixed harness reply `"No, that is all I can tell you."` was sent up to 3 times to allow the model to reach a final assessment. Cases finishing without a triage assessment after drain turns count as incorrect. Runs were executed in prioritized order with emergency/red-flag cases first (`TC016`–`TC030`), followed by remaining cases (`TC001`–`TC015`), with automatic provider fallback enabled across Gemini and OpenRouter. These results do not claim to match any earlier test run.

All 30 of 30 test cases completed evaluation without provider exhaustion.

### Evaluation Summary

| Metric | Measured Value | Details / Notes |
|---|---|---|
| **Date** | 2026-10-31 | Current build evaluation run |
| **Commit** | `908473f` | Branch `main` |
| **Providers Used** | Gemini (15), OpenRouter (11), Deterministic Red-Flag (4) | Multi-provider fallback chain |
| **Completed Cases (n)** | 30 / 30 | 30 completed cases; 0 provider error; 0 not run |
| **Drain Turns Used** | 10 turns across 6 cases | Fixed response sent when turns exhausted |
| **Triage Accuracy** | 24 / 30 (80.0%) | Wilson 95% CI: [62.7%, 90.5%] |
| **Emergency Recall** | 11 / 14 (78.6%) | Positive emergency support: 14 |
| **Under-Triage Count** | 3 cases | Case IDs: `TC017`, `TC022`, `TC030` |
| **Over-Triage Count** | 3 cases | Case IDs: `TC002`, `TC005`, `TC010` |
| **Language Accuracy** | 30 / 30 (100.0%) | English: 27/27, Hindi: 2/2, Telugu: 1/1 |

### Per-Class Classification Metrics

Only reported for classes with support $\ge$ 5:

| Triage Class | Support | Precision | Recall | F1-Score |
|---|:---:|:---:|:---:|:---:|
| `SELF_CARE` | 13 | 100.0% | 72.7% | 0.842 |
| `SEE_DOCTOR` | 11 | 45.5% | 100.0% | 0.625 |
| `EMERGENCY` | 14 | 100.0% | 78.6% | 0.880 |

*(Class `URGENT_CARE` excluded: support < 5 in the evaluation set).*

### Limitations
- **Author-Authored Benchmark**: Test inputs and expected labels were reconstructed from case descriptions by the author and are not clinician-reviewed.
- **Not Clinically Validated**: Care-AI has not undergone clinical validation, medical trials, or regulatory clearance.

---

## Workflow Latency Characteristics

Interaction duration varies based on whether a turn requires conversational intake or full clinical assessment:
- **Follow-up Intake Turn**: Executes a single conversational intake evaluation (ConvState transition) without triggering vector retrieval or multi-pass reasoning.
- **Full Clinical Assessment**: Executes multi-pass differential analysis, RAG context retrieval, red-flag scanning, claim-level fact-checking, and structured XAI formatting.

Response times depend on network latency, upstream cloud provider queue times, and configured model endpoints; the project does not assert fixed latency benchmarks or cloud SLAs.

---

## Quick Start

### Prerequisites
- **Python 3.10 to 3.12** (verified under Windows and Linux).
- At least one active API key from:
  - [Google AI Studio](https://aistudio.google.com/) (Gemini API)
  - [OpenRouter](https://openrouter.ai/)
  - [Groq Console](https://console.groq.com/)
  - *Or* a local [Ollama](https://ollama.com/) instance.

### 1. Clone Repository

```bash
git clone https://github.com/vinayjoseph003/care-ai.git
cd care-ai
```

### 2. Configure Virtual Environment

#### Windows (PowerShell)
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

#### macOS / Linux
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Copy the example environment template to create your local `.env`:

#### Windows (PowerShell)
```powershell
Copy-Item .env.example .env
```

#### macOS / Linux
```bash
cp .env.example .env
```

Edit `.env` to supply your API credentials and preferred primary provider:

```env
# Primary provider selection: gemini, openrouter, or groq
LLM_PRIMARY_PROVIDER=gemini

# Google Gemini (Recommended for low latency)
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-3.5-flash-lite

# OpenRouter (Secondary fallback)
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_MODEL=meta-llama/llama-3.3-70b-instruct

# Groq Cloud (Optional fast tertiary)
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile

# Local Ollama (Optional offline fallback)
OLLAMA_MODEL=qwen2.5:7b
OLLAMA_HOST=http://localhost:11434
```

> [!CAUTION]
> Never commit `.env` or paste live API credentials into public repositories. Use `.env.example` as a template for local development; `.env` and `.streamlit/secrets.toml` are included in `.gitignore` by default. When deploying to Streamlit Community Cloud, supply credentials via the platform's native secrets management (`st.secrets`) rather than committing real keys.

### 5. Launch the Streamlit Web UI

```bash
streamlit run ui/app.py
```
Open your browser at `http://localhost:8501`.

### 6. (Optional) Run via Terminal CLI

```bash
python main.py
```

---

## CLI Reference

When running the interactive terminal interface (`python main.py`), you can enter clinical symptoms or invoke built-in administrative commands:

| Command | Action | Description |
|---|---|---|
| `reset` | Session Reset | Clears conversation state, SOCRATES questions, and clinical memory for a fresh intake. |
| `severity` | View Severity Score | Displays the quantified mathematical severity score (1–7 scale) calculated from identified symptoms. |
| `log` | Inspect XAI Trace | Outputs the most recent Explainable AI log entry, differential candidates, and citations as formatted JSON. |
| `export` | Export Audit Trail | Writes the entire conversation transcript, clinical scores, and fact-checking audits to a JSON file. |
| `quit` | Exit | Safely terminates the interactive CLI session. |

---

## Verification & Automated Testing

The Care-AI repository contains a test suite of **28 automated test cases** organized across three phase-specific verification modules:

```bash
python -m unittest discover tests -v
```

### Verification Test Structure

```
tests/
├── test_phase1_verification.py   # 8 tests: Dependency completeness, sparse CSR index serialization,
│                                 #           cache reload speeds, query retrieval, session isolation
├── test_phase2_verification.py   # 10 tests: Router priority, dynamic Ollama probing, secure header auth,
│                                 #           live generation, automatic fallback, parsing resilience
└── test_phase3_verification.py   # 10 tests: Clean cold-start, primary provider priority, live Gemini turns,
                                  #           Telugu/multilingual handling, HTML injection defense, error masking
```

> [!NOTE]
> Tests marked as live integration tests require valid credentials in `.env` and consume a minimal number of API tokens. Unit tests verify sparse index serialization, error sanitization, and state isolation independently using mock fixtures.

---

## Project Structure

```
care-ai/
├── main.py                     # Interactive CLI entry point with command handler
├── requirements.txt            # Pinned dependencies (tested on Python 3.12)
├── .env.example                # Sample environment configuration template
├── .gitignore                  # Git exclusion rules (safeguards .env, logs, large files)
├── rebuild_cache.py            # Diagnostic script to regenerate the vector index cache
├── nhs_scraper.py              # Web scraper for UK NHS Conditions directory
├── nhs_parser.py               # HTML-to-JSON chunk parser for scraped NHS content
│
├── bot/                        # Core conversational orchestrator
│   └── careai.py               # CareAI class implementing the 10-pass IFVL pipeline
│
├── core/                       # AI/ML logic, retrieval, and routing
│   ├── knowledge_base.py       # Corpus loaders for CSVs, MedQuAD, MedlinePlus, ICD-10, NHS
│   ├── vector_store.py         # SciPy CSR sparse TF-IDF vector store with cache validation
│   ├── llm_router.py           # Resilient multi-provider router with instant failover
│   ├── prompts.py              # Strict system prompts, language guards, and clinical schemas
│   ├── doctor_finder.py        # Specialist recommendation engine & OpenStreetMap query
│   └── query_expander.py       # Symptom synonym expansion for enhanced TF-IDF recall
│
├── ui/                         # Streamlit browser interface
│   ├── app.py                  # Main Streamlit web application with session state isolation
│   └── doctor_finder_panel.py  # Specialist discovery panel with geolocation & OSM links
│
├── data/                       # Curated clinical datasets & knowledge files
│   ├── dataset.csv             # Disease-to-symptom mapping matrix
│   ├── symptom_Description.csv # Clinical descriptions of diseases
│   ├── symptom_precaution.csv  # Non-pharmacological precaution protocols
│   ├── Symptom-severity.csv    # Symptom severity scoring weights (1–7)
│   ├── icd10cm_order_2026.txt  # ICD-10-CM 2026 code classification (CMS)
│   ├── mplus_topics_*.xml      # MedlinePlus health topics XML export (NIH)
│   ├── nhs_raw.json            # Raw scraped NHS HTML dumps
│   ├── nhs_chunks.json         # Cleaned NHS clinical condition chunks
│   └── MedQuAD .../            # MedQuAD medical Q&A domain datasets (10 categories)
│
├── cache/                      # Serialized index storage
│   └── vector_index_v2.pkl     # Prebuilt SciPy CSR sparse index (~26.14 MiB (27.41 MB), tracked)
│
├── eval/                       # Evaluation framework & results
│   ├── README.md               # Evaluation guide & archived artifact notice
│   ├── test_cases.json         # Reconstructed test case inputs
│   ├── results_current.json    # Evaluated test case outputs
│   ├── score.py                # Standalone metrics scoring script
│   └── ...                     # Archived scripts and historical logs
│
├── tests/                      # Verification and unit test suite
│   ├── test_phase1_verification.py
│   ├── test_phase2_verification.py
│   └── test_phase3_verification.py
│
└── utils/                      # Shared auxiliary utilities
    └── display.py              # Terminal response formatting and visual cards
```

---

## Cache Maintenance

Care-AI stores its pre-indexed TF-IDF representation as a versioned sparse archive:
`cache/vector_index_v2.pkl` (~26.14 MiB (27.41 MB)).

### Index Regeneration
If you modify datasets in the `data/` directory or add new knowledge sources, regenerate the cache:

```bash
python rebuild_cache.py
```

### Operational Considerations
- **Memory Consumption During Rebuild**: Rebuilding parses uncompressed XML (`MedlinePlus`), large text files (`ICD-10-CM`), and 10 MedQuAD CSVs simultaneously. This process requires **2–4 GB of RAM** and takes 2–4 minutes depending on CPU performance.
- **Runtime Efficiency**: Once built, loading `vector_index_v2.pkl` via SciPy CSR sparse deserialization avoids the high memory footprint of dense matrix formats.
- **Legacy Cache Guard**: Do not restore the legacy `vector_index.pkl` (5.08 GB dense format). The engine automatically detects and purges incompatible or oversized cache formats to protect against system crashes.

---

## Privacy and Limitations

1. **Not Clinically Validated**: Care-AI is strictly an academic engineering research prototype for preliminary triage exploration. It has not undergone clinical trials, multi-center studies, or regulatory validation (such as FDA, EMA, or CDSCO certification as Software as a Medical Device). It cannot conduct physical examinations, laboratory tests, or imaging, and must not replace professional clinical judgment.
2. **Non-Clinician-Reviewed Phrase Lists**: Deterministic emergency red-flag triggers for Hindi and Telugu (in both native Devanagari/Telugu scripts and transliterated Latin script) are heuristic keyword rules that have not been reviewed by a qualified clinician or native-speaker clinical panel.
3. **Explicit Consent for Location**: The Doctor & Specialist Finder queries external OpenStreetMap map services (Overpass API at `overpass-api.de`) only after explicit user consent via an in-browser checkbox. Geographic coordinates are held ephemerally in session memory and are never persisted to disk.
4. **In-Memory Exports Only**: The web interface export utility produces in-memory JSON downloads (`st.download_button`) directly to the user's browser. Raw symptom inputs and session transcripts are not saved or logged to the host filesystem.
5. **Third-Party LLM Processing**: User conversation inputs and extracted clinical summaries are transmitted only to the configured LLM providers supported in the codebase: Google Gemini, OpenRouter, and Groq Cloud (or retained entirely on-premises if using a local Ollama instance). No Protected Health Information (PHI) or real patient identifiers should ever be entered.

---

## Copyright and Licensing

**Copyright © 2026 Care-AI contributors. All rights reserved.**

Care-AI was developed as an academic engineering research project. **The original project code is not currently released under an open-source license.** Permission to reproduce, modify, redistribute, sublicense, or commercially exploit the original project code is not granted except where authorized in writing by the applicable copyright holder(s), or where permitted by applicable law.

Third-party libraries, datasets, content, assets, and other external materials incorporated or referenced within this project remain subject to their respective licenses and terms:
- **MedQuAD**: Sourced from the U.S. National Library of Medicine (NIH) for academic research.
- **MedlinePlus**: Information courtesy of the National Library of Medicine (NLM), NIH.
- **ICD-10-CM 2026**: Public clinical classification published by the Centers for Medicare & Medicaid Services (CMS).
- **NHS Conditions**: Derived from the UK National Health Service (NHS.uk) content guidance.

This copyright notice does not claim ownership of external medical datasets, nor does it supersede contributor rights or platform terms.

*For licensing inquiries, collaboration requests, or evaluation access, please contact the repository maintainers through GitHub.*

---

## Authors & Acknowledgments

### Project Authors
- **Vinay Joseph** — [@vinayjoseph003](https://github.com/vinayjoseph003)
- **Rohith** — [@rohithbojja07](https://github.com/rohithbojja07)
- **Rani Chinthabathini** — [@ranichinthabathini](https://github.com/ranichinthabathini)

*Final-Year Engineering Research Project*

### Acknowledgments
- **Medical Corpora**: Sincere appreciation to the [National Institutes of Health (NIH)](https://www.nih.gov/), [MedlinePlus](https://medlineplus.gov/), [CMS](https://www.cms.gov/), and the [NHS](https://www.nhs.uk/) for providing publicly accessible medical data.
- **Open-Source Infrastructure**: Built using [Streamlit](https://streamlit.io/), [SciPy](https://scipy.org/), [NumPy](https://numpy.org/), and [Pandas](https://pandas.pydata.org/).
