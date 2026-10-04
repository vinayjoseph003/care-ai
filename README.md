<div align="center">

<h1>🏥 Care-AI</h1>

<p><strong>Multilingual Intelligent Medical Assistant with Fallback LLM System & Explainable AI</strong></p>

<p>
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/RAG-Powered-6DB33F?style=for-the-badge" />
  <img src="https://img.shields.io/badge/XAI-Enabled-8A2BE2?style=for-the-badge" />
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" />
</p>

<p>
  <a href="#-features">Features</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-knowledge-sources">Knowledge Sources</a> •
  <a href="#-evaluation">Evaluation</a> •
  <a href="#-project-structure">Project Structure</a>
</p>

</div>

---

## 📖 Overview

**Care-AI** is a final-year engineering project — a production-grade, conversational medical information chatbot designed to triage symptoms, answer multilingual health queries, and provide explainable, clinically-grounded responses.

It combines **Retrieval-Augmented Generation (RAG)** over a rich multi-source medical knowledge base with a **resilient multi-provider LLM router** that gracefully falls back across OpenRouter → Groq → Gemini → Ollama, ensuring zero downtime even under API rate limits.

> ⚠️ **Disclaimer:** Care-AI is an academic research prototype and is **not a substitute for professional medical advice**. Always consult a qualified healthcare provider for medical decisions.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🤖 **Multi-Provider LLM** | Sticky fallback chain: OpenRouter → Groq → Gemini → Ollama (local) |
| 🌍 **Multilingual** | Auto-detects and responds in the user's language |
| 🔍 **RAG-Powered** | Vector-store retrieval from 5 knowledge sources |
| 🧠 **Explainable AI (XAI)** | Traces every response back to its knowledge source |
| 🩺 **Symptom Triage** | Severity scoring: SELF_CARE / SEE_DOCTOR / URGENT_CARE / EMERGENCY |
| 🔢 **Symptom Scoring** | Weighted severity per symptom, aggregated to a risk score |
| 🏥 **Doctor Finder** | Suggests appropriate specialist based on symptoms |
| 💬 **Streamlit UI** | Full chat interface with streaming token output |
| 📊 **Session Export** | Export conversation + XAI logs as JSON |

---

## 🏗️ Architecture

The following diagram shows the modular architecture of CARE-AI, including UI, core bot logic, retrieval pipeline, and LLM routing.

<img width="1536" height="1024" alt="architecture" src="https://github.com/user-attachments/assets/d354f1f1-3ef8-447c-baa9-47223dc2361f" />

---

## 🔁 LLM Fallback Chain

Care-AI uses a **sticky provider strategy** — once a provider succeeds, all subsequent calls in the session reuse it. It only falls back when rate-limited after retries.

<img width="1536" height="1024" alt="llms" src="https://github.com/user-attachments/assets/c36eea7c-8298-468a-a3a4-b57d81b24b5b" />

---

## 📚 Knowledge Sources

Care-AI's RAG system is built over **5 medical knowledge sources**:

| Source | Type | Content |
|---|---|---|
| **Mendeley Disease-Symptom CSV** | Structured | 132 diseases, symptoms, severity weights, precautions |
| **MedQuAD** | Q&A | 16,000+ medical Q&A pairs across 10 clinical categories |
| **MedlinePlus (NIH)** | XML | Health topic summaries from the US National Library of Medicine |
| **ICD-10-CM 2026** | Codes | Full ICD-10 clinical classification with prefix-grouped chunks |
| **NHS Conditions (UK)** | Scraped | Real-world condition pages parsed from NHS.uk |

---

## ⚡ Quick Start

### Prerequisites

- Python 3.10+
- At least one API key (OpenRouter, Groq, or Gemini) **or** [Ollama](https://ollama.com) running locally

### 1. Clone the repository

```bash
git clone https://github.com/vinayjoseph003/care-ai.git
cd care-ai
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:

```env
# At least one of these is required
OPENROUTER_API_KEY=your_openrouter_key_here
GROQ_API_KEY=your_groq_key_here
GEMINI_API_KEY=your_gemini_key_here

# Optional: model overrides
OPENROUTER_MODEL=meta-llama/llama-3.3-70b-instruct:free
GROQ_MODEL=llama-3.3-70b-versatile
GEMINI_MODEL=gemini-2.0-flash-lite

# Optional: local Ollama
OLLAMA_MODEL=qwen2.5:7b
OLLAMA_HOST=http://localhost:11434
```

### 5. Run the Streamlit UI

```bash
streamlit run ui/app.py
```

### 5b. Or run the CLI version

```bash
python main.py
```

---

## 🖥️ CLI Commands

When running via `python main.py`, the following commands are available mid-conversation:

| Command | Action |
|---|---|
| `reset` | Clear conversation history and start fresh |
| `severity` | Display the symptom severity score for the last query |
| `log` | Print the latest XAI log entry as JSON |
| `export` | Save the full session log to a file |
| `quit` | Exit the application |

---

## 📊 Evaluation Results

Evaluated on **30 clinical test cases** across three risk categories (IEEE-grade metrics):

### Table I — Per-Category Clinical Performance

| Category | N | Triage Acc. | Lang Acc. | Hallucination-OK | Under-triage | XAI Score | Avg RT (s) |
|---|---|---|---|---|---|---|---|
| Common / Low Triage | 15 | **100.0%** | 100.0% | 80.0% | 0 | 66.7% | 48.2 |
| Neuro Red Flags | 8 | 87.5% | 100.0% | 62.5% | 1 | 75.0% | 74.3 |
| Cardiac Red Flags | 7 | 85.7% | 85.7% | 57.1% | 0 | 71.4% | 61.8 |
| **OVERALL** | **30** | **86%** | **92%** | **66.7%** | **1** | **71%** | **60.5** |

### Table II — Triage Classification Metrics

| Triage Level | Precision | Recall | F1-Score |
|---|---|---|---|
| SELF_CARE | 0.93 | 0.87 | **0.90** |
| SEE_DOCTOR | 0.85 | 0.89 | **0.87** |
| URGENT_CARE | 0.50 | 0.50 | 0.50 |
| EMERGENCY | 0.92 | 1.00 | **0.96** |
| **Weighted Avg** | — | — | **0.88** |

### Table III — System Reliability

| Metric | Value |
|---|---|
| Primary Provider Success | **96.7%** |
| Fallback Activation Rate | 3.3% |
| Fallback Success Rate | **100%** |
| Under-triage Cases | 1 / 30 |

> The **zero under-triage** constraint for EMERGENCY cases was met: all 7 EMERGENCY cases were correctly escalated (Recall = 1.00).

---

## 📁 Project Structure

```
care-ai/
├── main.py                  # CLI entry point
├── requirements.txt         # Python dependencies
├── .env.example             # Environment variable template
│
├── bot/                     # Core chatbot logic
│   └── careai.py            # CareAI class (conversation, XAI, session)
│
├── core/                    # AI/ML pipeline
│   ├── knowledge_base.py    # Multi-source data loader & RAG chunks
│   ├── vector_store.py      # TF-IDF + cosine similarity retrieval
│   ├── llm_router.py        # Multi-provider LLM fallback router
│   ├── prompts.py           # System prompts & clinical templates
│   ├── doctor_finder.py     # Specialist recommendation engine
│   └── query_expander.py    # Query expansion for better retrieval
│
├── ui/                      # Streamlit web interface
│   ├── app.py               # Main Streamlit app
│   └── doctor_finder_panel.py  # Doctor finder UI panel
│
├── utils/                   # Shared utilities
│   └── display.py           # CLI response formatting
│
├── data/                    # Knowledge base data files
│   ├── dataset.csv          # Mendeley disease-symptom data
│   ├── symptom_Description.csv
│   ├── symptom_precaution.csv
│   ├── Symptom-severity.csv
│   ├── nhs_chunks.json      # Pre-parsed NHS conditions
│   └── nhs_raw.json         # Raw NHS scrape output
│
├── eval/                    # Evaluation framework
│   ├── run_eval_v5.py       # Evaluation pipeline
│   ├── metrics_ieee.py      # IEEE-grade clinical metrics
│   ├── test_cases_evaluated.json
│   └── summary_final.csv
│
├── nhs_scraper.py           # NHS.uk conditions scraper
├── nhs_parser.py            # NHS HTML → JSON parser
└── rebuild_cache.py         # Rebuild vector store cache
```

---

## 🔑 Getting Free API Keys

| Provider | Free Tier | Sign Up |
|---|---|---|
| **OpenRouter** | Free models (Llama-3.3-70B) | [openrouter.ai](https://openrouter.ai) |
| **Groq** | Generous free tier | [console.groq.com](https://console.groq.com) |
| **Gemini** | 1,500 req/day free | [aistudio.google.com](https://aistudio.google.com) |
| **Ollama** | Fully local, unlimited | [ollama.com](https://ollama.com) |

---

## 🛠️ Rebuilding the Cache

If you add new data sources or update knowledge base files:

```bash
python rebuild_cache.py
```

To regenerate NHS data from scratch:

```bash
python nhs_scraper.py    # Scrapes NHS.uk → data/nhs_raw.json
python nhs_parser.py     # Parses raw → data/nhs_chunks.json
```

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'feat: add your feature'`)
4. Push and open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 👤 Authors

- **Vinay Joseph** – [@vinayjoseph003](https://github.com/vinayjoseph003)  
- **Rohith** – [@rohithbojja07](https://github.com/rohithbojja07)
- **Rani Chinthabathini** – [@ranichinthabathini](https://github.com/ranichinthabathini-byte)
<div align="center">
  <sub>Built as a Final Year Engineering Project | Academic Research Prototype</sub>
</div>
