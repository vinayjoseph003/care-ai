# ================================================================
#  bot/careai.py — Care-AI Clinical Intelligence Update
#
#  New in this version:
#  ✅ SOCRATES-style follow-up questions
#  ✅ Differential diagnosis (top 3 ranked conditions)
#  ✅ 4-level triage system (Self-care / Doctor / Urgent / ER)
#  ✅ Specialist referral suggestion
#  ✅ Perplexity-style inline citations
#  ✅ Severity scoring from dataset
# ================================================================

import json, re
from enum import Enum
from groq import Groq

from core import (
    VectorStore, SymptomScorer,
    LANG_DETECT_PROMPT, FOLLOWUP_PROMPT, REASONING_PROMPT,
    ANSWER_PROMPT, FACTCHECK_PROMPT, XAI_EXPLAINER_PROMPT,
)


class ConvState(Enum):
    GREETING  = "greeting"
    GATHERING = "gathering"
    READY     = "ready"
    ANSWERED  = "answered"


# Triage levels with display info
TRIAGE_CONFIG = {
    "SELF_CARE":   {"icon": "🟢", "label": "SELF-CARE",   "color": "green"},
    "SEE_DOCTOR":  {"icon": "🟡", "label": "SEE A DOCTOR","color": "yellow"},
    "URGENT_CARE": {"icon": "🔴", "label": "URGENT CARE", "color": "red"},
    "EMERGENCY":   {"icon": "🚨", "label": "EMERGENCY",   "color": "red"},
}

EMERGENCY_KEYWORDS = [
    "chest pain", "heart attack", "can't breathe", "cannot breathe",
    "difficulty breathing", "stroke", "suicidal", "want to die",
    "kill myself", "self-harm", "unconscious", "passed out",
    "seizure", "severe bleeding", "poisoning", "overdose",
    "throat swelling", "anaphylaxis", "not breathing",
]

# Specialist mapping for display
SPECIALIST_ICONS = {
    "gastroenterologist": "🫁", "cardiologist": "❤️",
    "neurologist": "🧠",        "dermatologist": "🩺",
    "psychiatrist": "🧘",       "pulmonologist": "🫁",
    "orthopedist": "🦴",        "urologist": "💧",
    "gynecologist": "🌸",       "endocrinologist": "⚗️",
    "general practitioner": "👨‍⚕️", "emergency": "🚨",
}


class CareAI:
    def __init__(self, knowledge_chunks: list, severity_map: dict, api_key: str):
        self.client  = Groq(api_key=api_key)
        self.model   = "llama-3.3-70b-versatile"
        self.vs      = VectorStore(knowledge_chunks)
        self.scorer  = SymptomScorer(severity_map)
        self._reset_state()
        print("✅ Care-AI ready!\n")

    def _reset_state(self):
        self.history         = []
        self.state           = ConvState.GREETING
        self.collected_info  = {}
        self.followup_count  = 0
        self.max_followups   = 3
        self.user_language   = {"language": "English", "code": "en"}
        self.xai_logs        = []
        self.severity_result = {}
        self.last_triage     = None
        self.last_specialist = None
        self.last_differential = []

    # ── LLM HELPERS ──────────────────────────────────────────
    def _llm(self, system, user_content, temp=0.2, max_tok=900):
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user_content},
            ],
            temperature=temp, max_tokens=max_tok,
        )
        return resp.choices[0].message.content.strip()

    def _llm_conv(self, system, temp=0.3, max_tok=1500):
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}] + self.history,
            temperature=temp, max_tokens=max_tok,
        )
        return resp.choices[0].message.content.strip()

    def _parse_json(self, raw, fallback):
        raw = re.sub(r'^```json\s*', '', raw.strip())
        raw = re.sub(r'^```\s*',     '', raw)
        raw = re.sub(r'\s*```$',     '', raw)
        try:    return json.loads(raw)
        except: return fallback

    # ── EMERGENCY ────────────────────────────────────────────
    def _is_emergency(self, msg):
        return any(kw in msg.lower() for kw in EMERGENCY_KEYWORDS)

    def _emergency_msg(self):
        return (
            "🚨 MEDICAL EMERGENCY DETECTED\n\n"
            "Please call your local emergency number (911 / 112 / 999) IMMEDIATELY\n"
            "or go to the nearest emergency room. Do not wait.\n\n"
            "Your safety is the top priority right now."
        )

    # ── PASS 1: LANGUAGE DETECTION ───────────────────────────
    def _detect_language(self, msg):
        raw = self._llm(LANG_DETECT_PROMPT, msg, temp=0.0, max_tok=60)
        return self._parse_json(raw, {"language": "English", "code": "en"})

    # ── PASS 2: SOCRATES FOLLOW-UP ENGINE ────────────────────
    def _check_followup(self):
        history_str = "\n".join(
            f"{'User' if m['role']=='user' else 'Assistant'}: {m['content']}"
            for m in self.history
        )
        prompt = (
            f"Conversation:\n{history_str}\n\n"
            f"SOCRATES info collected: {json.dumps(self.collected_info)}\n"
            f"Follow-ups asked: {self.followup_count}/{self.max_followups}\n"
            f"User language: {self.user_language['language']}"
        )
        raw = self._llm(FOLLOWUP_PROMPT, prompt, temp=0.1)
        return self._parse_json(raw, {
            "has_enough_info":   True,
            "collected_info":    {},
            "followup_question": None,
            "followup_reason":   "",
        })

    # ── PASS 3: RAG RETRIEVAL ────────────────────────────────
    def _retrieve(self, k=6):
        # Rich query from SOCRATES collected info + recent messages
        parts = [m['content'] for m in self.history[-4:] if m['role'] == 'user']
        for field in ['symptom_or_topic', 'site', 'character', 'associations', 'other_details']:
            val = self.collected_info.get(field)
            if val and str(val).strip() not in ('', '...', 'null', 'None'):
                parts.append(str(val))
        query = " ".join(parts)

        extracted = self.scorer.extract_symptoms_from_text(query)
        if extracted:
            return self.vs.search_by_symptoms(extracted, k=k)
        return self.vs.search(query, k=k)

    def _fmt_context(self, chunks):
        if not chunks:
            return "No relevant context found in the knowledge base."
        lines = ["## RETRIEVED MEDICAL CONTEXT (numbered for citation):"]
        for i, c in enumerate(chunks, 1):
            source   = c.get('source', 'Unknown')
            category = c.get('category', 'General')
            disease  = c.get('disease', c.get('title', ''))
            sev      = c.get('severity', '')
            lines.append(
                f"\n[{i}] Source: {source} | Category: {category}"
                + (f" | Condition: {disease}" if disease else "")
                + (f" | Severity: {sev}"      if sev      else "")
            )
            lines.append(f"    {c['text'][:500]}")
        return "\n".join(lines)

    def _fmt_sources(self, chunks):
        sources = []
        for i, c in enumerate(chunks, 1):
            source  = c.get('source', 'Unknown')
            disease = c.get('disease', c.get('title', ''))
            cat     = c.get('category', '')
            label   = f"[{i}] {source}"
            if disease: label += f" — {disease}"
            if cat:     label += f" ({cat})"
            sources.append(label)
        return sources

    # ── PASS 4: SYMPTOM SEVERITY SCORING ─────────────────────
    def _score_severity(self):
        all_text = " ".join(m['content'] for m in self.history if m['role'] == 'user')
        # Also include collected SOCRATES info
        all_text += " " + " ".join(
            str(v) for v in self.collected_info.values() if v
        )
        symptoms = self.scorer.extract_symptoms_from_text(all_text)
        return self.scorer.score(symptoms) if symptoms else {}

    # ── PASS 5: DIFFERENTIAL DIAGNOSIS + TRIAGE ──────────────
    def _reason(self, context_str):
        recent   = self.history[-8:]  # more history for clinical reasoning
        conv_str = "\n".join(
            f"{'User' if m['role']=='user' else 'Bot'}: {m['content']}"
            for m in recent
        )
        socrates_str = json.dumps(self.collected_info, indent=2)

        raw = self._llm(
            REASONING_PROMPT,
            f"Conversation:\n{conv_str}\n\n"
            f"SOCRATES clinical data:\n{socrates_str}\n\n"
            f"{context_str}",
            temp=0.1, max_tok=1000,
        )
        return self._parse_json(raw, {
            "query_category":       "UNKNOWN",
            "differential_diagnosis": [],
            "reasoning_trace":      "Reasoning failed.",
            "knowledge_sources":    ["Medical Dataset"],
            "confidence_score":     40,
            "confidence_explanation": "Parse error.",
            "uncertainty_zones":    ["Reasoning unavailable"],
            "context_coverage":     "NONE",
            "severity_assessment":  "UNKNOWN",
            "triage_level":         "SEE_DOCTOR",
            "triage_reason":        "Unable to determine — recommend seeing a doctor.",
            "specialist":           "General Practitioner",
            "specialist_reason":    "Start with a general assessment.",
            "emergency_detected":   False,
            "key_claims":           [],
        })

    # ── PASS 6: ANSWER GENERATION ────────────────────────────
    def _answer(self, context_str, reasoning, severity_result):
        sev_note = ""
        if severity_result:
            sev_note = (
                f"\n\n## SYMPTOM SEVERITY (from dataset):\n"
                f"Level: {severity_result.get('severity_level','N/A')}\n"
                f"High-risk symptoms: {severity_result.get('high_risk_symptoms', [])}\n"
                f"Recommendation: {severity_result.get('recommendation','')}"
            )

        triage   = reasoning.get('triage_level', 'SEE_DOCTOR')
        spec     = reasoning.get('specialist', 'General Practitioner')
        diff     = reasoning.get('differential_diagnosis', [])

        system = (
            f"{ANSWER_PROMPT}\n\n"
            f"User's language: {self.user_language['language']} "
            f"— respond ONLY in this language.\n\n"
            f"{context_str}"
            f"{sev_note}\n\n"
            f"## CLINICAL REASONING:\n{json.dumps(reasoning, indent=2)}\n\n"
            f"TRIAGE LEVEL: {triage}\n"
            f"RECOMMENDED SPECIALIST: {spec}\n"
            f"TOP DIFFERENTIAL: {diff[0]['condition'] if diff else 'Unknown'}\n\n"
            f"Use [1],[2],[3] citations after every factual claim."
        )
        return self._llm_conv(system, temp=0.3, max_tok=1500)

    # ── PASS 7: FACT-CHECK ───────────────────────────────────
    def _factcheck(self, answer, context_str):
        raw = self._llm(
            FACTCHECK_PROMPT,
            f"{context_str}\n\n## ANSWER TO VERIFY:\n{answer}",
            temp=0.1,
        )
        return self._parse_json(raw, {
            "verdict":            "UNVERIFIED",
            "verified_claims":    [],
            "unsupported_claims": ["Fact-check failed"],
            "hallucination_risk": "HIGH",
            "corrected_answer":   None,
            "summary":            "Fact-check failed. Treat with caution.",
        })

    # ── PASS 8: XAI PLAIN EXPLANATION ────────────────────────
    def _xai_explain(self, reasoning, factcheck):
        last_user = next(
            (m['content'] for m in reversed(self.history) if m['role'] == 'user'), ""
        )
        diff = reasoning.get('differential_diagnosis', [])
        top  = diff[0]['condition'] if diff else 'unknown'
        prompt = (
            f"User's question: {last_user}\n"
            f"User's language: {self.user_language['language']}\n\n"
            f"Top likely condition: {top}\n"
            f"Confidence: {reasoning.get('confidence_score')}%\n"
            f"Why confident: {reasoning.get('confidence_explanation')}\n"
            f"Sources: {', '.join(reasoning.get('knowledge_sources', []))}\n"
            f"Uncertain about: {', '.join(reasoning.get('uncertainty_zones', ['nothing']))}\n"
            f"Triage: {reasoning.get('triage_level')}\n"
            f"Fact-check: {factcheck.get('verdict')} | "
            f"Risk: {factcheck.get('hallucination_risk')}"
        )
        return self._llm(XAI_EXPLAINER_PROMPT, prompt, temp=0.4, max_tok=350)

    # ── MAIN CHAT ─────────────────────────────────────────────
    def chat(self, user_message: str) -> dict:
        # Emergency check
        if self._is_emergency(user_message):
            msg = self._emergency_msg()
            self.history.append({"role": "user",      "content": user_message})
            self.history.append({"role": "assistant",  "content": msg})
            return {"type": "emergency", "message": msg}

        # Language detect
        if len(self.history) == 0 or len(self.history) % 6 == 0:
            self.user_language = self._detect_language(user_message)

        self.history.append({"role": "user", "content": user_message})

        # SOCRATES follow-up engine
        if self.followup_count < self.max_followups:
            print("🤔 Checking if more info needed...", end=" ", flush=True)
            fu = self._check_followup()
            print("✅")

            # Merge SOCRATES collected info
            for k, v in fu.get('collected_info', {}).items():
                if v and str(v).strip() not in ('', '...', 'null', 'None'):
                    self.collected_info[k] = v

            if not fu.get('has_enough_info') and fu.get('followup_question'):
                self.followup_count += 1
                q = fu['followup_question']
                self.history.append({"role": "assistant", "content": q})
                self.state = ConvState.GATHERING
                return {
                    "type":             "followup",
                    "message":          q,
                    "collected_so_far": self.collected_info,
                    "followup_count":   self.followup_count,
                    "socrates_dim":     fu.get('followup_reason', ''),
                }

        self.state = ConvState.READY

        print("📖 Retrieving context...",         end=" ", flush=True)
        chunks      = self._retrieve(k=6)
        context_str = self._fmt_context(chunks)
        sources     = self._fmt_sources(chunks)
        print(f"✅ ({len(chunks)} docs)")

        print("🩺 Scoring symptoms...",            end=" ", flush=True)
        sev_result = self._score_severity()
        self.severity_result = sev_result
        print("✅")

        print("🧠 Differential diagnosis...",      end=" ", flush=True)
        reasoning  = self._reason(context_str)
        self.last_triage      = reasoning.get('triage_level', 'SEE_DOCTOR')
        self.last_specialist  = reasoning.get('specialist', 'General Practitioner')
        self.last_differential = reasoning.get('differential_diagnosis', [])
        print("✅")

        print("💬 Generating answer...",           end=" ", flush=True)
        answer     = self._answer(context_str, reasoning, sev_result)
        print("✅")

        print("🔎 Fact-checking...",               end=" ", flush=True)
        fc         = self._factcheck(answer, context_str)
        print("✅")

        final = (
            fc['corrected_answer']
            if fc.get('corrected_answer')
            and fc.get('hallucination_risk') in ('MEDIUM', 'HIGH')
            else answer
        )
        if final != answer:
            print("🔧 Auto-corrected based on fact-check.")

        print("💡 Generating XAI explanation...", end=" ", flush=True)
        xai = self._xai_explain(reasoning, fc)
        print("✅\n")

        self.history.append({"role": "assistant", "content": final})
        self.state = ConvState.ANSWERED

        # Build differential for display
        diff_display = [
            {
                "rank":      d.get('rank'),
                "condition": d.get('condition'),
                "probability": d.get('probability'),
            }
            for d in self.last_differential
        ]

        self.xai_logs.append({
            "turn":              len(self.xai_logs) + 1,
            "user_message":      user_message,
            "language":          self.user_language,
            "socrates_data":     self.collected_info,
            "retrieved_sources": sources,
            "severity":          sev_result,
            "differential":      diff_display,
            "triage":            self.last_triage,
            "specialist":        self.last_specialist,
            "reasoning":         reasoning,
            "factcheck":         fc,
            "final_answer":      final,
            "xai_explanation":   xai,
        })

        return {
            "type":               "answer",
            "message":            final,
            "xai_explanation":    xai,
            "sources":            sources,
            "confidence":         reasoning.get('confidence_score', 0),
            "hallucination_risk": fc.get('hallucination_risk', 'N/A'),
            "fact_check":         fc.get('verdict', 'N/A'),
            "language":           self.user_language['language'],
            "severity":           sev_result.get('severity_level', 'N/A'),
            "triage":             self.last_triage,
            "specialist":         self.last_specialist,
            "differential":       diff_display,
        }

    def reset(self):
        self._reset_state()
        print("🔄 Session reset!\n")

    def export_log(self, path="logs/session_log.json"):
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.xai_logs, f, indent=2)
        print(f"✅ Log exported → {path}")