# ================================================================
#  bot/careai.py — Care-AI Clinical Intelligence
#
#  ✅ SOCRATES-style follow-up questions
#  ✅ Differential diagnosis (top 3 ranked conditions)
#  ✅ 4-level triage system (Self-care / Doctor / Urgent / ER)
#  ✅ Specialist referral suggestion
#  ✅ Perplexity-style inline citations
#  ✅ Severity scoring from dataset
#  ✅ Streaming token-by-token output
#  ✅ Conversational / August-AI-style tone
#  ✅ Rolling context memory with auto-compression
#  ✅ [NEW] Treatment guidance (home care, escalation, lifestyle)
#  ✅ [NEW] Precautions from symptom_precaution.csv
#  ✅ [NEW] Red flag symptom alerts
#  ✅ [NEW] Symptom follow-up checker
#  ✅ [NEW] Mental health check-in for severe/chronic conditions
# ================================================================

import json, re, sys
from enum import Enum
from groq import Groq

from core import (
    VectorStore, SymptomScorer,
    LANG_DETECT_PROMPT, FOLLOWUP_PROMPT, REASONING_PROMPT,
    ANSWER_PROMPT, FACTCHECK_PROMPT, XAI_EXPLAINER_PROMPT,
)

# ── [NEW] Import treatment + red flag prompts ─────────────────
# Add these to your core/prompts.py (see prompts_additions.py output)
try:
    from core import TREATMENT_PROMPT, REDFLAGS_PROMPT
except ImportError:
    # Fallback inline if not yet added to core/__init__.py
    TREATMENT_PROMPT = """
You are a responsible medical guidance assistant. You do NOT prescribe drugs.
Given the diagnosed condition and triage level, provide:
1. HOME_CARE: Practical self-care steps (rest, hydration, hygiene, OTC general advice like saline rinse — never specific drug names)
2. ESCALATION: Exactly when to seek more help — "if no improvement in X days" or "if Y symptom appears, go to ER"
3. LIFESTYLE: Specific prevention tips and lifestyle adjustments for this condition
4. PRECAUTIONS: What to avoid (triggers, foods, activities, environments)

Return ONLY valid JSON:
{
  "home_care": ["step 1", "step 2", ...],
  "escalation_timeline": "If no improvement in X days, see a doctor. If [specific symptom], go to ER immediately.",
  "lifestyle_tips": ["tip 1", "tip 2", ...],
  "what_to_avoid": ["avoid 1", "avoid 2", ...]
}
"""

    REDFLAGS_PROMPT = """
You are a medical safety monitor. Given the condition and symptoms described,
list the specific RED FLAG symptoms the user must watch for that would indicate
their condition is worsening and requires immediate emergency care.

Return ONLY valid JSON:
{
  "red_flags": ["symptom 1", "symptom 2", ...],
  "emergency_threshold": "Go to ER immediately if any of the above appear."
}
"""


class ConvState(Enum):
    GREETING   = "greeting"
    GATHERING  = "gathering"
    READY      = "ready"
    ANSWERED   = "answered"
    FOLLOWUP   = "followup_check"


TRIAGE_CONFIG = {
    "SELF_CARE":   {"icon": "🟢", "label": "SELF-CARE",    "color": "green"},
    "SEE_DOCTOR":  {"icon": "🟡", "label": "SEE A DOCTOR", "color": "yellow"},
    "URGENT_CARE": {"icon": "🔴", "label": "URGENT CARE",  "color": "red"},
    "EMERGENCY":   {"icon": "🚨", "label": "EMERGENCY",    "color": "red"},
}

EMERGENCY_KEYWORDS = [
    "chest pain", "heart attack", "can't breathe", "cannot breathe",
    "difficulty breathing", "stroke", "suicidal", "want to die",
    "kill myself", "self-harm", "unconscious", "passed out",
    "seizure", "severe bleeding", "poisoning", "overdose",
    "throat swelling", "anaphylaxis", "not breathing",
]

# Conditions that warrant a mental health check-in
MENTAL_HEALTH_CONDITIONS = {
    "depression", "anxiety", "stress", "panic", "ptsd", "bipolar",
    "schizophrenia", "ocd", "eating disorder", "insomnia", "chronic pain",
    "cancer", "diabetes", "heart disease", "chronic", "terminal",
}

CONVERSATIONAL_WRAPPER = """\
You are Care-AI, a warm and knowledgeable medical assistant.

Your tone:
  • Calm and reassuring — never alarmist, never robotic
  • Conversational — speak like a caring doctor, not a textbook
  • Concise — one idea per sentence, short paragraphs
  • Honest — always flag uncertainty using the hallucination risk indicator

When asking SOCRATES follow-up questions:
  • Ask ONE question at a time, naturally woven into your reply
  • Example: "Got it — how long have you been feeling this way?"
  • Never list multiple questions at once

When giving a clinical assessment:
  • Open with a brief reassuring or urgent line matching the triage level
  • Then deliver the structured answer with inline citations [1][2][3]

{base_prompt}
"""


class CareAI:
    def __init__(self, knowledge_chunks: list, severity_map: dict,
                 api_key: str, precaution_map: dict = None):
        self.client        = Groq(api_key=api_key)
        self.model         = "llama-3.3-70b-versatile"
        self.vs            = VectorStore(knowledge_chunks)
        self.scorer        = SymptomScorer(severity_map)
        self.precaution_map = precaution_map or {}   # [NEW] disease → [precautions]
        self._reset_state()
        print("✅ Care-AI ready!\n")

    def _reset_state(self):
        self.history           = []
        self.state             = ConvState.GREETING
        self.collected_info    = {}
        self.followup_count    = 0
        self.max_followups     = 3
        self.user_language     = {"language": "English", "code": "en"}
        self.xai_logs          = []
        self.severity_result   = {}
        self.last_triage       = None
        self.last_specialist   = None
        self.last_differential = []
        self.last_top_condition = ""
        self.followup_asked    = False   # [NEW] track if follow-up check was sent
        self.memory_window     = 6
        self.session_summary   = ""

    # ── MEMORY HELPERS ────────────────────────────────────────

    def _add_turn(self, role: str, content: str):
        self.history.append({"role": role, "content": content})
        if len(self.history) > self.memory_window * 2:
            self._compress_old_turns()

    def _compress_old_turns(self):
        cutoff    = len(self.history) - self.memory_window
        old_turns = self.history[:cutoff]
        self.history = self.history[cutoff:]
        old_text = "\n".join(
            f"{t['role'].upper()}: {t['content'][:200]}" for t in old_turns
        )
        summary_prompt = (
            "Summarise this medical conversation in ≤5 sentences, "
            "preserving symptoms, diagnoses, and SOCRATES details:\n\n" + old_text
        )
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": summary_prompt}],
                max_tokens=200, temperature=0.0,
            )
            new_summary = resp.choices[0].message.content.strip()
            self.session_summary = (
                self.session_summary + "\n" + new_summary
                if self.session_summary else new_summary
            )
        except Exception:
            self.session_summary += old_text[:400]

    def _build_messages(self, system_prompt: str) -> list:
        messages = [{"role": "system", "content": system_prompt}]
        if self.session_summary:
            messages.append({
                "role": "system",
                "content": f"[Earlier conversation summary]\n{self.session_summary}",
            })
        collected = {
            k: v for k, v in self.collected_info.items()
            if v and str(v).strip() not in ('', '...', 'null', 'None')
        }
        if collected:
            note = "\n".join(f"  {k}: {v}" for k, v in collected.items())
            messages.append({
                "role": "system",
                "content": f"[SOCRATES data collected so far]\n{note}",
            })
        messages.extend(self.history)
        return messages

    # ── STREAMING OUTPUT ──────────────────────────────────────

    def _stream(self, messages: list, label: str = "") -> str:
        if label:
            print(f"\n  {label}", flush=True)
        full_text = []
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=1500,
                temperature=0.3,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta
                if not delta or not delta.content:
                    continue
                token = delta.content
                full_text.append(token)
                sys.stdout.write(token)
                sys.stdout.flush()
        except KeyboardInterrupt:
            print("\n  [streaming interrupted]")
        except Exception as e:
            print(f"\n  ⚠️  Stream error: {e}")
        print()
        return "".join(full_text)

    def _wrap_prompt(self, base_prompt: str) -> str:
        return CONVERSATIONAL_WRAPPER.format(base_prompt=base_prompt)

    # ── LLM HELPERS ───────────────────────────────────────────

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

    # ── EMERGENCY ─────────────────────────────────────────────

    def _is_emergency(self, msg):
        return any(kw in msg.lower() for kw in EMERGENCY_KEYWORDS)

    def _emergency_msg(self):
        return (
            "🚨 MEDICAL EMERGENCY DETECTED\n\n"
            "Please call your local emergency number (911 / 112 / 999) IMMEDIATELY\n"
            "or go to the nearest emergency room. Do not wait.\n\n"
            "Your safety is the top priority right now."
        )

    # ── PASS 1: LANGUAGE DETECTION ────────────────────────────

    def _detect_language(self, msg):
        raw = self._llm(LANG_DETECT_PROMPT, msg, temp=0.0, max_tok=60)
        return self._parse_json(raw, {"language": "English", "code": "en"})

    # ── PASS 2: SOCRATES FOLLOW-UP ENGINE ─────────────────────

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

    # ── PASS 3: RAG RETRIEVAL ─────────────────────────────────

    def _retrieve(self, k=6):
        parts = [m['content'] for m in self.history[-4:] if m['role'] == 'user']
        for field in ['symptom_or_topic', 'site', 'character', 'associations', 'other_details']:
            val = self.collected_info.get(field)
            if val and str(val).strip() not in ('', '...', 'null', 'None'):
                parts.append(str(val))
        query     = " ".join(parts)
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

    # ── PASS 4: SYMPTOM SEVERITY SCORING ──────────────────────

    def _score_severity(self):
        all_text = " ".join(m['content'] for m in self.history if m['role'] == 'user')
        all_text += " " + " ".join(
            str(v) for v in self.collected_info.values() if v
        )
        symptoms = self.scorer.extract_symptoms_from_text(all_text)
        return self.scorer.score(symptoms) if symptoms else {}

    # ── PASS 5: DIFFERENTIAL DIAGNOSIS + TRIAGE ───────────────

    def _reason(self, context_str):
        recent   = self.history[-8:]
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
            "query_category":         "UNKNOWN",
            "differential_diagnosis": [],
            "reasoning_trace":        "Reasoning failed.",
            "knowledge_sources":      ["Medical Dataset"],
            "confidence_score":       40,
            "confidence_explanation": "Parse error.",
            "uncertainty_zones":      ["Reasoning unavailable"],
            "context_coverage":       "NONE",
            "severity_assessment":    "UNKNOWN",
            "triage_level":           "SEE_DOCTOR",
            "triage_reason":          "Unable to determine — recommend seeing a doctor.",
            "specialist":             "General Practitioner",
            "specialist_reason":      "Start with a general assessment.",
            "emergency_detected":     False,
            "key_claims":             [],
        })

    # ── PASS 6: ANSWER GENERATION (streamed) ──────────────────

    def _answer_streamed(self, context_str, reasoning, severity_result):
        sev_note = ""
        if severity_result:
            sev_note = (
                f"\n\n## SYMPTOM SEVERITY (from dataset):\n"
                f"Level: {severity_result.get('severity_level','N/A')}\n"
                f"High-risk symptoms: {severity_result.get('high_risk_symptoms', [])}\n"
                f"Recommendation: {severity_result.get('recommendation','')}"
            )
        triage = reasoning.get('triage_level', 'SEE_DOCTOR')
        spec   = reasoning.get('specialist', 'General Practitioner')
        diff   = reasoning.get('differential_diagnosis', [])

        base_system = (
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
        messages = self._build_messages(self._wrap_prompt(base_system))
        return self._stream(messages, label="")

    # ── PASS 7: FACT-CHECK (internal, non-streamed) ───────────

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

    # ── PASS 8: XAI EXPLANATION (streamed) ────────────────────

    def _xai_explain_streamed(self, reasoning, factcheck):
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
        messages = [
            {"role": "system", "content": self._wrap_prompt(XAI_EXPLAINER_PROMPT)},
            {"role": "user",   "content": prompt},
        ]
        return self._stream(messages, label="💡 Why did I give this answer?")

    # ── [NEW] PASS 9: TREATMENT GUIDANCE ──────────────────────

    def _get_treatment_guidance(self, top_condition: str, triage: str) -> dict:
        """Generate ethical treatment guidance — no drug names, just care steps."""
        # Pull precautions from dataset if available
        dataset_precautions = []
        condition_lower = top_condition.lower()
        for disease, precautions in self.precaution_map.items():
            if disease.lower() in condition_lower or condition_lower in disease.lower():
                dataset_precautions = precautions
                break

        precaution_note = ""
        if dataset_precautions:
            precaution_note = (
                f"\n\nDataset precautions for {top_condition}: "
                + ", ".join(str(p) for p in dataset_precautions if p)
            )

        prompt = (
            f"Condition: {top_condition}\n"
            f"Triage level: {triage}\n"
            f"User language: {self.user_language['language']}"
            f"{precaution_note}\n\n"
            f"IMPORTANT: Do NOT mention any specific drug or medication names. "
            f"Only general self-care, lifestyle, and escalation guidance."
        )
        raw = self._llm(TREATMENT_PROMPT, prompt, temp=0.2, max_tok=600)
        return self._parse_json(raw, {
            "home_care":           ["Rest and stay hydrated.", "Monitor your symptoms."],
            "escalation_timeline": "If no improvement in 3 days, consult a doctor.",
            "lifestyle_tips":      ["Maintain a healthy routine."],
            "what_to_avoid":       ["Avoid strenuous activity until symptoms improve."],
        })

    # ── [NEW] RED FLAG ALERTS ─────────────────────────────────

    def _get_red_flags(self, top_condition: str) -> dict:
        """Get condition-specific red flag symptoms to watch for."""
        prompt = (
            f"Condition: {top_condition}\n"
            f"User language: {self.user_language['language']}"
        )
        raw = self._llm(REDFLAGS_PROMPT, prompt, temp=0.1, max_tok=300)
        return self._parse_json(raw, {
            "red_flags": [
                "High fever (above 39°C / 102°F)",
                "Difficulty breathing",
                "Severe or worsening pain",
            ],
            "emergency_threshold": "Go to ER immediately if any of the above appear.",
        })

    # ── [NEW] MENTAL HEALTH CHECK-IN ──────────────────────────

    def _needs_mental_health_checkin(self, condition: str, triage: str) -> bool:
        """Return True if condition warrants a mental health check-in."""
        if triage in ("URGENT_CARE", "EMERGENCY"):
            return True
        cond_lower = condition.lower()
        return any(kw in cond_lower for kw in MENTAL_HEALTH_CONDITIONS)

    def _mental_health_checkin(self, condition: str) -> str:
        """Generate a brief, warm mental health check-in message."""
        prompt = (
            f"The user may have {condition}. "
            f"Write a warm, 2-sentence mental health check-in. "
            f"Ask how they are coping emotionally. "
            f"Remind them support is available. "
            f"Do NOT diagnose. Language: {self.user_language['language']}."
        )
        return self._llm(
            "You are a compassionate care assistant. Be warm and brief.",
            prompt, temp=0.4, max_tok=120,
        )

    # ── [NEW] SYMPTOM FOLLOW-UP CHECKER ───────────────────────

    def _symptom_followup_prompt(self, condition: str) -> str:
        """
        Returns a follow-up check-in message to ask after the user's
        next message post-diagnosis. Shown once per session.
        """
        prompt = (
            f"The user was assessed for {condition}. "
            f"Write ONE warm, natural sentence asking if they are feeling better "
            f"or if symptoms have changed since we last spoke. "
            f"Language: {self.user_language['language']}."
        )
        return self._llm(
            "You are a caring medical assistant doing a gentle follow-up.",
            prompt, temp=0.4, max_tok=80,
        )

    # ── MAIN CHAT ──────────────────────────────────────────────

    def chat(self, user_message: str) -> dict:

        # Emergency check (local — instant, no API)
        if self._is_emergency(user_message):
            msg = self._emergency_msg()
            self._add_turn("user",      user_message)
            self._add_turn("assistant", msg)
            return {"type": "emergency", "message": msg}

        # [NEW] Symptom follow-up check-in (once, after first answer)
        if (
            self.state == ConvState.ANSWERED
            and not self.followup_asked
            and self.last_top_condition
        ):
            self.followup_asked = True
            followup_msg = self._symptom_followup_prompt(self.last_top_condition)
            self._add_turn("user",      user_message)
            self._add_turn("assistant", followup_msg)
            print(f"\n🔁 Care-AI (Follow-up): {followup_msg}\n")
            return {"type": "followup_check", "message": followup_msg}

        # Language detect
        if len(self.history) == 0 or len(self.history) % 6 == 0:
            self.user_language = self._detect_language(user_message)

        self._add_turn("user", user_message)

        # SOCRATES follow-up engine
        if self.followup_count < self.max_followups:
            print("🤔 Checking if more info needed...", end=" ", flush=True)
            fu = self._check_followup()
            print("✅")

            for k, v in fu.get('collected_info', {}).items():
                if v and str(v).strip() not in ('', '...', 'null', 'None'):
                    self.collected_info[k] = v

            if not fu.get('has_enough_info') and fu.get('followup_question'):
                self.followup_count += 1
                q = fu['followup_question']
                self._add_turn("assistant", q)
                self.state = ConvState.GATHERING
                return {
                    "type":             "followup",
                    "message":          q,
                    "collected_so_far": self.collected_info,
                    "followup_count":   self.followup_count,
                    "socrates_dim":     fu.get('followup_reason', ''),
                }

        self.state = ConvState.READY

        print("📖 Retrieving context...",     end=" ", flush=True)
        chunks      = self._retrieve(k=6)
        context_str = self._fmt_context(chunks)
        sources     = self._fmt_sources(chunks)
        print(f"✅ ({len(chunks)} docs)")

        print("🩺 Scoring symptoms...",       end=" ", flush=True)
        sev_result = self._score_severity()
        self.severity_result = sev_result
        print("✅")

        print("🧠 Differential diagnosis...", end=" ", flush=True)
        reasoning  = self._reason(context_str)
        self.last_triage       = reasoning.get('triage_level', 'SEE_DOCTOR')
        self.last_specialist   = reasoning.get('specialist', 'General Practitioner')
        self.last_differential = reasoning.get('differential_diagnosis', [])
        diff = self.last_differential
        self.last_top_condition = diff[0]['condition'] if diff else ""
        print("✅")

        # [NEW] Treatment guidance + red flags (parallel internal calls)
        print("💊 Generating treatment guidance...", end=" ", flush=True)
        treatment  = self._get_treatment_guidance(self.last_top_condition, self.last_triage)
        red_flags  = self._get_red_flags(self.last_top_condition)
        print("✅")

        # Print header before streaming
        from utils.display import print_triage_header
        print_triage_header(
            self.last_triage,
            self.last_differential,
            self.last_specialist,
        )

        # Stream main answer
        answer = self._answer_streamed(context_str, reasoning, sev_result)

        print("🔎 Fact-checking...", end=" ", flush=True)
        fc = self._factcheck(answer, context_str)
        print("✅")

        final = (
            fc['corrected_answer']
            if fc.get('corrected_answer')
            and fc.get('hallucination_risk') in ('MEDIUM', 'HIGH')
            else answer
        )
        if final != answer:
            print("🔧 Auto-corrected based on fact-check.")

        # Stream XAI explanation
        xai = self._xai_explain_streamed(reasoning, fc)

        # [NEW] Mental health check-in (if warranted)
        mental_health_msg = ""
        if self._needs_mental_health_checkin(self.last_top_condition, self.last_triage):
            mental_health_msg = self._mental_health_checkin(self.last_top_condition)

        self._add_turn("assistant", final)
        self.state = ConvState.ANSWERED

        diff_display = [
            {
                "rank":        d.get('rank'),
                "condition":   d.get('condition'),
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
            "treatment":         treatment,
            "red_flags":         red_flags,
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
            # [NEW]
            "treatment":          treatment,
            "red_flags":          red_flags,
            "mental_health_msg":  mental_health_msg,
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