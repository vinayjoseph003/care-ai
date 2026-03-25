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
#  ✅ Treatment guidance (home care, escalation, lifestyle)
#  ✅ Precautions from symptom_precaution.csv
#  ✅ Red flag symptom alerts
#  ✅ Symptom follow-up checker
#  ✅ Mental health check-in for severe/chronic conditions
#
#  [Phase 1 + 2 Upgrades]
#  ✅ [P1-A] Strict language enforcement across all prompts
#  ✅ [P1-B] 120-word answer cap — _stream_answer() 250 tok / _stream_xai() 450 tok
#  ✅ [P1-C] XAI rewritten: cause → reasoning → recommendation
#  ✅ [P1-D] Differential cards include description + key_evidence
#  ✅ [P2-A] Confidence-threshold follow-up (replaces fixed 3-cap)
#  ✅ [P2-B] Symptom-relevance gate on SOCRATES questions
#  ✅ [P2-C] Zero-information graceful handling
#
#  [Phase 3 — Hallucination Control]
#  ✅ [P3-A] Claim-level fact-checking (per-sentence verification)
#  ✅ [P3-B] Deterministic hallucination_risk from unsupported ratio
#  ✅ [P3-C] Auto-corrected answer strips unsupported claims
# ================================================================

import json, re, sys
from enum import Enum
from groq import Groq

from core import (
    VectorStore, SymptomScorer,
    LANG_DETECT_PROMPT, FOLLOWUP_PROMPT, REASONING_PROMPT,
    ANSWER_PROMPT, FACTCHECK_PROMPT, XAI_EXPLAINER_PROMPT,
    TREATMENT_PROMPT, REDFLAGS_PROMPT,
)


class ConvState(Enum):
    GREETING  = "greeting"
    GATHERING = "gathering"
    READY     = "ready"
    ANSWERED  = "answered"
    FOLLOWUP  = "followup_check"


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

MENTAL_HEALTH_CONDITIONS = {
    "depression", "anxiety", "stress", "panic", "ptsd", "bipolar",
    "schizophrenia", "ocd", "eating disorder", "insomnia", "chronic pain",
    "cancer", "diabetes", "heart disease", "chronic", "terminal",
}

# [P2-C] Words/phrases that signal user cannot provide more info
VAGUE_RESPONSES = {
    "i don't know", "idk", "not sure", "maybe", "i dont know",
    "no idea", "don't know", "dunno", "nope", "nothing", "none",
    "i am not sure", "not really", "can't say", "hard to say",
    "telidu", "telidhu", "teliyadu", "pata nahi", "nahi pata",
    "गुमान नहीं", "पता नहीं", "தெரியவில்லை",
}

CONVERSATIONAL_WRAPPER = """\
You are Care-AI, a warm and knowledgeable medical assistant.

Your tone:
  • Calm and reassuring — never alarmist, never robotic
  • Conversational — speak like a caring doctor, not a textbook
  • Concise — one idea per sentence, short paragraphs
  • Honest — always flag uncertainty

STRICT LANGUAGE RULE:
  • Detect the user's language from the conversation.
  • You MUST respond ENTIRELY in that language — every word.
  • Never switch to English or any other language.

When asking SOCRATES follow-up questions:
  • Ask ONE question at a time, naturally woven into your reply
  • Never list multiple questions at once

When giving a clinical assessment:
  • Open with a brief reassuring or urgent line matching the triage level
  • Keep the main answer to 80-120 words maximum
  • Use inline citations [1][2][3] after every factual claim

{base_prompt}
"""


class CareAI:
    def __init__(self, knowledge_chunks: list, severity_map: dict,
                 api_key: str, precaution_map: dict = None):
        self.client         = Groq(api_key=api_key)
        self.model          = "llama-3.3-70b-versatile"
        self.vs             = VectorStore(knowledge_chunks)
        self.scorer         = SymptomScorer(severity_map)
        self.precaution_map = precaution_map or {}
        self._reset_state()
        print("✅ Care-AI ready!\n")

    # ── STATE ──────────────────────────────────────────────────

    def _reset_state(self):
        self.history            = []
        self.state              = ConvState.GREETING
        self.collected_info     = {}
        self.followup_count     = 0
        # [P2-A] No hard cap — confidence threshold drives stopping
        self.user_language      = {"language": "English", "code": "en"}
        self.xai_logs           = []
        self.severity_result    = {}
        self.last_triage        = None
        self.last_specialist    = None
        self.last_differential  = []
        self.last_top_condition = ""
        self.followup_asked     = False
        self.memory_window      = 6
        self.session_summary    = ""
        # [P2-C] Track consecutive zero-info responses
        self.no_new_info_count  = 0

    # ── MEMORY ────────────────────────────────────────────────

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

    # ── STREAMING ─────────────────────────────────────────────

    def _stream(self, messages: list, label: str = "",
                max_tokens: int = 600, temperature: float = 0.3) -> str:
        """Generic streaming helper — use _stream_answer() or _stream_xai() for
        the two main pipeline passes; this stays as a fallback for anything else."""
        if label:
            print(f"\n  {label}", flush=True)
        full_text = []
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
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

    def _stream_answer(self, messages: list) -> str:
        """
        [P1-B] Pass 6 — Answer generation stream.
        250 tokens ≈ 120 words hard ceiling.
        Enforces the 80-120 word plain-language answer cap.
        """
        return self._stream(messages, label="", max_tokens=250, temperature=0.3)

    def _stream_xai(self, messages: list, label: str = "") -> str:
        """
        [P1-C] Pass 8 — XAI explanation stream.
        450 tokens — enough for 3 'Because X → Y' bullets + closing sentence
        without bleeding into the answer budget.
        """
        return self._stream(messages, label=label, max_tokens=450, temperature=0.3)

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
    # [P2-A] Confidence-threshold based stopping
    # [P2-B] Relevance gate — only asks relevant SOCRATES dims
    # [P2-C] Zero-info detection — stops after 2 vague responses

    def _is_vague_response(self, msg: str) -> bool:
        """Return True if the user's message provides no new clinical info."""
        msg_lower = msg.lower().strip()
        # Check against known vague phrases
        if any(v in msg_lower for v in VAGUE_RESPONSES):
            return True
        # Also flag very short responses with no medical content
        words = msg_lower.split()
        if len(words) <= 2 and not any(
            c.isdigit() for c in msg_lower
        ):
            return True
        return False

    # ── PATTERN DETECTOR — runs before LLM follow-up call ────
    # Detects symptom patterns from collected info + history
    # and injects reasoning hints into the follow-up prompt.
    # This is how Care-AI "connects the dots" like August AI does.

    def _detect_symptom_patterns(self) -> dict:
        """
        Analyze collected SOCRATES data + history for clinical patterns.
        Returns a dict of detected patterns and the next suggested dimension.
        """
        all_user_text = " ".join(
            m['content'].lower() for m in self.history if m['role'] == 'user'
        )
        info = self.collected_info

        patterns = []
        priority_question = None

        # ── Pattern 1: Positional / movement-triggered pain ───
        POSITIONAL_SIGNALS = [
            "bend", "bending", "lifting", "lift", "move", "moving",
            "position", "sit", "sitting", "stand", "stretch", "twist",
            "walk", "walking", "exercise", "pressure", "touch", "press",
            "worse when", "better when", "only when",
        ]
        is_positional = any(s in all_user_text for s in POSITIONAL_SIGNALS)

        # ── Pattern 2: Musculoskeletal character ──────────────
        MUSCLE_SIGNALS = [
            "cramp", "cramping", "sharp", "pull", "strain", "tight",
            "sore", "ache", "stiff", "muscle", "spasm",
        ]
        is_muscular = any(s in all_user_text for s in MUSCLE_SIGNALS)

        # ── Pattern 3: Trigger/activity not yet collected ─────
        trigger = info.get("trigger_activity")
        trigger_collected = (
            trigger and
            str(trigger).strip().lower() not in ("null", "none", "", "...")
        )

        # ── Pattern 4: Already asked about activity ───────────
        ACTIVITY_ASKED_SIGNALS = [
            "exercise", "workout", "physical activity", "gym",
            "lifting", "sport", "exertion", "activity before",
        ]
        activity_already_asked = any(
            s in m['content'].lower()
            for m in self.history if m['role'] == 'assistant'
            for s in ACTIVITY_ASKED_SIGNALS
        )

        # ── Connect the dots ──────────────────────────────────
        if is_positional and is_muscular and not trigger_collected and not activity_already_asked:
            patterns.append("POSITIONAL_MUSCULAR_PAIN")
            priority_question = (
                f"The pain is positional and muscular in character. "
                f"The most diagnostically important next question is: "
                f"did the user do any exercise, workout, heavy lifting, "
                f"or physical activity before this pain started? "
                f"Ask this question next — it may reveal the root cause immediately."
            )

        elif is_positional and not trigger_collected and not activity_already_asked:
            patterns.append("POSITIONAL_PAIN")
            priority_question = (
                f"The pain is triggered by movement or position. "
                f"Ask about recent physical activity or trigger event next."
            )

        return {
            "patterns":         patterns,
            "priority_hint":    priority_question,
            "is_positional":    is_positional,
            "is_muscular":      is_muscular,
            "trigger_missing":  not trigger_collected,
        }

    def _check_followup(self):
        history_str = "\n".join(
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
            for m in self.history
        )

        # [P2-C] Detect zero-information responses
        last_user_msg = next(
            (m['content'] for m in reversed(self.history) if m['role'] == 'user'), ""
        )
        if self._is_vague_response(last_user_msg):
            self.no_new_info_count += 1
        else:
            self.no_new_info_count = 0

        # [P2-C] Force stop after 2 consecutive vague responses
        if self.no_new_info_count >= 2:
            print("⏭️  User gave no new info — skipping further follow-up.")
            return {
                "has_enough_info":       True,
                "diagnostic_confidence": 50,
                "collected_info":        {},
                "followup_question":     None,
                "followup_reason":       "User unable to provide more information",
                "no_new_info_count":     self.no_new_info_count,
            }

        # Hard backstop: never exceed 5 follow-up questions
        if self.followup_count >= 5:
            print("⏭️  Max follow-ups reached — proceeding to answer.")
            return {
                "has_enough_info":       True,
                "diagnostic_confidence": 60,
                "collected_info":        {},
                "followup_question":     None,
                "followup_reason":       "Maximum follow-ups reached",
                "no_new_info_count":     self.no_new_info_count,
            }

        # ── Pattern detection — connect the dots before LLM ───
        patterns = self._detect_symptom_patterns()
        priority_hint = patterns.get("priority_hint", "")

        if priority_hint:
            print(f"🔍 Pattern detected: {patterns['patterns']}")

        prompt = (
            f"Conversation:\n{history_str}\n\n"
            f"SOCRATES info collected so far: {json.dumps(self.collected_info)}\n"
            f"Follow-ups asked so far: {self.followup_count}/5 (hard max)\n"
            f"User language: {self.user_language['language']}\n"
            f"Consecutive no-new-info responses: {self.no_new_info_count}\n"
            + (f"\nCLINICAL PATTERN DETECTED — PRIORITY INSTRUCTION:\n{priority_hint}\n"
               if priority_hint else "")
        )

        raw = self._llm(FOLLOWUP_PROMPT, prompt, temp=0.1)
        result = self._parse_json(raw, {
            "has_enough_info":       True,
            "diagnostic_confidence": 50,
            "collected_info":        {},
            "followup_question":     None,
            "followup_reason":       "",
            "no_new_info_count":     0,
        })

        # [P2-A] Override has_enough_info if confidence threshold met
        confidence = result.get("diagnostic_confidence", 50)
        if confidence >= 65:
            result["has_enough_info"] = True

        # Hard backstop (double-check after LLM response)
        if self.followup_count >= 5:
            result["has_enough_info"] = True
            result["followup_question"] = None

        return result

    # ── PASS 2.5: QUERY PRE-CLASSIFICATION ───────────────────
    # [P3-ROUTING] Fast single-call classifier — runs before RAG retrieval
    # so vector_store.search() receives the correct query_category for
    # source routing.  Avoids restructuring the pipeline.

    _CLASSIFY_PROMPT = (
        "Classify this medical query into ONE of these categories:\n"
        "SYMPTOM, MEDICATION, CONDITION, WELLNESS, MENTAL_HEALTH, EMERGENCY, UNKNOWN\n\n"
        "Rules:\n"
        "- SYMPTOM: user describes a physical symptom or complaint\n"
        "- MEDICATION: asks about a drug, dose, or side effect\n"
        "- CONDITION: asks what a named condition is\n"
        "- WELLNESS: general health/lifestyle question\n"
        "- MENTAL_HEALTH: emotional or psychological concern\n"
        "- EMERGENCY: life-threatening symptoms\n"
        "- UNKNOWN: anything else\n\n"
        "Return ONLY the single category word. No explanation."
    )

    def _classify_query(self) -> str:
        """Return query_category string for routing RAG retrieval."""
        last_msgs = " ".join(
            m['content'] for m in self.history[-4:] if m['role'] == 'user'
        )
        try:
            raw = self._llm(
                self._CLASSIFY_PROMPT, last_msgs,
                temp=0.0, max_tok=10,
            ).strip().upper()
            valid = {"SYMPTOM","MEDICATION","CONDITION","WELLNESS",
                     "MENTAL_HEALTH","EMERGENCY","UNKNOWN"}
            return raw if raw in valid else "SYMPTOM"
        except Exception:
            return "SYMPTOM"

    # ── PASS 3: RAG RETRIEVAL ─────────────────────────────────

    def _retrieve(self, k=6, query_category: str = "SYMPTOM"):
        parts = [m['content'] for m in self.history[-4:] if m['role'] == 'user']
        for field in ['symptom_or_topic', 'site', 'character', 'associations', 'trigger_activity', 'other_details']:
            val = self.collected_info.get(field)
            if val and str(val).strip() not in ('', '...', 'null', 'None'):
                parts.append(str(val))
        query     = " ".join(parts)
        extracted = self.scorer.extract_symptoms_from_text(query)
        if extracted:
            return self.vs.search_by_symptoms(
                extracted, k=k, query_category=query_category
            )
        return self.vs.search(query, k=k, query_category=query_category)

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
            f"{'User' if m['role'] == 'user' else 'Bot'}: {m['content']}"
            for m in recent
        )
        socrates_str = json.dumps(self.collected_info, indent=2)
        raw = self._llm(
            REASONING_PROMPT,
            f"Conversation:\n{conv_str}\n\n"
            f"SOCRATES clinical data:\n{socrates_str}\n\n"
            f"{context_str}",
            temp=0.1, max_tok=1200,
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
                f"Level: {severity_result.get('severity_level', 'N/A')}\n"
                f"High-risk symptoms: {severity_result.get('high_risk_symptoms', [])}\n"
                f"Recommendation: {severity_result.get('recommendation', '')}"
            )
        triage = reasoning.get('triage_level', 'SEE_DOCTOR')
        spec   = reasoning.get('specialist', 'General Practitioner')
        diff   = reasoning.get('differential_diagnosis', [])

        base_system = (
            f"{ANSWER_PROMPT}\n\n"
            f"User's language: {self.user_language['language']} "
            f"— YOU MUST respond ONLY in this language. Never switch languages.\n\n"
            f"{context_str}"
            f"{sev_note}\n\n"
            f"## CLINICAL REASONING:\n{json.dumps(reasoning, indent=2)}\n\n"
            f"TRIAGE LEVEL: {triage}\n"
            f"RECOMMENDED SPECIALIST: {spec}\n"
            f"TOP DIFFERENTIAL: {diff[0]['condition'] if diff else 'Unknown'}\n\n"
            f"WORD LIMIT: Your response must be 80-120 words maximum.\n"
            f"Use [1],[2],[3] citations after every factual claim."
        )
        messages = self._build_messages(self._wrap_prompt(base_system))
        return self._stream_answer(messages)

    # ── PASS 7: CLAIM-LEVEL FACT-CHECK ────────────────────────
    # [P3] Each sentence in the answer is checked individually
    # against a specific numbered context chunk.
    # Unsupported claims are stripped before showing to user.

    def _extract_claims(self, answer: str) -> list[str]:
        """Split answer into individual checkable claims (sentences)."""
        # Split on sentence boundaries, filter out very short fragments
        sentences = re.split(r'(?<=[.!?])\s+', answer.strip())
        claims = []
        for s in sentences:
            s = s.strip()
            # Skip citation markers, disclaimers, single words
            if len(s.split()) < 4:
                continue
            # Skip pure disclaimer lines
            if any(skip in s.lower() for skip in [
                "ai information only", "not a diagnosis",
                "consult a", "please note", "disclaimer"
            ]):
                continue
            claims.append(s)
        return claims

    def _factcheck(self, answer: str, context_str: str) -> dict:
        """
        Claim-level fact-checking:
        1. Extract individual claims from the answer
        2. Send all claims + context to FACTCHECK_PROMPT for per-claim verification
        3. Compute hallucination_risk from ratio of unsupported claims
        4. If risk is MEDIUM/HIGH, build corrected_answer from verified claims only
        """
        claims = self._extract_claims(answer)

        if not claims:
            return {
                "verdict":            "UNVERIFIED",
                "claim_checks":       [],
                "verified_claims":    [],
                "unsupported_claims": ["No checkable claims found"],
                "hallucination_risk": "MEDIUM",
                "corrected_answer":   None,
                "summary":            "Answer was too short to fact-check.",
            }

        # Build a numbered claim list for the prompt
        claims_str = "\n".join(f"Claim {i+1}: {c}" for i, c in enumerate(claims))

        raw = self._llm(
            FACTCHECK_PROMPT,
            f"{context_str}\n\n"
            f"## CLAIMS TO VERIFY (check each one individually):\n{claims_str}\n\n"
            f"## FULL ANSWER FOR CONTEXT:\n{answer}",
            temp=0.1,
            max_tok=1200,
        )

        fc = self._parse_json(raw, {
            "verdict":            "UNVERIFIED",
            "claim_checks":       [],
            "verified_claims":    [],
            "unsupported_claims": ["Fact-check parse failed"],
            "hallucination_risk": "HIGH",
            "corrected_answer":   None,
            "summary":            "Fact-check failed. Treat with caution.",
        })

        # ── Compute hallucination_risk from claim ratio ────────
        # Override whatever the LLM said with a deterministic calculation
        claim_checks  = fc.get("claim_checks", [])
        if claim_checks:
            total      = len(claim_checks)
            unsupported = sum(1 for c in claim_checks if not c.get("supported", True))
            ratio       = unsupported / total

            if ratio == 0:
                computed_risk = "LOW"
                computed_verdict = "VERIFIED"
            elif ratio <= 0.35:
                computed_risk = "MEDIUM"
                computed_verdict = "PARTIALLY_VERIFIED"
            else:
                computed_risk = "HIGH"
                computed_verdict = "UNVERIFIED"

            fc["hallucination_risk"] = computed_risk
            fc["verdict"]            = computed_verdict

            # ── Build corrected answer from verified claims only ──
            if computed_risk in ("MEDIUM", "HIGH") and not fc.get("corrected_answer"):
                verified_sentences = [
                    c["claim"] for c in claim_checks if c.get("supported", False)
                ]
                if verified_sentences:
                    fc["corrected_answer"] = " ".join(verified_sentences)
                else:
                    # Nothing verified — fall back to safe generic
                    fc["corrected_answer"] = (
                        "Based on your symptoms, I recommend seeing a doctor "
                        "for a proper assessment. I was unable to verify specific "
                        "claims against my medical knowledge base with enough confidence."
                    )

            # ── Populate verified/unsupported lists if empty ──────
            if not fc.get("verified_claims"):
                fc["verified_claims"] = [
                    c["claim"] for c in claim_checks if c.get("supported", False)
                ]
            if not fc.get("unsupported_claims"):
                fc["unsupported_claims"] = [
                    c["claim"] for c in claim_checks if not c.get("supported", True)
                ]

        return fc

    # ── PASS 8: XAI EXPLANATION (streamed) ────────────────────
    # [P1-C] Rewritten: cause → reasoning → recommendation format

    def _xai_explain_streamed(self, reasoning, factcheck):
        last_user = next(
            (m['content'] for m in reversed(self.history) if m['role'] == 'user'), ""
        )
        diff = reasoning.get('differential_diagnosis', [])
        top  = diff[0]['condition'] if diff else 'unknown'

        # [P1-C] Feed the chain-of-reasoning format inputs
        supporting = diff[0].get('supporting_symptoms', []) if diff else []
        key_evidence = diff[0].get('key_evidence', '') if diff else ''
        uncertainty = reasoning.get('uncertainty_zones', ['nothing specific'])
        triage = reasoning.get('triage_level', 'SEE_DOCTOR')
        triage_reason = reasoning.get('triage_reason', '')
        sources = reasoning.get('knowledge_sources', [])

        prompt = (
            f"User's message: {last_user}\n"
            f"User's language: {self.user_language['language']}\n\n"
            f"Top condition identified: {top}\n"
            f"Key symptom pointing to this: {key_evidence}\n"
            f"Supporting symptoms: {', '.join(supporting)}\n"
            f"What sources were used: {', '.join(sources)}\n"
            f"What the AI is uncertain about: {', '.join(uncertainty)}\n"
            f"Triage decision: {triage}\n"
            f"Reason for triage: {triage_reason}\n"
            f"Fact-check result: {factcheck.get('verdict', 'N/A')}\n\n"
            f"Write 3 bullets in 'Because X → Y' format, then one warm closing sentence.\n"
            f"Respond entirely in: {self.user_language['language']}"
        )
        messages = [
            {"role": "system", "content": self._wrap_prompt(XAI_EXPLAINER_PROMPT)},
            {"role": "user",   "content": prompt},
        ]
        return self._stream_xai(messages, label="💡 Why did I give this answer?")

    # ── PASS 9: TREATMENT GUIDANCE ────────────────────────────

    def _get_treatment_guidance(self, top_condition: str, triage: str) -> dict:
        """Generate ethical treatment guidance — no drug names, just care steps."""
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
            f"IMPORTANT: Do NOT mention any specific drug or medication names."
        )
        raw = self._llm(TREATMENT_PROMPT, prompt, temp=0.2, max_tok=600)
        return self._parse_json(raw, {
            "home_care":           ["Rest and stay hydrated.", "Monitor your symptoms."],
            "escalation_timeline": "If no improvement in 3 days, consult a doctor.",
            "lifestyle_tips":      ["Maintain a healthy routine."],
            "what_to_avoid":       ["Avoid strenuous activity until symptoms improve."],
        })

    # ── RED FLAG ALERTS ───────────────────────────────────────

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

    # ── MENTAL HEALTH CHECK-IN ────────────────────────────────

    def _needs_mental_health_checkin(self, condition: str, triage: str) -> bool:
        if triage in ("URGENT_CARE", "EMERGENCY"):
            return True
        cond_lower = condition.lower()
        return any(kw in cond_lower for kw in MENTAL_HEALTH_CONDITIONS)

    def _mental_health_checkin(self, condition: str) -> str:
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

    # ── SYMPTOM FOLLOW-UP CHECKER ─────────────────────────────

    def _symptom_followup_prompt(self, condition: str) -> str:
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

        # ── Emergency check (local, instant — no API call) ────
        if self._is_emergency(user_message):
            msg = self._emergency_msg()
            self._add_turn("user",      user_message)
            self._add_turn("assistant", msg)
            return {"type": "emergency", "message": msg}

        # ── Symptom follow-up check-in (once, after first answer)
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

        # ── Language detection ────────────────────────────────
        if len(self.history) == 0 or len(self.history) % 6 == 0:
            self.user_language = self._detect_language(user_message)

        self._add_turn("user", user_message)

        # ── SOCRATES follow-up engine ─────────────────────────
        # [P2-A] No hard cap — stops when confidence >= 65%
        # [P2-B] Relevance gate inside FOLLOWUP_PROMPT
        # [P2-C] Stops after 2 consecutive zero-info responses

        # Skip follow-up entirely if user already gave no info twice
        if self.no_new_info_count < 2:
            print("🤔 Checking if more info needed...", end=" ", flush=True)
            fu = self._check_followup()
            print("✅")

            for k, v in fu.get('collected_info', {}).items():
                if v and str(v).strip() not in ('', '...', 'null', 'None'):
                    self.collected_info[k] = v

            confidence = fu.get('diagnostic_confidence', 50)
            has_enough = fu.get('has_enough_info', True)

            if not has_enough and fu.get('followup_question'):
                self.followup_count += 1
                q = fu['followup_question']
                self._add_turn("assistant", q)
                self.state = ConvState.GATHERING
                return {
                    "type":                  "followup",
                    "message":               q,
                    "collected_so_far":      self.collected_info,
                    "followup_count":        self.followup_count,
                    "diagnostic_confidence": confidence,
                    "socrates_dim":          fu.get('followup_reason', ''),
                }
        else:
            print("⏭️  Proceeding to answer — user gave no new info twice.")

        self.state = ConvState.READY

        print("🔍 Classifying query...",       end=" ", flush=True)
        query_category = self._classify_query()
        print(f"✅ ({query_category})")

        print("📖 Retrieving context...",     end=" ", flush=True)
        chunks      = self._retrieve(k=6, query_category=query_category)
        context_str = self._fmt_context(chunks)
        sources     = self._fmt_sources(chunks)
        print(f"✅ ({len(chunks)} docs)")

        print("🩺 Scoring symptoms...",       end=" ", flush=True)
        sev_result = self._score_severity()
        self.severity_result = sev_result
        print("✅")

        print("🧠 Differential diagnosis...", end=" ", flush=True)
        reasoning = self._reason(context_str)
        self.last_triage        = reasoning.get('triage_level', 'SEE_DOCTOR')
        self.last_specialist    = reasoning.get('specialist', 'General Practitioner')
        self.last_differential  = reasoning.get('differential_diagnosis', [])
        diff = self.last_differential
        self.last_top_condition = diff[0]['condition'] if diff else ""
        print("✅")

        print("💊 Generating treatment guidance...", end=" ", flush=True)
        treatment = self._get_treatment_guidance(self.last_top_condition, self.last_triage)
        red_flags = self._get_red_flags(self.last_top_condition)
        print("✅")

        # Print triage header before streaming
        from utils.display import print_triage_header
        print_triage_header(
            self.last_triage,
            self.last_differential,
            self.last_specialist,
        )

        # Stream main answer
        answer = self._answer_streamed(context_str, reasoning, sev_result)

        print("🔎 Claim-level fact-checking...", end=" ", flush=True)
        fc = self._factcheck(answer, context_str)
        checks     = fc.get("claim_checks", [])
        n_verified = sum(1 for c in checks if c.get("supported", False))
        n_total    = len(checks)
        risk       = fc.get("hallucination_risk", "N/A")
        print(f"✅ ({n_verified}/{n_total} claims verified | Risk: {risk})")

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

        # Mental health check-in if warranted
        mental_health_msg = ""
        if self._needs_mental_health_checkin(self.last_top_condition, self.last_triage):
            mental_health_msg = self._mental_health_checkin(self.last_top_condition)

        self._add_turn("assistant", final)
        self.state = ConvState.ANSWERED

        # [P1-D] Include description + key_evidence in diff display
        diff_display = [
            {
                "rank":         d.get('rank'),
                "condition":    d.get('condition'),
                "probability":  d.get('probability'),
                "description":  d.get('description', ''),
                "key_evidence": d.get('key_evidence', ''),
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