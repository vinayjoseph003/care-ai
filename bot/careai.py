# ================================================================
#  bot/careai.py — Care-AI Clinical Intelligence
#
#  [Core features]
#  Multilingual, XAI, Hallucination-Controlled Medical Triage Chatbot
#  IFVL (Inquiry-First, Verify-Later) 10-pass pipeline
#
#  Pass layout:
#    1  Language detection
#    2  SOCRATES follow-up engine (confidence-threshold stopping)
#    2.5 Query pre-classification (local, no API call)
#    3  RAG retrieval (TF-IDF + symptom extractor)
#    4  Symptom severity scoring (dataset)
#    5  Differential diagnosis (self-consistency, 2 passes)
#    6  Answer generation (streamed, 80-120 word cap)
#    7  Claim-level fact-check (hallucination control)
#    8  XAI explanation (streamed, Because X -> Y format)
#    9  Treatment guidance
#    10 Red flag alerts + mental health check-in
#
#  Safety constraint: zero under-triage — triage disagreements always
#  resolve to the MORE urgent level.
#
#  [Phase 1 + 2 Upgrades]
#  [P1-A] Strict language enforcement across all prompts
#  [P1-B] 120-word answer cap — _stream_answer() 250 tok / _stream_xai() 450 tok
#  [P1-C] XAI rewritten: cause -> reasoning -> recommendation
#  [P1-D] Differential cards include description + key_evidence
#  [P2-A] Confidence-threshold follow-up (replaces fixed 3-cap)
#  [P2-B] Symptom-relevance gate on SOCRATES questions
#  [P2-C] Zero-information graceful handling
#
#  [Phase 3 — Hallucination Control]
#  [P3-A] Claim-level fact-checking (per-sentence verification)
#  [P3-B] Deterministic hallucination_risk from unsupported ratio
#  [P3-C] Auto-corrected answer strips unsupported claims
#  [P3-D] Self-consistency — reasoning runs twice, disagreements flagged
#
#  [v2.1 fixes]
#  - All .join() calls now coerce elements to str() — prevents TypeError
#    when LLM returns int/float in list fields (e.g. confidence: 85)
#  - _llm() error message matches new router error string (no emoji)
#  - reset() print is Windows cp1252 safe
# ================================================================

import json, re, sys, time
from enum import Enum

from core.llm_router import LLMRouter
from core.doctor_finder import find_nearby_doctors
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
    RESOLVED  = "resolved"   # user confirmed symptoms are gone / better


TRIAGE_CONFIG = {
    "SELF_CARE":   {"icon": "green",  "label": "SELF-CARE",    "color": "green"},
    "SEE_DOCTOR":  {"icon": "yellow", "label": "SEE A DOCTOR", "color": "yellow"},
    "URGENT_CARE": {"icon": "red",    "label": "URGENT CARE",  "color": "red"},
    "EMERGENCY":   {"icon": "red",    "label": "EMERGENCY",    "color": "red"},
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
    "gulam nahi", "pata nahi",
}

# Words/phrases that mean "I'm better / it's resolved" — skip pipeline, respond warmly
RESOLVED_SIGNALS = {
    # English — formal
    "cured", "got cured", "it got cured", "better now", "all good",
    "feeling better", "i'm fine now", "im fine now", "fine now",
    "recovered", "it's gone", "its gone", "went away", "gone now",
    "healed", "no more pain", "pain is gone", "no pain now",
    "no more", "all well", "much better", "totally fine",
    # English — casual / Indian-English
    "yea it", "yep it", "yeah it", "yes it got", "yes it's",
    "it cured", "got better", "became fine", "now fine", "now ok",
    "now okay", "am fine now", "i am fine", "i'm ok now", "i'm okay",
    "no problem now", "no issue now", "solved", "problem solved",
    # Hindi
    "theek ho gaya", "theek hai", "theek hoon", "sahi ho gaya",
    "theek hogaya", "bilkul theek", "ab theek", "haan theek",
    "ha theek", "theek ho", "dard nahi", "dard chala gaya",
    "thik ho gaya",
    # Telugu
    "naluga undi", "nalugatunnanu", "baagunna", "manchiga undi",
    "manchigaa undi", "naku nalugatundi", "pain pogindi",
    "pain ledu", "pain thaggindi",
    # Tamil
    "sari aaittu", "nalla irukken", "paravaillai",
    # Kannada
    "chennaagide", "sariyaagide",
}

CONVERSATIONAL_WRAPPER = """\
You are Care-AI, a warm and knowledgeable medical assistant.

Your tone:
  * Calm and reassuring — never alarmist, never robotic
  * Conversational — speak like a caring doctor, not a textbook
  * Concise — one idea per sentence, short paragraphs
  * Honest — always flag uncertainty

STRICT LANGUAGE RULE:
  * Detect the user's language from the conversation.
  * You MUST respond ENTIRELY in that language — every word.
  * Never switch to English or any other language.

When asking SOCRATES follow-up questions:
  * Ask ONE question at a time, naturally woven into your reply
  * Never list multiple questions at once

When giving a clinical assessment:
  * Open with a brief reassuring or urgent line matching the triage level
  * Keep the main answer to 80-120 words maximum
  * Use inline citations [1][2][3] after every factual claim

{base_prompt}
"""


class CareAI:
    def __init__(self, knowledge_chunks: list, severity_map: dict,
                 router_config: dict, precaution_map: dict = None,
                 # Legacy arg — kept for backwards compat with existing main.py/app.py
                 api_key: str = None):
        # If caller passes legacy api_key (old interface), inject into config
        if api_key and not router_config.get("groq_api_key"):
            router_config["groq_api_key"] = api_key

        self.router         = LLMRouter(router_config)
        # Keep .model for any code that inspects it (e.g., eval harness)
        self.model          = router_config.get("groq_model", "llama-3.3-70b-versatile")
        self.vs             = VectorStore(knowledge_chunks)
        self.scorer         = SymptomScorer(severity_map)
        self.precaution_map = precaution_map or {}
        self._reset_state()
        print("Care-AI ready!\n")

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
        # Stores the new symptom text while waiting for relation clarification
        self.pending_relation_check = None
        # Tracks how many turns have elapsed since the current complaint started.
        # Reset to 0 on every new complaint so FOLLOWUP_PROMPT knows this is
        # a fresh complaint even when old history is still present.
        self.new_complaint_turn = 0
        # Explicit list of follow-up questions already asked this complaint.
        # Passed to FOLLOWUP_PROMPT so LLM never repeats one.
        self.asked_followup_questions: list = []

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

        # Always pin the current structured SOCRATES data verbatim so clinical
        # fields (trigger_activity, site, onset, etc.) survive compression even
        # if the LLM summary paraphrases or drops them.
        collected_clean = {
            k: v for k, v in self.collected_info.items()
            if v and str(v).strip() not in ('', '...', 'null', 'None')
        }
        structured_pin = ""
        if collected_clean:
            structured_pin = (
                "\n\n[PINNED SOCRATES DATA — preserve exactly as-is]:\n"
                + json.dumps(collected_clean, indent=2)
            )

        summary_prompt = (
            "Summarise this medical conversation in 4 sentences. "
            "Preserve: chief complaint, top diagnosis, triage level, and any "
            "key clinical details mentioned. Do NOT invent information.\n\n"
            + old_text
        )
        try:
            resp_text = self.router.complete(
                messages=[{"role": "user", "content": summary_prompt}],
                temp=0.0, max_tokens=180,
            )
            new_summary = resp_text + structured_pin
            self.session_summary = (
                self.session_summary + "\n" + new_summary
                if self.session_summary else new_summary
            )
        except Exception:
            # Fallback: store raw text + pinned structured data
            self.session_summary += old_text[:350] + structured_pin

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
        """Generic streaming helper."""
        if label:
            print(f"\n  {label}", flush=True)
        full_text = []
        try:
            for token in self.router.stream(messages, max_tokens=max_tokens, temp=temperature):
                full_text.append(token)
                sys.stdout.write(token)
                sys.stdout.flush()
        except KeyboardInterrupt:
            print("\n  [streaming interrupted]")
        except Exception as e:
            print(f"\n  Stream error: {e}")
        print()
        return "".join(full_text)

    def _stream_answer(self, messages: list) -> str:
        """
        [P1-B] Pass 6 — Answer generation stream.
        250 tokens ~= 120 words hard ceiling.
        """
        return self._stream(messages, label="", max_tokens=250, temperature=0.3)

    def _stream_xai(self, messages: list, label: str = "") -> str:
        """
        [P1-C] Pass 8 — XAI explanation stream.
        450 tokens — enough for 3 'Because X -> Y' bullets + closing sentence.
        """
        return self._stream(messages, label=label, max_tokens=450, temperature=0.3)

    def _wrap_prompt(self, base_prompt: str) -> str:
        return CONVERSATIONAL_WRAPPER.format(base_prompt=base_prompt)

    # ── LLM HELPERS ───────────────────────────────────────────

    def _llm(self, system, user_content, temp=0.2, max_tok=900):
        try:
            messages = [
                {"role": "system", "content": system},
                {"role": "user",   "content": user_content},
            ]
            return self.router.complete(messages, temp=temp, max_tokens=max_tok)
        except Exception as e:
            err = str(e)
            # Surface a friendly message if ALL providers are exhausted
            # Note: matches the new router error string (no emoji prefix)
            if "All LLM providers failed" in err:
                raise RuntimeError(
                    "All API providers are currently rate-limited or unavailable.\n"
                    "Options:\n"
                    "  - Wait a few minutes and retry\n"
                    "  - Add more API keys to .env\n"
                    "  - Run 'ollama serve' for unlimited local fallback"
                ) from e
            raise

    def _llm_conv(self, system, temp=0.3, max_tok=1500):
        messages = [{"role": "system", "content": system}] + self.history
        return self.router.complete(messages, temp=temp, max_tokens=max_tok)

    def _parse_json(self, raw, fallback):
        """Tolerant JSON parser — tries multiple extraction strategies."""
        if not raw or not raw.strip():
            return fallback
        text = raw.strip()

        # Strategy 1: strip markdown fences
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```\s*$', '', text)
        try:
            return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            pass

        # Strategy 2: find first { ... } block via greedy match
        m = re.search(r'(\{[\s\S]*\})', text)
        if m:
            try:
                return json.loads(m.group(1))
            except (json.JSONDecodeError, ValueError):
                pass

        # Strategy 3: find first [ ... ] block (for array responses)
        m = re.search(r'(\[[\s\S]*\])', text)
        if m:
            try:
                return json.loads(m.group(1))
            except (json.JSONDecodeError, ValueError):
                pass

        # Strategy 4: try removing trailing commas (common LLM mistake)
        cleaned = re.sub(r',\s*([}\]])', r'\1', text)
        try:
            return json.loads(cleaned)
        except (json.JSONDecodeError, ValueError):
            pass

        return fallback

    # ── EMERGENCY ─────────────────────────────────────────────

    def _is_emergency(self, msg):
        return any(kw in msg.lower() for kw in EMERGENCY_KEYWORDS)

    def _emergency_msg(self):
        return (
            "MEDICAL EMERGENCY DETECTED\n\n"
            "Please call your local emergency number (911 / 112 / 999) IMMEDIATELY\n"
            "or go to the nearest emergency room. Do not wait.\n\n"
            "Your safety is the top priority right now."
        )

    def _is_resolved(self, msg: str) -> bool:
        """Return True if user is saying their symptoms have resolved / they are better."""
        msg_lower = msg.lower().strip()
        return any(sig in msg_lower for sig in RESOLVED_SIGNALS)

    def _resolved_reply(self, condition: str) -> str:
        """Generate a warm closing message when the user says they are better."""
        prompt = (
            f"The user was assessed for {condition} and has just said their symptoms "
            f"are gone or they are feeling better.\n"
            f"Write a warm, friendly 2-sentence reply:\n"
            f"  Sentence 1: Express genuine happiness that they are feeling better.\n"
            f"  Sentence 2: One simple prevention or self-care reminder for {condition}, "
            f"and tell them they can start a new chat anytime.\n"
            f"Keep it conversational — like a caring friend, not a doctor.\n"
            f"Language: {self.user_language['language']}. No medical jargon."
        )
        try:
            return self._llm(
                "You are a warm, caring health companion. Be brief and friendly.",
                prompt, temp=0.4, max_tok=100,
            )
        except Exception:
            return (
                "That's wonderful to hear — glad you're feeling better! "
                "Take care of yourself, and feel free to start a new chat anytime."
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
        if any(v in msg_lower for v in VAGUE_RESPONSES):
            return True
        words = msg_lower.split()
        if len(words) <= 2 and not any(c.isdigit() for c in msg_lower):
            return True
        return False

    def _is_asking_question(self, msg: str) -> bool:
        """
        Return True if the user's message contains a question or new information
        that deserves a real answer rather than a wellness check-in.
        """
        msg_lower = msg.lower().strip()
        if "?" in msg:
            return True
        QUESTION_STARTERS = {
            "what", "why", "how", "when", "where", "which", "who",
            "is it", "can i", "should i", "do i", "will it", "could",
            "would", "does", "am i", "are there", "is this",
        }
        if any(msg_lower.startswith(q) or f" {q} " in msg_lower
               for q in QUESTION_STARTERS):
            return True
        if len(msg_lower.split()) > 6:
            return True
        return False

    # ── PATTERN DETECTOR — runs before LLM follow-up call ────

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

        POSITIONAL_SIGNALS = [
            "bend", "bending", "lifting", "lift", "move", "moving",
            "position", "sit", "sitting", "stand", "stretch", "twist",
            "walk", "walking", "exercise", "pressure", "touch", "press",
            "worse when", "better when", "only when",
        ]
        is_positional = any(s in all_user_text for s in POSITIONAL_SIGNALS)

        MUSCLE_SIGNALS = [
            "cramp", "cramping", "sharp", "pull", "strain", "tight",
            "sore", "ache", "stiff", "muscle", "spasm",
        ]
        is_muscular = any(s in all_user_text for s in MUSCLE_SIGNALS)

        trigger = info.get("trigger_activity")
        trigger_collected = (
            trigger and
            str(trigger).strip().lower() not in ("null", "none", "", "...")
        )

        ACTIVITY_ASKED_SIGNALS = [
            "exercise", "workout", "physical activity", "gym",
            "lifting", "sport", "exertion", "activity before",
        ]
        activity_already_asked = any(
            s in m['content'].lower()
            for m in self.history if m['role'] == 'assistant'
            for s in ACTIVITY_ASKED_SIGNALS
        )

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
        if self.new_complaint_turn <= 1:
            relevant_history = self.history[-2:]
        else:
            relevant_history = self.history

        history_str = "\n".join(
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
            for m in relevant_history
        )

        last_user_msg = next(
            (m['content'] for m in reversed(self.history) if m['role'] == 'user'), ""
        )
        if self._is_vague_response(last_user_msg):
            self.no_new_info_count += 1
        else:
            self.no_new_info_count = 0

        if self.no_new_info_count >= 2:
            print("  User gave no new info — skipping further follow-up.")
            return {
                "has_enough_info":       True,
                "diagnostic_confidence": 50,
                "collected_info":        {},
                "followup_question":     None,
                "followup_reason":       "User unable to provide more information",
                "no_new_info_count":     self.no_new_info_count,
            }

        if self.followup_count >= 5:
            print("  Max follow-ups reached — proceeding to answer.")
            return {
                "has_enough_info":       True,
                "diagnostic_confidence": 60,
                "collected_info":        {},
                "followup_question":     None,
                "followup_reason":       "Maximum follow-ups reached",
                "no_new_info_count":     self.no_new_info_count,
            }

        patterns = self._detect_symptom_patterns()
        priority_hint = patterns.get("priority_hint", "")

        if priority_hint:
            print(f"  Pattern detected: {patterns['patterns']}")

        already_asked_block = ""
        if self.asked_followup_questions:
            already_asked_block = (
                "\nQUESTIONS ALREADY ASKED THIS SESSION (DO NOT REPEAT ANY OF THESE):\n"
                + "\n".join(f"  - {q}" for q in self.asked_followup_questions)
                + "\nYou MUST ask a DIFFERENT question covering a dimension not yet explored.\n"
            )

        new_complaint_block = ""
        if self.new_complaint_turn <= 1:
            new_complaint_block = (
                "\n[NEW COMPLAINT] Turn 1 of a fresh assessment.\n"
                "The conversation history above may contain a PREVIOUS unrelated condition.\n"
                "IGNORE any confidence you might derive from the previous condition.\n"
                "Treat this as if you are starting fresh. diagnostic_confidence must be LOW (< 40)\n"
                "unless the user's CURRENT message alone provides exceptionally rich detail.\n"
                "You MUST ask at least one SOCRATES question before proceeding.\n"
                "has_enough_info MUST be false for turn 1 of a new complaint.\n"
            )

        prompt = (
            f"Conversation:\n{history_str}\n\n"
            f"SOCRATES info collected so far: {json.dumps(self.collected_info)}\n"
            f"Follow-ups asked so far: {self.followup_count}/5 (hard max)\n"
            f"Current complaint turn number: {self.new_complaint_turn}\n"
            f"User language: {self.user_language['language']}\n"
            f"Consecutive no-new-info responses: {self.no_new_info_count}\n"
            + already_asked_block
            + new_complaint_block
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

        confidence = result.get("diagnostic_confidence", 50)
        if confidence >= 65:
            result["has_enough_info"] = True

        if self.new_complaint_turn <= 1 and self.followup_count == 0:
            result["has_enough_info"] = False
            if not result.get("followup_question"):
                result["followup_question"] = self._fallback_opening_question()

        if self.followup_count >= 5:
            result["has_enough_info"] = True
            result["followup_question"] = None

        return result

    def _fallback_opening_question(self) -> str:
        """
        Safety net: generate a relevant opening SOCRATES question when
        the LLM failed to produce one on turn 1 of a new complaint.
        """
        last_user = next(
            (m['content'] for m in reversed(self.history) if m['role'] == 'user'), ""
        )
        prompt = (
            f"The user has just described a new symptom: '{last_user[:200]}'\n"
            f"Write ONE warm, specific opening question to begin a SOCRATES assessment.\n"
            f"Ask about the most important missing piece of information "
            f"(onset, location, character, or severity).\n"
            f"Language: {self.user_language['language']}. Be brief and natural."
        )
        try:
            return self._llm(
                "You are a caring medical intake assistant.",
                prompt, temp=0.3, max_tok=80,
            )
        except Exception:
            return (
                "I'd like to understand this better — "
                "can you tell me when it started and exactly where you feel it?"
            )

    # ── PASS 2.5: QUERY PRE-CLASSIFICATION ───────────────────

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
        socrates_text = " ".join(
            str(v) for v in self.collected_info.values()
            if v and str(v).strip() not in ("", "...", "null", "None")
        )
        user_msgs = [m["content"] for m in self.history if m["role"] == "user"]
        anchor_msgs = []
        if user_msgs:
            anchor_msgs.append(user_msgs[0])
        if len(user_msgs) > 1:
            anchor_msgs.append(user_msgs[-1])
        last_msgs = (socrates_text + " " + " ".join(anchor_msgs)).strip()
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
        collected_fields = [
            self.collected_info.get(f)
            for f in ['symptom_or_topic', 'site', 'character',
                      'associations', 'trigger_activity', 'other_details']
        ]
        has_collected = any(
            v and str(v).strip() not in ('', '...', 'null', 'None')
            for v in collected_fields
        )

        if not has_collected:
            parts = [
                m['content'] for m in self.history[-2:]
                if m['role'] == 'user'
            ]
        else:
            parts = [m['content'] for m in self.history[-4:] if m['role'] == 'user']
            for val in collected_fields:
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
            lines.append(f"    {c['text'][:1200]}")
        return "\n".join(lines)

    def _fmt_sources(self, chunks):
        sources = []
        for i, c in enumerate(chunks, 1):
            source  = c.get('source', 'Unknown')
            disease = c.get('disease', c.get('title', ''))
            cat     = c.get('category', '')
            label   = f"[{i}] {source}"
            if disease: label += f" -- {disease}"
            if cat:     label += f" ({cat})"
            sources.append(label)
        return sources

    # ── PASS 4: SYMPTOM SEVERITY SCORING ──────────────────────

    def _score_severity(self):
        all_text = " ".join(str(m['content']) for m in self.history if m['role'] == 'user')
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

    # ── PASS 5b: SELF-CONSISTENCY CHECK ──────────────────────
    # [P3-D] Run reasoning twice at different temperatures.
    # Triage disagreement -> always use the more urgent level (safety).

    _TRIAGE_URGENCY = {
        "SELF_CARE":   0,
        "SEE_DOCTOR":  1,
        "URGENT_CARE": 2,
        "EMERGENCY":   3,
    }

    CONSISTENCY_THRESHOLD = 65

    def _reason_consistent(self, context_str: str) -> dict:
        """
        Adaptive self-consistency:
        - Always runs pass 1 (temp=0.1)
        - Only runs pass 2 (temp=0.4) if pass 1 confidence < CONSISTENCY_THRESHOLD
          OR if differential has only 1 condition (weak evidence)
        """
        print("  [run 1/2]", end=" ", flush=True)
        result_a = self._reason(context_str)

        confidence_a = result_a.get('confidence_score', 50)
        diff_a_count = len(result_a.get('differential_diagnosis', []))
        needs_second = (
            confidence_a < self.CONSISTENCY_THRESHOLD
            or diff_a_count < 2
        )

        if not needs_second:
            uncertainty = list(result_a.get('uncertainty_zones', []))
            uncertainty.append(
                f"Self-consistency: skipped (confidence {confidence_a}% >= "
                f"{self.CONSISTENCY_THRESHOLD}% threshold — single pass sufficient)."
            )
            result_a['uncertainty_zones'] = uncertainty
            print(f"\n  Self-consistency skipped (confidence={confidence_a}%)")
            return result_a

        print("  [run 2/2]", end=" ", flush=True)
        recent   = self.history[-8:]
        conv_str = "\n".join(
            f"{'User' if m['role'] == 'user' else 'Bot'}: {m['content']}"
            for m in recent
        )
        socrates_str = json.dumps(self.collected_info, indent=2)
        raw_b = self._llm(
            REASONING_PROMPT,
            f"Conversation:\n{conv_str}\n\n"
            f"SOCRATES clinical data:\n{socrates_str}\n\n"
            f"{context_str}",
            temp=0.4, max_tok=1200,
        )
        result_b = self._parse_json(raw_b, result_a)

        diff_a = result_a.get('differential_diagnosis', [])
        diff_b = result_b.get('differential_diagnosis', [])
        top_a  = diff_a[0].get('condition', '').lower().strip() if diff_a else ''
        top_b  = diff_b[0].get('condition', '').lower().strip() if diff_b else ''

        triage_a = result_a.get('triage_level', 'SEE_DOCTOR')
        triage_b = result_b.get('triage_level', 'SEE_DOCTOR')

        uncertainty = list(result_a.get('uncertainty_zones', []))
        confidence  = result_a.get('confidence_score', 50)

        if triage_a != triage_b:
            urgency_a = self._TRIAGE_URGENCY.get(triage_a, 1)
            urgency_b = self._TRIAGE_URGENCY.get(triage_b, 1)
            if urgency_b > urgency_a:
                result_a['triage_level']  = triage_b
                result_a['triage_reason'] = (
                    f"[Escalated by self-consistency check] "
                    f"{result_b.get('triage_reason', '')}"
                )
            uncertainty.append(
                f"Triage uncertainty: two reasoning passes disagreed "
                f"({triage_a} vs {triage_b}) — using more urgent level."
            )
            confidence = max(30, confidence - 10)

        conditions_agree = (
            top_a == top_b
            or top_a in top_b
            or top_b in top_a
            or (len(top_a) > 6 and len(top_b) > 6 and
                any(word in top_b for word in top_a.split() if len(word) > 5))
        )

        if not conditions_agree and top_a and top_b:
            uncertainty.append(
                f"Diagnosis uncertainty: two reasoning passes disagreed on "
                f"top condition ('{diff_a[0].get('condition', top_a)}' vs "
                f"'{diff_b[0].get('condition', top_b)}'). "
                f"Using more conservative estimate — see doctor for confirmation."
            )
            confidence = max(30, confidence - 15)
            print(f"\n  Self-consistency disagreement: '{top_a}' vs '{top_b}'")
        else:
            uncertainty_msg = "Self-consistency check passed — both reasoning passes agreed."
            if uncertainty_msg not in uncertainty:
                uncertainty.append(uncertainty_msg)
            print(f"\n  Self-consistency passed: both runs -> '{top_a}'")

        result_a['uncertainty_zones'] = uncertainty
        result_a['confidence_score']  = confidence
        return result_a

    # ── ZERO-OVERLAP DIFFERENTIAL FILTER ─────────────────────────
    # [P4] Drop differentials with no symptom evidence from user report.

    def _filter_zero_overlap_differentials(self, differentials: list) -> list:
        """
        Remove differentials whose supporting_symptoms have zero overlap
        with symptoms actually mentioned by the user.
        Always keeps at least 1 differential (rank 1) even if no overlap.
        """
        if not differentials:
            return differentials

        all_user_text = " ".join(
            m["content"].lower() for m in self.history if m["role"] == "user"
        )
        all_user_text += " " + " ".join(
            str(v).lower() for v in self.collected_info.values() if v
        )

        filtered = []
        for dx in differentials:
            supporting = [str(s).lower() for s in dx.get("supporting_symptoms", [])]
            if not supporting:
                filtered.append(dx)
                continue
            overlap = any(sym in all_user_text for sym in supporting)
            if overlap:
                filtered.append(dx)
            else:
                print(f"  Dropped zero-overlap differential: {dx.get('condition')}")

        return filtered if filtered else differentials[:1]

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
            f"-- YOU MUST respond ONLY in this language. Never switch languages.\n\n"
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
    # [P3-A/B/C] Per-sentence verification against knowledge base

    def _extract_claims(self, answer: str) -> list:
        """Split answer into individual checkable claims (sentences)."""
        sentences = re.split(r'(?<=[.!?])\s+', answer.strip())
        claims = []
        for s in sentences:
            s = s.strip()
            if len(s.split()) < 4:
                continue
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

            if computed_risk in ("MEDIUM", "HIGH") and not fc.get("corrected_answer"):
                verified_sentences = [
                    c["claim"] for c in claim_checks if c.get("supported", False)
                ]
                if len(verified_sentences) >= 2:
                    join_prompt = (
                        "Rewrite these verified medical facts as a single coherent answer "
                        "in 80-100 words. Keep all the factual content. "
                        f"Language: {self.user_language['language']}. "
                        "Do not add new claims. Do not change the facts.\n\n"
                        + "\n".join(f"- {s}" for s in verified_sentences)
                    )
                    try:
                        fc["corrected_answer"] = self._llm(
                            "You are a medical writer. Rewrite clearly and concisely.",
                            join_prompt, temp=0.2, max_tok=200,
                        )
                    except Exception:
                        fc["corrected_answer"] = " ".join(verified_sentences)
                elif verified_sentences:
                    fc["corrected_answer"] = (
                        verified_sentences[0]
                        + " Please see a doctor for a full assessment."
                    )
                else:
                    fc["corrected_answer"] = (
                        "Based on your symptoms, I recommend seeing a doctor "
                        "for a proper assessment. I was unable to verify specific "
                        "claims against my medical knowledge base with enough confidence."
                    )

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
    # [P1-C] Rewritten: cause -> reasoning -> recommendation format

    def _xai_explain_streamed(self, reasoning, factcheck):
        last_user = next(
            (m['content'] for m in reversed(self.history) if m['role'] == 'user'), ""
        )
        diff = reasoning.get('differential_diagnosis', [])
        top  = diff[0]['condition'] if diff else 'unknown'

        supporting   = diff[0].get('supporting_symptoms', []) if diff else []
        key_evidence = diff[0].get('key_evidence', '') if diff else ''
        uncertainty  = reasoning.get('uncertainty_zones', ['nothing specific'])
        triage       = reasoning.get('triage_level', 'SEE_DOCTOR')
        triage_reason = reasoning.get('triage_reason', '')
        sources      = reasoning.get('knowledge_sources', [])

        # [FIX v2.1] All .join() calls coerce elements to str() to prevent
        # TypeError when the LLM returns integers in list fields
        # (e.g. confidence: 85 leaking into supporting_symptoms).
        prompt = (
            f"User's message: {last_user}\n"
            f"User's language: {self.user_language['language']}\n\n"
            f"Top condition identified: {top}\n"
            f"Key symptom pointing to this: {key_evidence}\n"
            f"Supporting symptoms: {', '.join(str(s) for s in supporting)}\n"
            f"What sources were used: {', '.join(str(s) for s in sources)}\n"
            f"What the AI is uncertain about: {', '.join(str(u) for u in uncertainty)}\n"
            f"Triage decision: {triage}\n"
            f"Reason for triage: {triage_reason}\n"
            f"Fact-check result: {factcheck.get('verdict', 'N/A')}\n\n"
            f"Write 3 bullets in 'Because X -> Y' format, then one warm closing sentence.\n"
            f"Respond entirely in: {self.user_language['language']}"
        )
        messages = [
            {"role": "system", "content": self._wrap_prompt(XAI_EXPLAINER_PROMPT)},
            {"role": "user",   "content": prompt},
        ]
        return self._stream_xai(messages, label="Why did I give this answer?")

    # ── PASS 9: TREATMENT GUIDANCE ────────────────────────────

    def _get_selfcare_guidance(self, top_condition: str) -> dict:
        """
        Lightweight treatment guidance for SELF_CARE triage.
        Uses a shorter prompt (max 300 tokens vs 600) — saves ~300 tokens.
        """
        dataset_precautions = []
        condition_lower = top_condition.lower()
        for disease, precautions in self.precaution_map.items():
            if disease.lower() in condition_lower or condition_lower in disease.lower():
                dataset_precautions = precautions
                break

        precaution_note = ""
        if dataset_precautions:
            precaution_note = (
                f"\nDataset precautions: "
                + ", ".join(str(p) for p in dataset_precautions[:3] if p)
            )

        prompt = (
            f"Condition: {top_condition} (SELF-CARE level — mild case)\n"
            f"Language: {self.user_language['language']}"
            f"{precaution_note}\n\n"
            f"Give 2-3 simple home care steps and one escalation rule.\n"
            f"NO drug names. Return JSON only:\n"
            f'{{"home_care": ["step1","step2"],'
            f'"escalation_timeline": "If no improvement in X days see doctor.",'
            f'"lifestyle_tips": ["tip1"],'
            f'"what_to_avoid": ["avoid1"]}}'
        )
        raw = self._llm(TREATMENT_PROMPT, prompt, temp=0.2, max_tok=300)
        return self._parse_json(raw, {
            "home_care":           ["Rest well and stay hydrated.", "Monitor your symptoms closely."],
            "escalation_timeline": "If no improvement in 3 days, consult a doctor.",
            "lifestyle_tips":      ["Maintain a healthy routine."],
            "what_to_avoid":       ["Avoid strenuous activity until you feel better."],
        })

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

    # ── PASS 10: RED FLAG ALERTS ──────────────────────────────

    def _get_red_flags(self, top_condition: str) -> dict:
        """Get condition-specific red flag symptoms to watch for."""
        prompt = (
            f"Condition: {top_condition}\n"
            f"User language: {self.user_language['language']}"
        )
        raw = self._llm(REDFLAGS_PROMPT, prompt, temp=0.1, max_tok=300)
        return self._parse_json(raw, {
            "red_flags": [
                "High fever (above 39C / 102F)",
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

    # ── NEW COMPLAINT DETECTOR ───────────────────────────────

    _SYMPTOM_SEEDS = {
        "pain", "ache", "aching", "hurt", "hurting", "sore", "soreness",
        "fever", "temperature", "cough", "coughing", "cold", "flu",
        "headache", "migraine", "nausea", "vomiting", "diarrhea", "diarrhoea",
        "constipation", "bloating", "cramp", "cramping", "rash", "rashes",
        "itch", "itching", "swelling", "swollen", "fatigue", "tired",
        "dizziness", "dizzy", "breathless", "breathing", "chest",
        "stomach", "abdomen", "abdominal", "back", "neck", "throat",
        "nose", "nostril", "ear", "eye", "skin", "joint", "muscle",
        "bleeding", "blood", "infection", "infected", "feeling",
        "discharge", "burning", "numbness", "weakness", "anxiety",
        "depression", "stress", "insomnia", "sleep",
        "mouth", "oral", "palate", "tongue", "gum", "gums", "tooth",
        "teeth", "jaw", "lip", "lips", "cheek", "tonsil", "tonsils",
        "ulcer", "ulcers", "blister", "rough", "bumps", "lump", "lesion",
        "dry mouth", "taste", "chewing", "swallowing",
        "earache", "hearing", "ringing", "sinus", "sinuses", "nasal",
        "runny nose", "blocked nose", "sneezing", "hoarse", "hoarseness",
        "voice", "tinnitus",
        "nodule", "patch", "peeling", "flaking", "dry", "tingling",
        "prickling",
        "urine", "urination", "bowel", "stool", "gas", "indigestion",
        "heartburn", "reflux", "appetite", "weight",
    }

    _UNRELATED_SIGNALS = {
        "different", "unrelated", "separate", "another issue", "new problem",
        "not related", "nothing to do", "different problem", "new complaint",
        "different issue", "not the same", "something else", "other issue",
        "different symptom", "new symptom", "not connected",
    }

    _RELATED_SIGNALS = {
        "yes", "yeah", "related", "same", "connected", "linked", "part of",
        "because of", "due to", "from the", "same issue", "same problem",
        "think so", "probably", "maybe related", "might be",
    }

    def _has_symptom_keyword(self, msg: str) -> bool:
        """Return True if message contains a new symptom keyword."""
        msg_lower = msg.lower()
        STATUS_ONLY = {
            "yes", "no", "yeah", "nope", "better", "worse", "same",
            "still", "ok", "okay", "fine", "not really", "a bit",
            "improving", "improved", "much better", "little better",
        }
        words = set(msg_lower.split())
        if words and words.issubset(STATUS_ONLY):
            return False
        return any(seed in msg_lower for seed in self._SYMPTOM_SEEDS)

    def _detect_relation_reply(self, msg: str) -> str:
        """
        After asking "is this related to your [condition]?", classify reply.
        Returns: 'related' | 'unrelated' | 'unsure'
        """
        msg_lower = msg.lower()
        if any(sig in msg_lower for sig in self._UNRELATED_SIGNALS):
            return "unrelated"
        if any(sig in msg_lower for sig in self._RELATED_SIGNALS):
            return "related"
        try:
            prompt = (
                f"The user was assessed for: {self.last_top_condition}.\n"
                f"They then said: '{msg}'\n"
                f"Is this new symptom RELATED to {self.last_top_condition}, "
                f"UNRELATED (a separate new problem), or UNSURE?\n"
                f"Reply with exactly one word: RELATED, UNRELATED, or UNSURE."
            )
            raw = self._llm(
                "You are a triage classifier. Reply with exactly one word only.",
                prompt, temp=0.0, max_tok=5,
            ).strip().upper()
            if raw in ("RELATED", "UNRELATED", "UNSURE"):
                return raw.lower()
        except Exception:
            pass
        return "unsure"

    def _ask_relation_question(self, new_symptom: str, condition: str) -> str:
        """Generate a one-sentence clarifying question about relation."""
        prompt = (
            f"The user was previously assessed for: {condition}.\n"
            f"They now mention: '{new_symptom}'.\n"
            f"Write ONE warm, natural sentence asking if this new symptom "
            f"is related to their {condition} or if it is a completely "
            f"separate new issue they want help with.\n"
            f"Language: {self.user_language['language']}."
        )
        return self._llm(
            "You are a caring medical assistant. Be warm and brief.",
            prompt, temp=0.3, max_tok=60,
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

        # ── Resolved / better check ───────────────────────────
        if (
            self.state == ConvState.ANSWERED
            and self._is_resolved(user_message)
            and self.last_top_condition
        ):
            self.state = ConvState.RESOLVED
            reply = self._resolved_reply(self.last_top_condition)
            self._add_turn("user",      user_message)
            self._add_turn("assistant", reply)
            print(f"\nCare-AI (Resolved): {reply}\n")
            return {"type": "resolved", "message": reply}

        # ── If already RESOLVED, treat any new message as a fresh complaint ──
        if self.state == ConvState.RESOLVED:
            self.collected_info           = {}
            self.followup_count           = 0
            self.no_new_info_count        = 0
            self.followup_asked           = True
            self.new_complaint_turn       = 0
            self.asked_followup_questions = []
            self.state                    = ConvState.GATHERING

        # ── New symptom / relation detection ─────────────────────
        if self.pending_relation_check and self.state == ConvState.ANSWERED:
            relation = self._detect_relation_reply(user_message)
            print(f"  Relation reply detected: {relation}")

            if relation == "unrelated":
                self.collected_info             = {}
                self.followup_count             = 0
                self.no_new_info_count          = 0
                self.followup_asked             = True
                self.pending_relation_check     = None
                self.state                      = ConvState.GATHERING
                self.new_complaint_turn         = 0
                self.asked_followup_questions   = []
                print("  Unrelated complaint — SOCRATES state reset.")

            else:
                self.pending_relation_check = None
                self.followup_asked         = True
                print("  Related complaint — keeping conversation context.")

        elif (
            self.state == ConvState.ANSWERED
            and self._has_symptom_keyword(user_message)
            and self.last_top_condition
        ):
            self.pending_relation_check   = user_message
            self.new_complaint_turn       = 0
            self.asked_followup_questions = []
            relation_q = self._ask_relation_question(
                user_message, self.last_top_condition
            )
            self._add_turn("user",      user_message)
            self._add_turn("assistant", relation_q)
            print(f"\nCare-AI (Relation check): {relation_q}\n")
            return {"type": "relation_check", "message": relation_q}

        elif (
            self.state == ConvState.ANSWERED
            and not self.followup_asked
            and self.last_top_condition
            and not self._is_asking_question(user_message)
            and self._is_vague_response(user_message)
        ):
            self.followup_asked = True
            followup_msg = self._symptom_followup_prompt(self.last_top_condition)
            self._add_turn("user",      user_message)
            self._add_turn("assistant", followup_msg)
            print(f"\nCare-AI (Follow-up): {followup_msg}\n")
            return {"type": "followup_check", "message": followup_msg}

        # ── Language detection ────────────────────────────────
        if self.state == ConvState.ANSWERED and not self.followup_asked:
            self.followup_asked = True

        _msg_words = len(user_message.strip().split())
        if len(self.history) == 0:
            self.user_language = self._detect_language(user_message)
        elif len(self.history) % 6 == 0 and _msg_words >= 4:
            _detected = self._detect_language(user_message)
            _cur_code = self.user_language.get("code", "en")
            _new_code = _detected.get("code", "en")
            if _cur_code == "en" or _new_code == _cur_code:
                self.user_language = _detected

        self._add_turn("user", user_message)
        self.new_complaint_turn += 1

        # ── SOCRATES follow-up engine ─────────────────────────
        if self.no_new_info_count < 2:
            print("  Checking if more info needed...", end=" ", flush=True)
            fu = self._check_followup()
            print("done")

            for k, v in fu.get('collected_info', {}).items():
                if v and str(v).strip() not in ('', '...', 'null', 'None'):
                    self.collected_info[k] = v

            confidence = fu.get('diagnostic_confidence', 50)
            has_enough = fu.get('has_enough_info', True)

            if not has_enough and fu.get('followup_question'):
                self.followup_count += 1
                q = fu['followup_question']
                self.asked_followup_questions.append(q)
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
            print("  Proceeding to answer — user gave no new info twice.")

        self.state = ConvState.READY

        print("  Pre-classifying query...",     end=" ", flush=True)
        query_category = self._classify_query()
        print(f"done ({query_category})")

        print("  Retrieving context...",     end=" ", flush=True)
        chunks      = self._retrieve(k=6, query_category=query_category)
        context_str = self._fmt_context(chunks)
        sources     = self._fmt_sources(chunks)
        print(f"done ({len(chunks)} docs)")

        print("  Scoring symptoms...",       end=" ", flush=True)
        sev_result = self._score_severity()
        self.severity_result = sev_result
        print("done")

        print("  Differential diagnosis (self-consistency)...", end=" ", flush=True)
        reasoning = self._reason_consistent(context_str)
        self.last_triage        = reasoning.get('triage_level', 'SEE_DOCTOR')
        self.last_specialist    = reasoning.get('specialist', 'General Practitioner')
        self.last_differential  = self._filter_zero_overlap_differentials(
            reasoning.get('differential_diagnosis', [])
        )
        diff = self.last_differential
        self.last_top_condition = diff[0]['condition'] if diff else ""
        print("done")

        print("  Generating treatment guidance...", end=" ", flush=True)
        if self.last_triage == "SELF_CARE":
            treatment = self._get_selfcare_guidance(self.last_top_condition)
            red_flags = self._get_red_flags(self.last_top_condition)
        else:
            treatment = self._get_treatment_guidance(self.last_top_condition, self.last_triage)
            red_flags = self._get_red_flags(self.last_top_condition)
        print("done")

        from utils.display import print_triage_header
        print_triage_header(
            self.last_triage,
            self.last_differential,
            self.last_specialist,
        )

        answer = self._answer_streamed(context_str, reasoning, sev_result)

        print("  Claim-level fact-checking...", end=" ", flush=True)
        fc = self._factcheck(answer, context_str)
        checks     = fc.get("claim_checks", [])
        n_verified = sum(1 for c in checks if c.get("supported", False))
        n_total    = len(checks)
        risk       = fc.get("hallucination_risk", "N/A")
        print(f"done ({n_verified}/{n_total} claims verified | Risk: {risk})")

        final = (
            fc['corrected_answer']
            if fc.get('corrected_answer')
            and fc.get('hallucination_risk') in ('MEDIUM', 'HIGH')
            else answer
        )
        if final != answer:
            print("  Auto-corrected based on fact-check.")

        xai = self._xai_explain_streamed(reasoning, fc)

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
            "nearby_doctors":     None,
        }

    # ── DOCTOR FINDER ─────────────────────────────────────────

    def find_nearby_doctors(self, user_lat: float, user_lon: float) -> dict:
        """
        Look up nearby doctors for the current specialist recommendation.
        Called by ui/app.py after it captures browser geolocation.
        Returns None if triage is SELF_CARE or no specialist known.
        """
        if self.last_triage == "SELF_CARE":
            return None
        if not self.last_specialist:
            return None
        try:
            return find_nearby_doctors(
                specialist=self.last_specialist,
                user_lat=user_lat,
                user_lon=user_lon,
            )
        except Exception as e:
            return {
                "specialist":     self.last_specialist,
                "results":        [],
                "radius_used_km": 5,
                "error": f"Could not fetch nearby doctors: {e}",
            }

    def reset(self):
        self._reset_state()
        self.router.reset_providers()
        print("[Session reset]\n")

    def export_log(self, path="logs/session_log.json"):
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.xai_logs, f, indent=2)
        print(f"Log exported -> {path}")