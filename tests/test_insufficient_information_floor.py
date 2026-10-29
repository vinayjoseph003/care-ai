"""
tests/test_insufficient_information_floor.py — Safety triage floor tests for insufficient information.

Tests requirements:
1. Allowlist of approved clinical fields and meaningful clinical evidence:
   - Unrelated fields (user preferences, metadata, session data) cannot satisfy clinical info check.
   - Generic values like "yes" in other_details or "ok" in character do not satisfy check.
   - Non-meaningful / trivial values (idk, null, none) do not satisfy clinical info check.
   - Meaningful primary symptom/topic in symptom_or_topic or affirmative symptom description in approved fields satisfies check.
2. Symptom-seed false positives & negation handling:
   - Broad conversational tokens ("I am back", "feeling fine", "cold outside", "noisy neighbors sleep") do NOT satisfy check.
   - Negated symptoms ("no pain, no fever", "mujhe bukhar nahi hai", "naku noppi ledu", "मुझे बुखार नहीं है", "నాకు జ్వరం లేదు") do NOT satisfy check.
   - Contrastive conjunction with unnegated symptom ("I have no fever, but I have a headache") satisfies check.
3. Multilingual native scripts:
   - Devanagari Hindi ("मुझे सिरदर्द है", "मुझे बुखार और खांसी है") and Telugu script ("నాకు జ్వరం ఉంది", "నాకు తలనొప్పిగా ఉంది") satisfy check.
4. Cross-complaint history isolation:
   - An earlier resolved headache conversation in self.history cannot validate a fresh uninformative complaint.
5. Conservative rule distinguishing inadequate information from repeated uninformative answers:
   - When adequate clinical information already exists, repeated "idk" (no_new_info_count >= 2) alone does NOT escalate SELF_CARE.
   - When adequate clinical information does NOT exist, SELF_CARE is escalated to SEE_DOCTOR.
6. Preservation of emergency triage:
   - Emergency triages (EMERGENCY, URGENT_CARE) and red flags are NEVER downgraded.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

for _mod in ("scipy", "scipy.sparse"):
    if _mod not in sys.modules:
        try:
            __import__(_mod)
        except ImportError:
            sys.modules[_mod] = MagicMock()

from core.triage_floor import (
    apply_triage_floor,
    has_adequate_clinical_info,
    APPROVED_CLINICAL_FIELDS,
)
from bot.careai import CareAI, ConvState


class TestInsufficientInformationFloorPure(unittest.TestCase):
    """Pure unit tests for has_adequate_clinical_info and apply_triage_floor."""

    def test_unrelated_fields_and_metadata_cannot_satisfy_clinical_check(self):
        """Unrelated fields such as preferences, session metadata, or arbitrary keys cannot satisfy check."""
        unrelated_payloads = [
            {"user_preference": "dark_mode"},
            {"preferred_language": "English", "theme": "compact"},
            {"session_id": "sess_123456", "user_id": "user_789"},
            {"metadata": {"browser": "Chrome", "os": "Windows"}},
            {"arbitrary_field": "some arbitrary text", "notes": "patient arrived"},
            {"favorite_color": "blue", "font_size": "large"},
        ]
        for payload in unrelated_payloads:
            with self.subTest(payload=payload):
                self.assertFalse(
                    has_adequate_clinical_info(payload),
                    f"Payload with unrelated fields unexpectedly satisfied clinical check: {payload}",
                )

    def test_generic_affirmative_and_status_values_cannot_satisfy_clinical_check(self):
        """Generic values like 'yes' in other_details or 'ok' in character cannot establish adequacy."""
        generic_payloads = [
            {"other_details": "yes"},
            {"character": "ok"},
            {"character": "okay"},
            {"other_details": "fine"},
            {"trigger_activity": "sitting"},
            {"severity": "4"},
            {"other_details": "yeah", "character": "good"},
            {"onset": "yesterday"},
        ]
        for payload in generic_payloads:
            with self.subTest(payload=payload):
                self.assertFalse(
                    has_adequate_clinical_info(payload),
                    f"Generic payload unexpectedly satisfied check: {payload}",
                )

    def test_primary_symptom_or_topic_satisfies_check_when_meaningful(self):
        """Primary symptom_or_topic with a meaningful non-trivial value satisfies check."""
        valid_topics = [
            {"symptom_or_topic": "mild headache"},
            {"symptom_or_topic": "fever and chills"},
            {"symptom_or_topic": "cough"},
            {"symptom_or_topic": "sore throat"},
            {"symptom_or_topic": "chest tightness"},
        ]
        for payload in valid_topics:
            with self.subTest(payload=payload):
                self.assertTrue(
                    has_adequate_clinical_info(payload),
                    f"Valid symptom_or_topic payload failed check: {payload}",
                )

    def test_approved_clinical_fields_with_symptom_satisfies_check(self):
        """Approved fields with substantive symptom descriptions satisfy check."""
        valid_payloads = [
            {"site": "mild headache"},
            {"associations": "fever"},
            {"character": "sharp stomach pain"},
            {"other_details": "severe migraine"},
        ]
        for payload in valid_payloads:
            with self.subTest(payload=payload):
                self.assertTrue(has_adequate_clinical_info(payload))

    def test_has_adequate_clinical_info_with_empty_or_trivial_fields(self):
        """Approved fields with empty, placeholder, or uninformative values must NOT satisfy check."""
        trivial_payloads = [
            {},
            None,
            {"symptom_or_topic": ""},
            {"symptom_or_topic": "   "},
            {"symptom_or_topic": "null", "site": "none"},
            {"symptom_or_topic": "idk", "character": "not sure"},
            {"symptom_or_topic": "...", "other_details": "i don't know"},
            {"symptom_or_topic": "unknown", "trigger_activity": "n/a"},
            {"symptom_or_topic": "nope", "associations": "nil"},
            {"symptom_or_topic": "nothing", "site": "undefined"},
        ]
        for payload in trivial_payloads:
            with self.subTest(payload=payload):
                self.assertFalse(has_adequate_clinical_info(payload))

    def test_conversational_false_positives_do_not_satisfy_check(self):
        """Broad conversational tokens ('back', 'feeling', 'sleep', 'voice') must not count as symptoms."""
        conversational_phrases = [
            "I am back",
            "I'm feeling fine",
            "I feel fine",
            "It is freezing cold outside",
            "I am feeling bored",
            "I could not sleep because of my noisy neighbors",
            "I lost my voice",
            "hello what is this",
            "please help me",
        ]
        for phrase in conversational_phrases:
            with self.subTest(phrase=phrase):
                self.assertFalse(
                    has_adequate_clinical_info({}, user_text=phrase),
                    f"Conversational phrase unexpectedly counted as symptom: '{phrase}'",
                )

    def test_negated_symptoms_do_not_satisfy_check(self):
        """Negated symptoms across English, Hindi, and Telugu must not satisfy check."""
        negated_phrases = [
            "I have no pain, no fever",
            "I do not have any pain",
            "I don't have a headache.",
            "No cough, no cold",
            "I'm not feeling well.",
            "I have no symptoms.",
            "without any pain or swelling",
            "mujhe bukhar nahi hai",
            "naku noppi ledu",
            "मुझे बुखार नहीं है",
            "मुझे कोई लक्षण नहीं है",
            "నాకు జ్వరం లేదు",
            "నాకు ఎటువంటి లక్షణాలు లేవు",
        ]
        for phrase in negated_phrases:
            with self.subTest(phrase=phrase):
                self.assertFalse(
                    has_adequate_clinical_info({}, user_text=phrase),
                    f"Negated symptom phrase unexpectedly counted as symptom: '{phrase}'",
                )

    def test_multilingual_affirmative_symptoms_satisfy_check(self):
        """Affirmative symptom terms in English, Hindi (Devanagari/Latin), and Telugu (Script/Latin) satisfy check."""
        affirmative_phrases = [
            "I have a mild headache",
            "I have a sore throat",
            "My back pain has been bothering me",
            "mujhe bukhar hai",
            "naku noppi undi",
            "मुझे सिरदर्द है",
            "मुझे बुखार और खांसी है",
            "నాకు జ్వరం ఉంది",
            "నాకు తలనొప్పిగా ఉంది",
        ]
        for phrase in affirmative_phrases:
            with self.subTest(phrase=phrase):
                self.assertTrue(
                    has_adequate_clinical_info({}, user_text=phrase),
                    f"Affirmative symptom phrase failed check: '{phrase}'",
                )

    def test_contrastive_conjunction_preserves_unnegated_symptom(self):
        """Contrastive conjunctions preserve the unnegated symptom in compound sentences."""
        compound_phrases = [
            "I have no fever, but I have a headache",
            "I have no fever, but a severe headache.",
            "No chest pain, but I can't breathe.",
            "I have pain, but no fever.",
            "no cough however severe back pain",
            "mujhe bukhar nahi hai lekin dard hai",
            "मुझे दर्द है लेकिन बुखार नहीं",
            "నాకు నొప్పి ఉంది కానీ జ్వరం లేదు",
        ]
        for phrase in compound_phrases:
            with self.subTest(phrase=phrase):
                self.assertTrue(
                    has_adequate_clinical_info({}, user_text=phrase),
                    f"Compound phrase with contrastive conjunction failed check: '{phrase}'",
                )

    def test_floor_escalates_on_insufficient_info(self):
        """When has_sufficient_info is False, SELF_CARE is escalated to SEE_DOCTOR."""
        triage, reason = apply_triage_floor(
            triage="SELF_CARE",
            differential=[{"condition": "Tension Headache"}],
            top_condition="Tension Headache",
            confidence=85,
            severity_level="LOW",
            conditions_agree=True,
            has_sufficient_info=False,
        )
        self.assertEqual(triage, "SEE_DOCTOR")
        self.assertEqual(reason, "INSUFFICIENT_INFORMATION")

    def test_floor_preserves_self_care_when_adequate_info_exists_even_with_repeated_idk(self):
        """When adequate clinical information already exists, repeated 'idk' does NOT alone escalate SELF_CARE."""
        triage, reason = apply_triage_floor(
            triage="SELF_CARE",
            differential=[{"condition": "Tension Headache"}],
            top_condition="Tension Headache",
            confidence=85,
            severity_level="LOW",
            conditions_agree=True,
            has_sufficient_info=True,
            no_new_info_count=2,
        )
        self.assertEqual(triage, "SELF_CARE")
        self.assertIsNone(reason)

    def test_floor_preserves_emergency_and_urgent_care_unconditionally(self):
        """Emergency and urgent care triages are NEVER downgraded, even with insufficient info."""
        triage_em, reason_em = apply_triage_floor(
            triage="EMERGENCY",
            differential=[{"condition": "Myocardial Infarction"}],
            top_condition="Myocardial Infarction",
            confidence=90,
            severity_level="HIGH",
            conditions_agree=True,
            has_sufficient_info=False,
            no_new_info_count=3,
        )
        self.assertEqual(triage_em, "EMERGENCY")
        self.assertIsNone(reason_em)

        triage_uc, reason_uc = apply_triage_floor(
            triage="URGENT_CARE",
            differential=[{"condition": "Acute Appendicitis"}],
            top_condition="Acute Appendicitis",
            confidence=80,
            severity_level="HIGH",
            conditions_agree=True,
            has_sufficient_info=False,
            no_new_info_count=2,
        )
        self.assertEqual(triage_uc, "URGENT_CARE")
        self.assertIsNone(reason_uc)


class TestInsufficientInformationPipeline(unittest.TestCase):
    """Pipeline integration tests with CareAI."""

    def _setup_bot(self, reasoning_triage="SELF_CARE", confidence=85, collected=None, no_new_info_count=0):
        mock_router = MagicMock()
        mock_vs = MagicMock()
        mock_vs.search.return_value = []
        mock_vs.search_by_symptoms.return_value = []
        mock_scorer = MagicMock()
        mock_scorer.extract_symptoms_from_text.return_value = []
        mock_scorer.score.return_value = {}

        bot = CareAI(
            vector_store=mock_vs,
            symptom_scorer=mock_scorer,
            router=mock_router,
        )
        bot._detect_language = MagicMock(return_value={"language": "English", "code": "en"})
        bot.no_new_info_count = no_new_info_count
        bot.collected_info = dict(collected or {})

        bot._check_followup = MagicMock(return_value={
            "has_enough_info": True,
            "diagnostic_confidence": 75,
            "collected_info": dict(collected or {}),
            "followup_question": None,
        })
        bot._classify_query = MagicMock(return_value="SYMPTOM")
        bot._retrieve = MagicMock(return_value=[])
        bot._llm = MagicMock(return_value="{}")

        canned_reasoning = {
            "triage_level": reasoning_triage,
            "differential_diagnosis": [
                {
                    "rank": 1,
                    "condition": "Mild Headache",
                    "probability": "HIGH",
                    "supporting_symptoms": ["headache"],
                    "key_evidence": "mild pain",
                }
            ],
            "specialist": "General Practitioner",
            "confidence_score": confidence,
            "conditions_agree": True,
            "triage_reason": "Based on symptoms",
            "uncertainty_zones": [],
            "knowledge_sources": ["medical guide"],
        }
        bot._reason_consistent = MagicMock(return_value=canned_reasoning)
        bot._factcheck = MagicMock(return_value={
            "claim_checks": [],
            "hallucination_risk": "LOW",
            "verdict": "VERIFIED",
        })

        recorded_prompts = []

        def fake_stream_answer(messages):
            recorded_prompts.append({"type": "answer", "messages": messages})
            return "Answer [1]."

        def fake_stream_xai(messages, label=""):
            recorded_prompts.append({"type": "xai", "messages": messages})
            return "Explanation."

        bot._stream_answer = fake_stream_answer
        bot._stream_xai = fake_stream_xai

        return bot, recorded_prompts

    def test_pipeline_floors_when_only_unrelated_metadata_provided(self):
        """When intake contains only metadata/preferences and no clinical data, SELF_CARE is floored to SEE_DOCTOR."""
        bot, prompts = self._setup_bot(
            reasoning_triage="SELF_CARE",
            confidence=85,
            collected={"user_preference": "dark_mode", "session_id": "abc_123"},
            no_new_info_count=0,
        )
        res = bot.chat("hello there")

        self.assertEqual(bot.last_triage, "SEE_DOCTOR")
        self.assertEqual(res.get("triage"), "SEE_DOCTOR")
        self.assertEqual(bot.xai_logs[-1].get("triage_floor_reason"), "INSUFFICIENT_INFORMATION")
        self.assertTrue(bot.xai_logs[-1].get("low_confidence"))
        self.assertEqual(bot.xai_logs[-1].get("low_confidence_reason"), "INSUFFICIENT_INFORMATION")

    def test_pipeline_floors_when_only_generic_affirmative_provided(self):
        """When intake contains only generic affirmatives like other_details='yes', SELF_CARE is floored to SEE_DOCTOR."""
        bot, prompts = self._setup_bot(
            reasoning_triage="SELF_CARE",
            confidence=85,
            collected={"other_details": "yes", "character": "ok"},
            no_new_info_count=0,
        )
        res = bot.chat("ok")

        self.assertEqual(bot.last_triage, "SEE_DOCTOR")
        self.assertEqual(res.get("triage"), "SEE_DOCTOR")
        self.assertEqual(bot.xai_logs[-1].get("triage_floor_reason"), "INSUFFICIENT_INFORMATION")

    def test_pipeline_floors_when_conversational_false_positive_and_repeated_idk(self):
        """Conversational phrases like 'I am back' with repeated 'idk' are floored to SEE_DOCTOR."""
        bot, prompts = self._setup_bot(
            reasoning_triage="SELF_CARE",
            confidence=85,
            collected={},
            no_new_info_count=2,
        )
        bot.current_complaint_user_messages = ["I am back"]
        res = bot.chat("idk")

        self.assertEqual(bot.last_triage, "SEE_DOCTOR")
        self.assertEqual(res.get("triage"), "SEE_DOCTOR")
        self.assertEqual(bot.xai_logs[-1].get("triage_floor_reason"), "INSUFFICIENT_INFORMATION")

    def test_pipeline_floors_when_user_negates_symptoms_and_repeated_idk(self):
        """Negated symptom phrases like 'I have no pain, no fever' with repeated 'idk' are floored to SEE_DOCTOR."""
        bot, prompts = self._setup_bot(
            reasoning_triage="SELF_CARE",
            confidence=85,
            collected={},
            no_new_info_count=2,
        )
        bot.current_complaint_user_messages = ["I have no pain, no fever"]
        res = bot.chat("idk")

        self.assertEqual(bot.last_triage, "SEE_DOCTOR")
        self.assertEqual(res.get("triage"), "SEE_DOCTOR")
        self.assertEqual(bot.xai_logs[-1].get("triage_floor_reason"), "INSUFFICIENT_INFORMATION")

    def test_pipeline_preserves_self_care_with_multilingual_devanagari_symptom(self):
        """Native Devanagari Hindi symptom satisfies adequacy and preserves SELF_CARE when confidence is high."""
        bot, prompts = self._setup_bot(
            reasoning_triage="SELF_CARE",
            confidence=85,
            collected={"symptom_or_topic": "सिरदर्द"},
            no_new_info_count=0,
        )
        res = bot.chat("मुझे सिरदर्द है")

        self.assertEqual(bot.last_triage, "SELF_CARE")
        self.assertEqual(res.get("triage"), "SELF_CARE")
        self.assertIsNone(bot.xai_logs[-1].get("triage_floor_reason"))

    def test_pipeline_preserves_self_care_with_multilingual_telugu_symptom(self):
        """Native Telugu script symptom satisfies adequacy and preserves SELF_CARE when confidence is high."""
        bot, prompts = self._setup_bot(
            reasoning_triage="SELF_CARE",
            confidence=85,
            collected={"symptom_or_topic": "తలనొప్పి"},
            no_new_info_count=0,
        )
        res = bot.chat("నాకు తలనొప్పిగా ఉంది")

        self.assertEqual(bot.last_triage, "SELF_CARE")
        self.assertEqual(res.get("triage"), "SELF_CARE")
        self.assertIsNone(bot.xai_logs[-1].get("triage_floor_reason"))

    def test_pipeline_cross_complaint_history_isolation(self):
        """An earlier resolved headache conversation cannot validate a new, uninformative complaint."""
        bot, prompts = self._setup_bot(
            reasoning_triage="SELF_CARE",
            confidence=85,
            collected={},
            no_new_info_count=2,
        )
        # Simulate prior resolved complaint in history
        bot.history = [
            {"role": "user", "content": "I have had a severe headache for two days"},
            {"role": "assistant", "content": "Here is advice."},
            {"role": "user", "content": "I am all cured now, feeling better"},
            {"role": "assistant", "content": "Glad to hear that!"},
        ]
        bot.state = ConvState.RESOLVED
        bot.last_top_condition = "Tension Headache"
        bot.collected_info = {}
        bot.current_complaint_user_messages = []

        # User now initiates a fresh complaint with vague/uninformative responses
        res1 = bot.chat("something else is bothering me")
        # Ensure that during assessment of this uninformative second complaint,
        # the old headache does NOT satisfy adequacy:
        self.assertEqual(bot.last_triage, "SEE_DOCTOR")
        self.assertEqual(res1.get("triage"), "SEE_DOCTOR")
        self.assertEqual(bot.xai_logs[-1].get("triage_floor_reason"), "INSUFFICIENT_INFORMATION")

    def test_pipeline_preserves_self_care_when_symptom_info_is_adequate(self):
        """When valid symptom information exists and confidence is high, SELF_CARE is preserved."""
        bot, prompts = self._setup_bot(
            reasoning_triage="SELF_CARE",
            confidence=85,
            collected={"symptom_or_topic": "slight tension headache"},
            no_new_info_count=0,
        )
        res = bot.chat("I have a slight headache")

        self.assertEqual(bot.last_triage, "SELF_CARE")
        self.assertEqual(res.get("triage"), "SELF_CARE")
        self.assertIsNone(bot.xai_logs[-1].get("triage_floor_reason"))
        self.assertFalse(bot.xai_logs[-1].get("low_confidence"))

    def test_pipeline_preserves_emergency_triage_unconditionally(self):
        """When presentation is emergency, triage is never downgraded even if intake was minimal."""
        bot, prompts = self._setup_bot(
            reasoning_triage="EMERGENCY",
            confidence=95,
            collected={},
            no_new_info_count=2,
        )
        # 1. Early emergency red flag check returns type emergency
        res = bot.chat("I cannot breathe")
        self.assertEqual(res.get("type"), "emergency")

        # 2. Assessment pipeline with emergency triage maintains EMERGENCY
        bot._is_emergency = MagicMock(return_value=False)
        res2 = bot.chat("unspecified critical state")
        self.assertEqual(bot.last_triage, "EMERGENCY")
        self.assertEqual(res2.get("triage"), "EMERGENCY")


if __name__ == "__main__":
    unittest.main()
