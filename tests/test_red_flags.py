# ==============================================================================
#  tests/test_red_flags.py — Unit tests for Emergency Pre-check
#
#  Offline, deterministic test suite covering:
#    - Controls & bug regression cases from negation under-triage issue
#    - English emergency red flags
#    - Hindi native (Devanagari) emergency red flags
#    - Hindi Latin-script emergency red flags
#    - Telugu native emergency red flags
#    - Telugu Latin-script emergency red flags
#    - Negated emergency phrases (English, Hindi, Telugu)
#    - Substring false positive rejection (e.g. keystroke, heart)
#    - Benign non-emergency messages
#    - Mixed negation & clause boundaries where true emergencies must fire
#    - Intrinsic negators in emergency phrases
#    - Contract validation & edge cases
# ==============================================================================

import importlib.util
import os
import sys
import unittest

try:
    from core.red_flags import check_red_flags, normalize_text, split_clauses, RULES
except ModuleNotFoundError:
    # Allows offline tests to run cleanly under environments without heavy dependencies
    core_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "core", "red_flags.py"))
    spec = importlib.util.spec_from_file_location("core.red_flags", core_path)
    red_flags_mod = importlib.util.module_from_spec(spec)
    sys.modules["core.red_flags"] = red_flags_mod
    spec.loader.exec_module(red_flags_mod)
    check_red_flags = red_flags_mod.check_red_flags
    normalize_text = red_flags_mod.normalize_text
    split_clauses = red_flags_mod.split_clauses
    RULES = red_flags_mod.RULES


class TestClauseSplittingAndNormalization(unittest.TestCase):
    def test_clause_splitting(self):
        clauses = split_clauses("I have chest pain. No fever.")
        self.assertEqual(clauses, ["I have chest pain", "No fever"])

        clauses_comma = split_clauses("chest pain, no cough")
        self.assertEqual(clauses_comma, ["chest pain", "no cough"])

        clauses_mixed = split_clauses("Severe chest pain! Is it a heart attack? No idea;\nplease help")
        self.assertEqual(clauses_mixed, ["Severe chest pain", "Is it a heart attack", "No idea", "please help"])

    def test_nfkc_and_lowercase(self):
        self.assertEqual(normalize_text("CHEST PAIN"), "chest pain")
        self.assertEqual(normalize_text("सीने में दर्द"), "सीने में दर्द")

    def test_contractions_and_punctuation(self):
        self.assertEqual(normalize_text("Can't breathe!"), "cant breathe")
        self.assertEqual(normalize_text("I don't have chest pain."), "i dont have chest pain")
        self.assertEqual(normalize_text("self-harm"), "self harm")

    def test_indic_combining_marks_preserved(self):
        norm_hi = normalize_text("सीने में दर्द नहीं है।")
        self.assertEqual(norm_hi, "सीने में दर्द नहीं है")
        norm_te = normalize_text("ఛాతీ నొప్పి లేదు!")
        self.assertEqual(norm_te, "ఛాతీ నొప్పి లేదు")

    def test_whitespace_collapsing(self):
        self.assertEqual(normalize_text("  chest   pain  "), "chest pain")


class TestUnderTriageRegression(unittest.TestCase):
    """Specific regression suite for under-triage bugs and controls."""

    def test_controls_must_be_emergency(self):
        controls = [
            ("chest pain", "cardiac"),
            ("I have chest pain", "cardiac"),
            ("I can't breathe", "breathing"),
            ("मुझे सीने में दर्द है", "cardiac"),
            ("seene mein dard hai", "cardiac"),
            ("నాకు ఛాతీ నొప్పి ఉంది", "cardiac"),
        ]
        for text, exp_cat in controls:
            res = check_red_flags(text)
            self.assertTrue(res["emergency"], f"Control failed: {text}")
            self.assertEqual(res["category"], exp_cat)

    def test_bug_cases_must_be_emergency(self):
        bug_cases = [
            ("I have chest pain. No fever.", "cardiac"),
            ("chest pain, no cough", "cardiac"),
            ("no fever. chest pain", "cardiac"),
            ("I have chest pain and no fever", "cardiac"),
            ("my chest pain is not gone", "cardiac"),
            ("मुझे सीने में दर्द है, बुखार नहीं है", "cardiac"),
            ("seene mein dard hai bukhar nahi hai", "cardiac"),
            ("నాకు ఛాతీ నొప్పి ఉంది జ్వరం లేదు", "cardiac"),
        ]
        for text, exp_cat in bug_cases:
            res = check_red_flags(text)
            self.assertTrue(res["emergency"], f"Bug case under-triaged: {text} -> {res}")
            self.assertEqual(res["category"], exp_cat)

    def test_must_be_false_cases(self):
        false_cases = [
            "I do not have chest pain",
            "chest pain nahi hai",
            "I have no chest pain",
            "seene mein dard nahi hai",
            "chaati noppi ledu",
            "सीने में दर्द नहीं है",
            "ఛాతీ నొప్పి లేదు",
        ]
        for text in false_cases:
            res = check_red_flags(text)
            self.assertFalse(res["emergency"], f"Expected False, but got True: {text} -> {res}")
            self.assertIsNone(res["category"])
            self.assertEqual(res["rule_ids"], [])

    def test_extra_present_plus_negated_symptom(self):
        """10 extra cases: present emergency symptom + different negated symptom in en/hi/hi_latn/te/te_latn."""
        cases = [
            # English
            ("I have difficulty breathing but no fever", "breathing"),
            ("chest tightness, no vomiting", "cardiac"),
            # Hindi native
            ("सांस लेने में तकलीफ है लेकिन जुकाम नहीं है", "breathing"),
            ("सीने में दर्द है उल्टी नहीं है", "cardiac"),
            # Hindi Latin
            ("saans lene mein takleef hai bukhar nahi hai", "breathing"),
            ("chehra tedha lag raha hai sardard nahi hai", "stroke"),
            # Telugu native
            ("శ్వాస తీసుకోవడం కష్టం గా ఉంది కానీ దగ్గు లేదు", "breathing"),
            ("ఛాతీ నొప్పి ఉంది వాంతులు లేవు", "cardiac"),
            # Telugu Latin
            ("oopiri aadatledu jwaram ledu", "breathing"),
            ("gunde noppi undi daggu ledu", "cardiac"),
        ]
        for text, exp_cat in cases:
            res = check_red_flags(text)
            self.assertTrue(res["emergency"], f"Failed on present+negated: {text} -> {res}")
            self.assertEqual(res["category"], exp_cat)

    def test_extra_negated_must_be_false(self):
        """Extra negated cases that must evaluate to emergency=False."""
        cases = [
            "patient has no chest tightness",
            "denies difficulty breathing",
            "सांस लेने में तकलीफ नहीं है",
            "chehra tedha nahi hai",
            "శ్వాస తీసుకోవడం కష్టం లేదు",
            "gunde noppi ledu",
            "without chest pain or fever",
            "never had chest pain",
        ]
        for text in cases:
            res = check_red_flags(text)
            self.assertFalse(res["emergency"], f"Expected False, got True: {text} -> {res}")


class TestEnglishRedFlags(unittest.TestCase):
    def test_cardiac_en(self):
        cases = [
            "I have severe chest pain since this morning",
            "Sudden onset of chest tightness",
            "My father is having a heart attack right now",
        ]
        for text in cases:
            res = check_red_flags(text)
            self.assertTrue(res["emergency"], f"Failed to detect: {text}")
            self.assertEqual(res["category"], "cardiac")
            self.assertIn("cardiac", res["rule_ids"])

    def test_breathing_en(self):
        cases = [
            "Please help me, I can't breathe",
            "Patient cannot breathe properly",
            "Sudden difficulty breathing after exertion",
            "The patient is not breathing at all",
        ]
        for text in cases:
            res = check_red_flags(text)
            self.assertTrue(res["emergency"], f"Failed to detect: {text}")
            self.assertEqual(res["category"], "breathing")
            self.assertIn("breathing", res["rule_ids"])

    def test_stroke_en(self):
        cases = [
            "I noticed face drooping on his right side",
            "Sudden slurred speech and confusion",
            "Experiencing weakness on one side of the body",
            "I think my mother is having a stroke",
        ]
        for text in cases:
            res = check_red_flags(text)
            self.assertTrue(res["emergency"], f"Failed to detect: {text}")
            self.assertEqual(res["category"], "stroke")
            self.assertIn("stroke", res["rule_ids"])

    def test_consciousness_seizure_en(self):
        cases = [
            "The patient had an unexpected seizure",
            "He was found unconscious on the floor",
            "I suddenly passed out while standing",
        ]
        for text in cases:
            res = check_red_flags(text)
            self.assertTrue(res["emergency"], f"Failed to detect: {text}")
            self.assertEqual(res["category"], "consciousness_seizure")
            self.assertIn("consciousness_seizure", res["rule_ids"])

    def test_bleeding_poisoning_anaphylaxis_en(self):
        cases = [
            "There is severe bleeding from the wound",
            "Suspected accidental overdose of medication",
            "I suspect carbon monoxide poisoning",
            "Experiencing sudden throat swelling after eating nuts",
            "Signs of severe allergic shock and anaphylaxis",
        ]
        for text in cases:
            res = check_red_flags(text)
            self.assertTrue(res["emergency"], f"Failed to detect: {text}")
            self.assertEqual(res["category"], "bleeding_poisoning_anaphylaxis")
            self.assertIn("bleeding_poisoning_anaphylaxis", res["rule_ids"])

    def test_self_harm_en(self):
        cases = [
            "I am feeling suicidal today",
            "I feel so hopeless and want to die",
            "Thoughts that I might kill myself",
            "Struggling with intense urges of self-harm",
        ]
        for text in cases:
            res = check_red_flags(text)
            self.assertTrue(res["emergency"], f"Failed to detect: {text}")
            self.assertEqual(res["category"], "self_harm")
            self.assertIn("self_harm", res["rule_ids"])


class TestHindiNativeRedFlags(unittest.TestCase):
    def test_cardiac_hi(self):
        res = check_red_flags("मुझे सीने में दर्द हो रहा है")
        self.assertTrue(res["emergency"])
        self.assertEqual(res["category"], "cardiac")

    def test_breathing_hi(self):
        for text in ["सांस नहीं आ रही है बहुत घबराहट है", "सांस लेने में तकलीफ हो रही है"]:
            res = check_red_flags(text)
            self.assertTrue(res["emergency"], f"Failed for {text}")
            self.assertEqual(res["category"], "breathing")

    def test_consciousness_hi(self):
        for text in ["मरीज अचानक बेहोश हो गया", "उसे अचानक दौरा पड़ा"]:
            res = check_red_flags(text)
            self.assertTrue(res["emergency"], f"Failed for {text}")
            self.assertEqual(res["category"], "consciousness_seizure")

    def test_self_harm_hi(self):
        for text in ["वह आत्महत्या करने की बात कर रहा है", "मैं अब मरना चाहता हूँ"]:
            res = check_red_flags(text)
            self.assertTrue(res["emergency"], f"Failed for {text}")
            self.assertEqual(res["category"], "self_harm")


class TestHindiLatinRedFlags(unittest.TestCase):
    def test_hindi_latin(self):
        cases = [
            ("bahut tej seene mein dard ho raha hai", "cardiac"),
            ("mujhe saans nahi aa rahi jaldi batao", "breathing"),
            ("saans lene mein takleef hai", "breathing"),
            ("achanak chehra tedha lag raha hai", "stroke"),
            ("voh achanak behosh ho gaya", "consciousness_seizure"),
            ("usse kal raat daura aaya tha", "consciousness_seizure"),
            ("atmahatya ke vichaar aa rahe hain", "self_harm"),
            ("main ab marna chahta hoon bas", "self_harm"),
        ]
        for text, exp_cat in cases:
            res = check_red_flags(text)
            self.assertTrue(res["emergency"], f"Failed for: {text}")
            self.assertEqual(res["category"], exp_cat)


class TestTeluguNativeRedFlags(unittest.TestCase):
    def test_cardiac_te(self):
        res = check_red_flags("నాకు ఛాతీ నొప్పి వస్తోంది చాలా ఎక్కువగా ఉంది")
        self.assertTrue(res["emergency"])
        self.assertEqual(res["category"], "cardiac")

    def test_breathing_te(self):
        for text in ["నాకు ఊపిరి ఆడటం లేదు కాపాడండి", "శ్వాస తీసుకోవడం కష్టం గా ఉంది"]:
            res = check_red_flags(text)
            self.assertTrue(res["emergency"], f"Failed for {text}")
            self.assertEqual(res["category"], "breathing")

    def test_consciousness_te(self):
        res = check_red_flags("అతను స్పృహ తప్పి పడిపోయాడు")
        self.assertTrue(res["emergency"])
        self.assertEqual(res["category"], "consciousness_seizure")

    def test_self_harm_te(self):
        for text in ["ఆమె ఆత్మహత్య చేసుకోవాలని చూస్తోంది", "నాకు చనిపోవాలని అనిపిస్తుంది"]:
            res = check_red_flags(text)
            self.assertTrue(res["emergency"], f"Failed for {text}")
            self.assertEqual(res["category"], "self_harm")


class TestTeluguLatinRedFlags(unittest.TestCase):
    def test_telugu_latin(self):
        cases = [
            ("naku chaati noppi ga undi doctor", "cardiac"),
            ("gunde noppi modalaindi", "cardiac"),
            ("naku oopiri aadatledu ventane randi", "breathing"),
            ("spruha tappindi achanak ga", "consciousness_seizure"),
            ("atmahatya chesukovalani undi", "self_harm"),
            ("naku chanipovalani undi", "self_harm"),
        ]
        for text, exp_cat in cases:
            res = check_red_flags(text)
            self.assertTrue(res["emergency"], f"Failed for: {text}")
            self.assertEqual(res["category"], exp_cat)


class TestSubstringFalsePositives(unittest.TestCase):
    def test_keystroke_and_heart(self):
        cases = [
            "Counting every keystroke on the mechanical keyboard",
            "My heart is full of happiness and gratitude",
            "I have severe heartburn after eating spicy pizza",
            "She is my sweetheart",
            "Applying fresh paint on the bedroom wall",
            "Looking outward towards the ocean",
            "Experiencing a mild headache and fatigue",
        ]
        for text in cases:
            res = check_red_flags(text)
            self.assertFalse(res["emergency"], f"False positive triggered for: {text} -> {res}")


class TestBenignMessages(unittest.TestCase):
    def test_benign_conversations(self):
        cases = [
            "Hello, good morning doctor",
            "What are the best home remedies for a common cold?",
            "I twisted my left ankle while jogging in the park yesterday",
            "Can high blood sugar cause frequent urination?",
            "Feeling slightly tired after a long day at work",
            "My throat feels a bit itchy today",
            "What should I eat to improve digestion?",
        ]
        for text in cases:
            res = check_red_flags(text)
            self.assertFalse(res["emergency"], f"Benign message flagged: {text} -> {res}")


class TestMixedNegation(unittest.TestCase):
    def test_mixed_negation_must_fire(self):
        cases = [
            ("I have no fever but cannot breathe", "breathing"),
            ("No cough but experiencing severe chest pain", "cardiac"),
            ("No headache, however weakness on one side appeared", "stroke"),
            ("I have chest pain but no fever", "cardiac"),
            ("fever nahi hai lekin saans lene mein takleef hai", "breathing"),
            ("jwaram ledu kani chaati noppi undi", "cardiac"),
        ]
        for text, exp_cat in cases:
            res = check_red_flags(text)
            self.assertTrue(res["emergency"], f"Mixed negation must fire: {text}")
            self.assertEqual(res["category"], exp_cat)


class TestPhrasesWithInternalNegators(unittest.TestCase):
    def test_intrinsic_negators_must_fire(self):
        cases = [
            ("saans nahi aa rahi", "breathing"),
            ("oopiri aadatledu", "breathing"),
            ("the patient is not breathing", "breathing"),
            ("I can't breathe right now", "breathing"),
            ("He cannot breathe at all", "breathing"),
        ]
        for text, exp_cat in cases:
            res = check_red_flags(text)
            self.assertTrue(res["emergency"], f"Intrinsic negator phrase must fire: {text}")
            self.assertEqual(res["category"], exp_cat)


class TestMultiRuleAndContract(unittest.TestCase):
    def test_contract_keys(self):
        res = check_red_flags("chest pain")
        self.assertIn("emergency", res)
        self.assertIn("category", res)
        self.assertIn("rule_ids", res)
        self.assertIsInstance(res["emergency"], bool)
        self.assertIsInstance(res["rule_ids"], list)

    def test_multiple_red_flags(self):
        res = check_red_flags("I have chest pain and cannot breathe")
        self.assertTrue(res["emergency"])
        self.assertIn("cardiac", res["rule_ids"])
        self.assertIn("breathing", res["rule_ids"])
        self.assertEqual(res["category"], "cardiac")

    def test_partial_negation_multi_flag(self):
        res = check_red_flags("No chest pain, but I cannot breathe")
        self.assertTrue(res["emergency"])
        self.assertEqual(res["category"], "breathing")
        self.assertEqual(res["rule_ids"], ["breathing"])

    def test_empty_and_invalid_inputs(self):
        for val in ["", "   ", None, 12345, [], {}]:
            res = check_red_flags(val)
            self.assertFalse(res["emergency"])
            self.assertIsNone(res["category"])
            self.assertEqual(res["rule_ids"], [])

    def test_negation_distance_window(self):
        self.assertFalse(check_red_flags("patient denies any chest pain")["emergency"])
        self.assertFalse(check_red_flags("patient denies ever having chest pain")["emergency"])
        self.assertFalse(check_red_flags("patient denies having had severe chest pain")["emergency"])
        res_distant = check_red_flags("patient denies having had any previous severe chest pain")
        self.assertTrue(res_distant["emergency"])


if __name__ == "__main__":
    unittest.main()
