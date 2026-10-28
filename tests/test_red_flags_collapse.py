"""
tests/test_red_flags_collapse.py — Tests for collapse, unresponsiveness, choking, convulsions,
and related acute emergency red-flag phrases.

Verifies:
- Trigger cases:
  * "my father collapsed and is unresponsive"
  * "he fainted and is not waking up"
  * "she is choking on food"
  * "child is having convulsions"
  * "he stopped breathing"
  * "uncontrolled bleeding won't stop"
  * "usne zehar kha liya"
  * "dam ghut raha hai"
  * "papa gir gaye aur hosh nahi hai"
  * "fits vachayi"
  * Additional new phrases: "won't wake up", "wont wake up", "gasping for air", "telivi ledu", "fits vastunnayi", "होश नहीं"
- Non-trigger cases:
  * "my fever is not responding to paracetamol"
  * "cough not responding to syrup"
  * "I fainted briefly yesterday after giving blood"
  * "this fits my description"
  * "he is not unresponsive"
  * "he is responsive and talking"
  * "I do not have convulsions"
"""

import unittest
from unittest.mock import MagicMock
import sys

# Mock scipy if not present in environment
for _mod in ("scipy", "scipy.sparse"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from core.red_flags import check_red_flags


class TestRedFlagsCollapse(unittest.TestCase):
    # ── 1. Positive Emergency Triggers ──────────────────────────

    def test_father_collapsed_and_unresponsive(self):
        res = check_red_flags("my father collapsed and is unresponsive")
        self.assertTrue(res["emergency"])
        self.assertIn("consciousness_seizure", res["rule_ids"])

    def test_fainted_and_not_waking_up(self):
        res = check_red_flags("he fainted and is not waking up")
        self.assertTrue(res["emergency"])
        self.assertIn("consciousness_seizure", res["rule_ids"])

    def test_choking_on_food(self):
        res = check_red_flags("she is choking on food")
        self.assertTrue(res["emergency"])
        self.assertIn("breathing", res["rule_ids"])

    def test_child_having_convulsions(self):
        res = check_red_flags("child is having convulsions")
        self.assertTrue(res["emergency"])
        self.assertIn("consciousness_seizure", res["rule_ids"])

    def test_stopped_breathing(self):
        res = check_red_flags("he stopped breathing")
        self.assertTrue(res["emergency"])
        self.assertIn("breathing", res["rule_ids"])

    def test_uncontrolled_bleeding_wont_stop(self):
        res = check_red_flags("uncontrolled bleeding won't stop")
        self.assertTrue(res["emergency"])
        self.assertIn("bleeding_poisoning_anaphylaxis", res["rule_ids"])

    def test_poisoning_usne_zehar_kha_liya(self):
        res = check_red_flags("usne zehar kha liya")
        self.assertTrue(res["emergency"])
        self.assertIn("bleeding_poisoning_anaphylaxis", res["rule_ids"])

    def test_breathing_dam_ghut_raha_hai(self):
        res = check_red_flags("dam ghut raha hai")
        self.assertTrue(res["emergency"])
        self.assertIn("breathing", res["rule_ids"])

    def test_consciousness_papa_gir_gaye_aur_hosh_nahi_hai(self):
        res = check_red_flags("papa gir gaye aur hosh nahi hai")
        self.assertTrue(res["emergency"])
        self.assertIn("consciousness_seizure", res["rule_ids"])

    def test_seizure_fits_vachayi(self):
        res = check_red_flags("fits vachayi")
        self.assertTrue(res["emergency"])
        self.assertIn("consciousness_seizure", res["rule_ids"])

    def test_additional_new_phrases(self):
        # won't wake up / wont wake up
        res1 = check_red_flags("she won't wake up")
        self.assertTrue(res1["emergency"])
        res2 = check_red_flags("patient wont wake up")
        self.assertTrue(res2["emergency"])

        # gasping for air
        res3 = check_red_flags("patient is gasping for air")
        self.assertTrue(res3["emergency"])
        self.assertIn("breathing", res3["rule_ids"])

        # bleeding wont stop
        res4 = check_red_flags("wound is bleeding wont stop")
        self.assertTrue(res4["emergency"])

        # telivi ledu & fits vastunnayi
        res5 = check_red_flags("ataniki telivi ledu")
        self.assertTrue(res5["emergency"])
        res6 = check_red_flags("fits vastunnayi")
        self.assertTrue(res6["emergency"])

        # Devanagari hosh nahi
        res7 = check_red_flags("मरीज को होश नहीं है")
        self.assertTrue(res7["emergency"])
        self.assertIn("consciousness_seizure", res7["rule_ids"])

    # ── 2. Negative Guards (Must NOT Trigger) ───────────────────

    def test_fever_not_responding_to_paracetamol(self):
        res = check_red_flags("my fever is not responding to paracetamol")
        self.assertFalse(res["emergency"])

    def test_cough_not_responding_to_syrup(self):
        res = check_red_flags("cough not responding to syrup")
        self.assertFalse(res["emergency"])

    def test_fainted_briefly_yesterday(self):
        res = check_red_flags("I fainted briefly yesterday after giving blood")
        self.assertFalse(res["emergency"])

    def test_fits_my_description(self):
        res = check_red_flags("this fits my description")
        self.assertFalse(res["emergency"])

    def test_not_unresponsive(self):
        res = check_red_flags("he is not unresponsive")
        self.assertFalse(res["emergency"])

    def test_responsive_and_talking(self):
        res = check_red_flags("he is responsive and talking")
        self.assertFalse(res["emergency"])

    def test_do_not_have_convulsions(self):
        res = check_red_flags("I do not have convulsions")
        self.assertFalse(res["emergency"])


if __name__ == "__main__":
    unittest.main()

