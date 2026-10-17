"""tests for personal.cloze — drafts-only cloze flashcard drafting.

Pinned: the candidate selector (dates, standalone numbers, capitalized
noun phrases; sentence-initial capitals EXCLUDED — a capitalized
sentence start is punctuation, not a proper noun), math-line refusal,
the mandatory review gate (approve/reject is one-way, re-deciding is
refused), promotion only from approved drafts (no bypass), and the
scheduler contract — approve() creates the card through srs.new_card(),
the FSRS entry point, and nothing schedules itself.
"""
import unittest

from assistant.capabilities.brain.personal import cloze


NOTE = ("Albert Einstein published the photoelectric paper in 1905. "
        "The Mitochondria produces ATP for the cell machinery needs. "
        "On Tuesday I met Dana for coffee near the old park.")


class SelectorTests(unittest.TestCase):
    def test_sentence_initial_names_excluded(self):
        result = cloze.draft_cards(NOTE, "n.md")
        answers = [d["answer"] for d in result["drafts"]]
        self.assertNotIn("Albert Einstein", answers)

    def test_capitalized_phrase_mid_sentence_caught(self):
        result = cloze.draft_cards(NOTE, "n.md")
        answers = [d["answer"] for d in result["drafts"]]
        self.assertIn("Mitochondria", answers)

    def test_numbers_blanked(self):
        result = cloze.draft_cards(NOTE, "n.md")
        answers = [d["answer"] for d in result["drafts"]]
        self.assertIn("1905", answers)

    def test_questions_use_cloze_marker(self):
        result = cloze.draft_cards(NOTE, "n.md")
        for d in result["drafts"]:
            self.assertIn("[...]", d["question"])
            self.assertNotIn("[...]", d["answer"])

    def test_math_lines_skipped_and_counted(self):
        text = ("The equation y = x^2 + c holds here for every x. "
                "The Amazon basin drains a vast continental area.")
        result = cloze.draft_cards(text, "n.md")
        self.assertEqual(result["skipped_math"], 1)
        self.assertTrue(result["drafts"])

    def test_max_cards_cap(self):
        long_text = ("The Atlantic Ocean borders many coastal nations. "
                     "The Pacific Ocean is the deepest ocean basin known. "
                     "The Indian Ocean washes the eastern African coast. "
                     "The Arctic Ocean freezes over every single winter.")
        result = cloze.draft_cards(long_text, "n.md", max_cards=2)
        self.assertEqual(len(result["drafts"]), 2)

    def test_honesty_note_carried(self):
        result = cloze.draft_cards(NOTE, "n.md")
        self.assertIn("review is mandatory", result["note"])


class ReviewGateTests(unittest.TestCase):
    def _state_with_draft(self):
        state = {}
        d = cloze.draft_cards(NOTE, "n.md")
        self.assertTrue(d["drafts"])
        state.setdefault(cloze.STATE_KEY, {})["d0"] = dict(d["drafts"][0],
                                                           status="draft")
        return state

    def test_approve_creates_real_fsrs_card(self):
        from assistant.capabilities.brain.personal import srs
        state = self._state_with_draft()
        draft = cloze.approve(state, "d0")
        self.assertEqual(draft["status"], "approved")
        self.assertEqual(draft["card"], srs.new_card())

    def test_reject_records_and_hides(self):
        state = self._state_with_draft()
        cloze.approve(state, "d0")
        state[cloze.STATE_KEY]["d1"] = dict(state[cloze.STATE_KEY]["d0"],
                                            status="draft", question="q",
                                            answer="a")
        cloze.reject(state, "d1", reason="trivial")
        self.assertEqual(cloze.open_drafts(state), [])
        self.assertEqual(state[cloze.STATE_KEY]["d1"]["reason"], "trivial")

    def test_redeciding_refused(self):
        state = self._state_with_draft()
        cloze.approve(state, "d0")
        with self.assertRaises(ValueError):
            cloze.reject(state, "d0")
        with self.assertRaises(ValueError):
            cloze.approve(state, "d0")

    def test_promotion_requires_approval(self):
        state = self._state_with_draft()
        with self.assertRaises(ValueError):
            cloze.promote(state, "d0")
        cloze.approve(state, "d0")
        card = cloze.promote(state, "d0")
        self.assertEqual(card["answer"], state[cloze.STATE_KEY]["d0"]["answer"])

    def test_unknown_draft_refused(self):
        with self.assertRaises(KeyError):
            cloze.approve({}, "ghost")


if __name__ == "__main__":
    unittest.main()
