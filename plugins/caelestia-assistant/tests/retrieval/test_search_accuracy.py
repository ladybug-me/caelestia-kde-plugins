"""Golden-query accuracy: each query must surface its expected doc in top-3.

The goldens are phrased the way a user would ask, not the way the corpus is
written; if one fails, fix the corpus header (tags/synonyms) or tokenizer —
never weaken the test.
"""

from __future__ import annotations

import unittest
from typing import List, Set, Tuple

from assistant.capabilities.retrieval.search import Searcher

# (query, ids that are acceptable anywhere in the top-3)
GOLDEN_QUERIES: List[Tuple[str, Set[str]]] = [
    ("vesktop freezes when I screenshare", {"ISS-402", "DOC-TS-03"}),
    ("colors revert after reboot", {"ISS-763", "DOC-TS-03"}),
    ("install fails cmake missing", {"DOC-TS-01"}),
    ("workspace pills not loading", {"DOC-TS-03", "ISS-416"}),
    ("how do I free disk space used by the shell", {f"DOC-TS-{n:02d}" for n in range(1, 12)}),
    ("what is the KWin bridge", {"DOC-ARCH-KWIN", "DOC-TS-07"}),
    ("bluetooth turns on then turns off and nothing pairs", {"ISS-006"}),
    ("quickshell has crashed and window switcher shows no windows", {"ISS-418", "DOC-TS-03"}),
    ("error: Unrecognized pragma DefaultEnv QS_NO_RELOAD_POPUP", {"ISS-333"}),
    ("quickshell gives an error after updating caelestia", {"ISS-528", "DOC-TS-03"}),
    ("game with gamescope launches in half the screen", {"ISS-641", "DOC-TS-07"}),
    ("gpu usage widget shows half of the real usage", {"ISS-616", "DOC-TS-07"}),
    ("live video wallpaper does not appear", {"ISS-409", "DOC-TS-03"}),
    ("colour scheme keeps switching to material you dark", {"ISS-014", "DOC-TS-03"}),
    ("random chromium apps crash spotify discord steam", {"ISS-461"}),
    ("lock screen shows the breeze greeter instead of caelestia", {"DOC-TS-04", "DOC-ARCH-LOCKSCREEN"}),
    ("missing qml module metadata after build", {"DOC-TS-01"}),
    ("how do I update the shell", {"DOC-README", "DOC-TS-10"}),
    ("screen flashes and stutters every second", {"DOC-TS-03", "ISS-014"}),
    ("launcher opens on the wrong monitor when fullscreen video on the other one", {"ISS-416", "DOC-TS-03"}),
    ("what can the assistant do", {"ISS-120", "ISS-802"}),
    # additions for issues closed after #763, including the post-packaging #805
    ("lock screen falls back to the built-in one after update", {"ISS-805"}),
    ("caelestia services module not found kscreenlocker", {"ISS-805", "DOC-ARCH-LOCKSCREEN"}),
    ("colors look less vibrant after the color engine update", {"ISS-790"}),
    ("matugen vs materialyoucolor palette", {"ISS-790"}),
    ("brightness slider shows a moon icon instead of a sun", {"ISS-744"}),
    ("local bin not in path for bash after install", {"ISS-765"}),
    ("different paths for install.sh and package installs", {"ISS-773"}),
    ("notifications contain a lot of numbers and escape sequences", {"ISS-774"}),
    ("ambient glow colors from window previews", {"ISS-738"}),
    ("how do I adjust color intensity", {"ISS-791", "ISS-790"}),
    ("shell crashes without much information", {"ISS-792"}),
    # additions for issues closed after #805, including the refreshed #120 direction
    ("no notification badges on dock icons", {"ISS-788"}),
    ("has the ai assistant work fallen to the backlog", {"ISS-120"}),
]


class TestGoldenQueryAccuracy(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.searcher = Searcher.from_file()

    def test_golden_query_count(self) -> None:
        self.assertGreaterEqual(len(GOLDEN_QUERIES), 10)

    def test_expected_doc_in_top3(self) -> None:
        failures: List[str] = []
        for query, acceptable in GOLDEN_QUERIES:
            top3 = [hit["doc_id"] for hit in self.searcher.search(query, k=3)]
            if not set(top3) & acceptable:
                failures.append(f"{query!r} -> top3 {top3}, expected one of {sorted(acceptable)}")
        self.assertEqual(failures, [], "\n".join(failures))

    def test_results_shape_and_limit(self) -> None:
        results = self.searcher.search("vesktop screenshare freeze", k=5)
        self.assertLessEqual(len(results), 5)
        for hit in results:
            self.assertEqual(set(hit), {"doc_id", "title", "score", "source", "snippet"})
        # deterministic: same query, same order
        again = self.searcher.search("vesktop screenshare freeze", k=5)
        self.assertEqual([h["doc_id"] for h in results], [h["doc_id"] for h in again])


if __name__ == "__main__":
    unittest.main()
