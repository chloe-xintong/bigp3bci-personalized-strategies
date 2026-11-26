import sys
import unittest
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from export_unmatched_blink_context import context_rows, covered


class UnmatchedContextTests(unittest.TestCase):
    def test_previous_match_skips_unmatched_and_keeps_manual_pairs(self):
        candidates = pd.DataFrame([
            dict(blink_candidate_id=f'B{i}', onset_seconds=a, offset_seconds=b,
                 peak_seconds=(a+b)/2, filter_startup=False)
            for i, a, b in [(1, 0, 1), (2, 2, 3), (3, 4, 5), (4, 6, 7)]])
        manual = pd.DataFrame([dict(manual_id='M1', start_sec=2.1, end_sec=2.3),
                               dict(manual_id='M2', start_sec=2.4, end_sec=2.6)])
        rows = context_rows(candidates, manual, [(0, 5)], {})
        self.assertEqual(rows[0]['previous_matched_candidate_id'], '')
        self.assertEqual([r['blink_candidate_id'] for r in rows], ['B1', 'B3', 'B3', 'B4', 'B4'])
        self.assertEqual({r['previous_matched_candidate_id'] for r in rows[1:]}, {'B2'})
        self.assertEqual([r['previous_match_manual_id'] for r in rows[1:3]], ['M1', 'M2'])
        self.assertEqual(rows[1]['gap_from_previous_candidate_end_seconds'], 1)
        self.assertFalse(rows[-1]['fully_within_complete_review'])

    def test_touching_boundaries_do_not_match(self):
        candidates = pd.DataFrame([dict(blink_candidate_id='B1', onset_seconds=1,
            offset_seconds=2, peak_seconds=1.5, filter_startup=True)])
        manual = pd.DataFrame([dict(manual_id='M1', start_sec=0, end_sec=1)])
        rows = context_rows(candidates, manual, [(0, 3)], {})
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]['filter_startup'])
        self.assertTrue(rows[0]['fully_within_complete_review'])

    def test_review_union_requires_no_gap(self):
        self.assertTrue(covered(1, 3, [(0, 2), (2, 4)]))
        self.assertFalse(covered(1, 3, [(0, 2), (2.1, 4)]))


if __name__ == '__main__':
    unittest.main()
