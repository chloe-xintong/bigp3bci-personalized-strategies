import sys
import unittest
from pathlib import Path
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from build_double_blind_review import allocate
from score_double_blind_review import agreement


class DoubleBlindReviewTests(unittest.TestCase):
    def test_allocation_respects_population_and_total(self):
        counts=pd.Series({'a':3,'b':8,'c':20})
        n=allocate(counts,13)
        self.assertEqual(int(n.sum()),13)
        self.assertTrue((n<=counts).all())

    def test_kappa(self):
        result,_=agreement(['true_blink','other_artifact'],['true_blink','other_artifact'])
        self.assertEqual(result['cohens_kappa'],1)
        result,_=agreement(['true_blink','other_artifact'],['other_artifact','true_blink'])
        self.assertEqual(result['cohens_kappa'],-1)

    def test_kappa_degenerate_is_unavailable(self):
        result,_=agreement(['uncertain'],['uncertain'])
        self.assertIsNone(result['cohens_kappa'])


if __name__=='__main__':unittest.main()
