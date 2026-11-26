from pathlib import Path
import sys
import unittest
from collections import Counter
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from bigp3bci_history_amount import enumerate_subsets,summarize_history


class HistoryAmountTests(unittest.TestCase):
    def test_complete_enumeration(self):
        subsets=enumerate_subsets()
        self.assertEqual(len(subsets),64)
        self.assertEqual([Counter(h for h,_ in subsets)[h] for h in range(7)],[1,6,15,20,15,6,1])
        self.assertEqual(len({slots for _,slots in subsets}),64)

    def synthetic(self):
        rows=[]
        for s in range(10):
            for h,slots in enumerate_subsets():
                n1=sum(i<3 for i in slots);n2=h-n1
                for k in [0,3,6,12]:
                    available=bool(h or k)
                    rows.append(dict(subject_id=f'S{s}',H=h,k=k,n_s1=n1,n_s2=n2,model_available=available,
                        flash_roc_auc=.5+.01*s+.003*h+.001*k+.001*(n2-n1) if available else np.nan,
                        character_accuracy=.5 if available else np.nan))
        return pd.DataFrame(rows)

    def test_paired_differences_and_equal_session_file_counts(self):
        raw=self.synthetic();subjects=sorted(raw.subject_id.unique())
        _,macro,_,sessions=summarize_history(raw,subjects,[0,3,6,12],replicates=100,seed=1)
        for r in macro[macro.k.gt(0)].itertuples():
            self.assertAlmostEqual(r.delta_vs_current,.003*r.H)
            self.assertAlmostEqual(r.delta_ci_low,r.delta_ci_high)
        self.assertTrue(macro[macro.k.eq(0)].delta_vs_current.isna().all())
        for r in sessions.itertuples():self.assertAlmostEqual(r.delta_s2_minus_s1,.002*r.H)

    def test_missing_subset_is_not_silently_omitted(self):
        raw=self.synthetic()
        raw=raw.drop(raw[raw.H.eq(1)].index[0])
        with self.assertRaises(ValueError):summarize_history(raw,sorted(raw.subject_id.unique()),[0,3,6,12],replicates=10)


if __name__=='__main__':unittest.main()
