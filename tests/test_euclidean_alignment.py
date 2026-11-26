from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from bigp3bci_euclidean_alignment import (trial_covariances,reference_covariance,inverse_sqrt,align_epochs,
                                          align_features,bootstrap_deltas,direction_rule)


def window_means(epochs,n_windows):
    edges=np.linspace(0,epochs.shape[2],n_windows+1).astype(int)
    return np.stack([epochs[:,:,a:b].mean(axis=2) for a,b in zip(edges[:-1],edges[1:])],axis=2).reshape(len(epochs),-1)


class EuclideanAlignmentTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(0);mixing=rng.normal(size=(16,16))
        self.epochs=np.einsum('cd,ndt->nct',mixing,rng.normal(size=(200,16,64)))

    def test_aligned_reference_is_identity(self):
        w=inverse_sqrt(reference_covariance(self.epochs,1e-9))
        mean=trial_covariances(align_epochs(self.epochs,w)).mean(axis=0)
        np.testing.assert_allclose(mean,np.eye(16),atol=1e-6)

    def test_feature_space_alignment_equals_epoch_space(self):
        w=inverse_sqrt(reference_covariance(self.epochs,1e-3))
        direct=window_means(align_epochs(self.epochs,w),16)
        np.testing.assert_allclose(align_features(window_means(self.epochs,16),w),direct,atol=1e-10)

    def test_shrinkage_makes_rank_deficient_reference_invertible(self):
        flat=self.epochs.copy();flat[:,0]=flat[:,1]
        with self.assertRaises(ValueError):inverse_sqrt(reference_covariance(flat,1e-12)*0)
        self.assertTrue(np.isfinite(inverse_sqrt(reference_covariance(flat,1e-3))).all())

    def test_direction_rule_and_paired_bootstrap(self):
        self.assertEqual(direction_rule([.01]*8+[-.01]*2,8)['direction'],'consistent')
        self.assertEqual(direction_rule([.01]*7+[-.01]*3,8)['direction'],'inconsistent_report_per_subject')
        table=pd.DataFrame([dict(subject_id=f'S{s}',k=k,d=.02) for s in range(10) for k in [3,6]])
        rows=bootstrap_deltas(table,'d',[f'S{s}' for s in range(10)],[3,6],200,1,8)
        for r in rows:self.assertAlmostEqual(r['ci_low'],.02);self.assertTrue(r['ci_excludes_zero'])
        with self.assertRaises(ValueError):bootstrap_deltas(table.iloc[1:],'d',[f'S{s}' for s in range(10)],[3],10,1,8)


if __name__=='__main__':unittest.main()
