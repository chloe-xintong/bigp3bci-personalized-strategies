import sys
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from bigp3bci_probability import balanced_log_loss,binary_class_weights,fit_grouped_training_platt
from bigp3bci_flash_early_stopping import FlashLogLossEarlyStopping


class ProbabilityMetricTests(unittest.TestCase):
    def test_equal_class_contribution_independent_of_negative_repeats(self):
        expected=-.5*(np.log(.2)+np.log(.9))
        self.assertAlmostEqual(balanced_log_loss([0,1],[.1,.2]),expected)
        self.assertAlmostEqual(balanced_log_loss([0,0,0,1],[.1,.1,.1,.2]),expected)
        self.assertAlmostEqual(balanced_log_loss([0,1],[.5,.5]),np.log(2))
        w=binary_class_weights([0,0,0,1])
        self.assertEqual(3*w[0],w[1])

    def test_invalid_probabilities_not_silently_accepted(self):
        for p in [[-.1,.9],[.1,1.1],[.1,np.nan]]:
            with self.assertRaises(ValueError):balanced_log_loss([0,1],p)

    def test_whole_character_cross_fit_and_single_character_fallback(self):
        groups=np.repeat(['a','b','c'],4)
        y=np.tile([0,1,0,1],3)
        x=np.column_stack([np.repeat([0,1,2],4),y+.01])
        model=dict(weights=np.array([0.,1.]),intercept=0.)
        fitted_groups=[]
        def fit(a,b,c):
            fitted_groups.append(set(a[:,0].astype(int)))
            return model
        with patch('bigp3bci_probability.fit_swlda',side_effect=lambda a,b,c:dict(fit(a,b,c),status='test')):
            cal=fit_grouped_training_platt(x,y,groups,{},model)
        self.assertEqual(len(fitted_groups),3)
        for i,used in enumerate(fitted_groups):self.assertNotIn(i,used)
        self.assertFalse(cal['validation_labels_used'])
        with patch('bigp3bci_probability.fit_swlda') as fit:
            single=fit_grouped_training_platt(x[:4],y[:4],groups[:4],{},model)
            fit.assert_not_called()
        self.assertEqual(single['method'],'training_in_sample_single_character')

    def test_logloss_checkpoint_uses_minimum(self):
        stop=FlashLogLossEarlyStopping(patience=2)
        self.assertTrue(stop.update([0,1],[.3,.7],0)['improved'])
        self.assertTrue(stop.update([0,1],[.1,.9],1)['improved'])
        self.assertFalse(stop.update([0,1],[.2,.8],2)['should_stop'])
        result=stop.update([0,1],[.25,.75],3)
        self.assertTrue(result['should_stop']);self.assertEqual(result['best_epoch'],1)


if __name__=='__main__':unittest.main()
