import sys
import unittest
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from bigp3bci_calibration_budget import make_plan,training_keys
from bigp3bci_flash_early_stopping import FlashAUCEarlyStopping


class CalibrationBudgetTests(unittest.TestCase):
    def setUp(self):
        self.train=pd.DataFrame([dict(subject_id='F_05',session_id=s,edf_name=s+'_Train.edf',
            character_trial_id=i+1,character_key=s+'_'+str(i),acquisition_order=i,
            set='train' if i<2 else 'validation') for s in ['SE001','SE002','SE003'] for i in range(3)])

    def test_final_character_never_used_for_calibration(self):
        p=make_plan(self.train,['SE001','SE002'],'SE003')
        self.assertEqual(p['validation'],['SE003_2'])
        for k in range(3):
            for scheme in ['current_only','history_plus_current','history_se001_plus_current','history_se002_plus_current']:
                self.assertFalse(set(training_keys(p,scheme,k))&set(p['validation']))
            self.assertEqual(training_keys(p,'current_only',k),p['calibration'][:k])
        with self.assertRaises(ValueError):training_keys(p,'current_only',3)

    def test_single_history_isolation_and_endpoints(self):
        p=make_plan(self.train,['SE001','SE002'],'SE003')
        for scheme,s in [('history_se001_plus_current','SE001'),('history_se002_plus_current','SE002')]:
            self.assertEqual(training_keys(p,scheme,1),p['history_by_session'][s]+p['calibration'][:1])
        self.assertEqual(training_keys(p,'history_only',0),training_keys(p,'history_plus_current',0))
        self.assertEqual(training_keys(p,'current_max_endpoint',2),training_keys(p,'current_only',2))

    def test_labels_do_not_change_plan(self):
        a=make_plan(self.train,['SE001','SE002'],'SE003')
        other=self.train.copy();other['true_character_index']=35
        self.assertEqual(a,make_plan(other.sample(frac=1,random_state=1),['SE001','SE002'],'SE003'))

    def test_auc_early_stopping_tracks_best_epoch(self):
        stop=FlashAUCEarlyStopping(patience=2)
        self.assertTrue(stop.update([0,0,1,1],[0,1,0,1],0)['improved'])
        self.assertTrue(stop.update([0,0,1,1],[0,0,1,1],1)['improved'])
        self.assertFalse(stop.update([0,0,1,1],[0,1,0,1],2)['should_stop'])
        result=stop.update([0,0,1,1],[0,1,0,1],3)
        self.assertTrue(result['should_stop']);self.assertEqual(result['best_epoch'],1)
        with self.assertRaises(ValueError):stop.update([1,1],[.2,.3],4)


if __name__=='__main__':unittest.main()
