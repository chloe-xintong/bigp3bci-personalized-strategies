"""Causal main/recovery localization tests, separate from frozen triggering."""
from pathlib import Path
import json,os,sys,unittest
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('_MNE_FAKE_HOME_DIR',str(ROOT));sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pandas as pd
from bigp3bci_blink_boundaries import localization_signal,localize_boundaries,epoch_boundary_links

class BoundaryTests(unittest.TestCase):
 def setUp(self):
  self.cfg=json.loads((ROOT/'configs/blink_boundary_localization.json').read_text())
  self.fs=256;self.channels=['EEG_F3','EEG_Fz','EEG_F4','EEG_Pz']
  t=np.arange(12*256)/256
  wave=70e-6*np.exp(-.5*((t-4)/.12)**2)
  self.x=np.stack([wave,wave,wave,wave*.1])
  self.old=pd.DataFrame([dict(edf_name='synthetic.edf',blink_candidate_id='B1',onset_sample=1011,offset_sample_exclusive=1200,decision_sample=1240,onset_seconds=1011/256,offset_seconds=1200/256,decision_seconds=1240/256)])
 def localize(self,x,old=None):
  loc,mask,_,_=localization_signal(x,self.fs,self.cfg['localization_branch'])
  return localize_boundaries(x,x,loc,self.old if old is None else old,self.channels,self.fs,self.cfg,mask)
 def test_history_onset_and_independent_recovery(self):
  old=self.old.copy(deep=True);b,_=self.localize(self.x)
  r=b.iloc[0]
  self.assertLess(r.main_onset_sample,self.old.onset_sample.iloc[0])
  self.assertGreater(r.main_offset_sample_exclusive,r.main_peak_sample)
  self.assertGreaterEqual(r.recovery_onset_sample,r.main_offset_sample_exclusive)
  self.assertGreaterEqual(r.decision_sample,r.main_decision_sample)
  self.assertGreaterEqual(r.main_decision_sample,self.old.decision_sample.iloc[0])
  self.assertFalse(r.recovery_proves_clean_signal)
  pd.testing.assert_frame_equal(old,self.old)
 def test_future_cannot_change_completed_boundaries(self):
  b,_=self.localize(self.x);r=b.iloc[0];cut=int(r.decision_sample)+1
  self.assertLess(cut,self.x.shape[1])
  changed=self.x.copy();changed[:,cut:]+=1.
  c,_=self.localize(changed)
  prefix,_=self.localize(self.x[:,:cut])
  cols=['main_onset_sample','main_peak_sample','main_offset_sample_exclusive','main_decision_sample','recovery_offset_sample_exclusive','decision_sample']
  pd.testing.assert_frame_equal(b[cols],c[cols])
  pd.testing.assert_frame_equal(b[cols],prefix[cols])
 def test_polarity_symmetry(self):
  pos,_=self.localize(self.x);neg,_=self.localize(-self.x)
  cols=['main_onset_sample','main_peak_sample','main_offset_sample_exclusive','recovery_offset_sample_exclusive']
  pd.testing.assert_frame_equal(pos[cols],neg[cols])
 def test_adjacent_lobes_flagged_not_silently_merged(self):
  t=np.arange(self.x.shape[1])/256
  wave=70e-6*(np.exp(-.5*((t-4)/.05)**2)+np.exp(-.5*((t-4.35)/.05)**2))
  x=np.stack([wave,wave,wave,wave*.1])
  b,audit=self.localize(x)
  self.assertEqual(b.blink_candidate_id.tolist(),['B1'])
  self.assertTrue(b.possible_merged_lobes.iloc[0]);self.assertGreater(len(audit),0)
 def test_unknown_boundary_uses_old_interval(self):
  b,_=self.localize(np.zeros_like(self.x))
  self.assertFalse(b.localization_usable_for_provisional_comparison.iloc[0])
  e=pd.DataFrame([dict(event_index=0,onset_sample=1100,epoch_stop_sample_exclusive=1305,epoch_decision_seconds=1305/256)])
  flags,links=epoch_boundary_links(e,b,self.fs,self.cfg)
  self.assertTrue(flags.has_old_overlap.iloc[0]);self.assertTrue(flags.has_main_policy_overlap.iloc[0])

if __name__=='__main__':unittest.main()
