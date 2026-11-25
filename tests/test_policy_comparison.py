"""Policy comparison tests: common baseline, independent criteria and future guard."""
from pathlib import Path
import json
import sys
import unittest
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from bigp3bci_policy_comparison import evaluate_policies

class PolicyTests(unittest.TestCase):
 def setUp(self):
  self.c=json.loads((ROOT/'configs/three_policy_comparison.json').read_text())
  self.d=json.loads((ROOT/'configs/expanded_pilot.json').read_text())['blink_detector']
  self.channels=self.c['qc_channels']
  self.e=pd.DataFrame([dict(event_index=i,label='target',complete_epoch=True,filter_startup=False,
   nonfinite_flag=False,nonfinite_channels='',filter_recovery=False,filter_recovery_channels='',
   has_blink_candidate=i==2,epoch_stop_sample_exclusive=2000+i,epoch_decision_seconds=(2000+i)/256) for i in range(4)])
  self.p=np.full((4,16),20.);self.p[1,0]=300.;self.p[3]=0
  self.q=pd.DataFrame([dict(event_index=3,channel=self.channels[0],label='CANDIDATE_constant',known_by_epoch_decision=True)])
  self.b=pd.DataFrame([dict(event_index=2,blink_decision_seconds=8.1,known_by_epoch_decision=False)])
 def run_policy(self,e=None,n=5000):
  return evaluate_policies(self.e if e is None else e,self.p,self.q,self.b,self.channels,256,n,self.d,self.c)
 def test_three_distinct_criteria(self):
  r=self.run_policy().pivot(index='event_index',columns='strategy',values='would_keep')
  self.assertTrue(r.loc[1,'faults_only']);self.assertFalse(r.loc[1,'ptp_250uv']);self.assertTrue(r.loc[1,'blink_overlap'])
  self.assertTrue(r.loc[2,'faults_only']);self.assertTrue(r.loc[2,'ptp_250uv']);self.assertFalse(r.loc[2,'blink_overlap'])
  self.assertFalse(r.loc[3].any())
 def test_labels_and_low_ptp_not_used_for_base(self):
  e=self.e.copy();e.label='unknown'
  pd.testing.assert_frame_equal(self.run_policy().drop(columns='label'),self.run_policy(e).drop(columns='label'))
  q=self.q.copy();self.q=self.q.iloc[:0]
  r=self.run_policy();self.q=q
  self.assertTrue(r[(r.event_index==3)&r.strategy.eq('faults_only')].would_keep.iloc[0])
 def test_absence_confirmation_wait_and_eof(self):
  r=self.run_policy(n=2050)
  blink=r[r.strategy.eq('blink_overlap')]
  self.assertTrue((blink.additional_wait_seconds>1.3).all())
  self.assertTrue(blink.pending_at_eof.all())
  # Pending does not silently turn a retrospective keep into a signal fault.
  self.assertTrue(blink.loc[blink.event_index.eq(0),'would_keep'].iloc[0])
  self.assertFalse(r[r.strategy.eq('faults_only')].pending_at_eof.any())

if __name__=='__main__':unittest.main()
