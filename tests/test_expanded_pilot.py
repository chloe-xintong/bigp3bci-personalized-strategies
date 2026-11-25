"""Behavioral tests for independent candidate detection and causal availability."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('_MNE_FAKE_HOME_DIR',str(ROOT))
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pandas as pd
from bigp3bci_expanded_pilot import (detect_blinks,basic_quality_candidates,associate_epochs,
                                     simulate_thresholds,manual_comparison,BLINK_COLUMNS)

class ExpandedPilotTests(unittest.TestCase):
    def setUp(self):
        self.cfg=json.loads((ROOT/'configs/expanded_pilot.json').read_text())
        self.cfg['filter']['startup_seconds']=0
        self.fs=256
        self.channels=self.cfg['simulation']['qc_channel_sets']['all16']
        time=np.arange(self.fs*12)/self.fs
        self.y=np.zeros((16,len(time)))
        wave=65e-6*np.exp(-.5*((time-4)/.14)**2)
        for ch in self.cfg['blink_detector']['frontal_channels']:self.y[self.channels.index(ch)]=wave
        for ch in self.cfg['blink_detector']['posterior_channels']:self.y[self.channels.index(ch)]=wave*.1
    def detect(self,y):
        return detect_blinks(y,self.fs,self.channels,self.cfg['blink_detector'],self.cfg['filter'],'synthetic.edf')[0]
    def test_bipolar_and_not_single_fz_peak(self):
        positive=self.detect(self.y);negative=self.detect(-self.y)
        self.assertEqual(len(positive),1);self.assertEqual(len(negative),1)
        self.assertEqual(positive.peak_sample.tolist(),negative.peak_sample.tolist())
        fz=np.zeros_like(self.y);fz[self.channels.index('EEG_Fz')]=self.y[self.channels.index('EEG_Fz')]*10
        self.assertEqual(len(self.detect(fz)),0)
    def test_no_input_mutation_and_causal_decision(self):
        original=self.y.copy();full=self.detect(self.y)
        np.testing.assert_array_equal(original,self.y)
        row=full.iloc[0]
        self.assertGreater(row.decision_seconds,row.offset_seconds)
        self.assertGreater(row.decision_seconds,row.peak_seconds)
        for cut in [int(row.peak_sample)+1,int(row.decision_sample),int(row.decision_sample)+1,2000]:
            prefix=self.detect(self.y[:,:cut])
            expected=full[full.decision_sample<cut]
            self.assertEqual(prefix.onset_sample.tolist(),expected.onset_sample.tolist())
            self.assertEqual(prefix.decision_sample.tolist(),expected.decision_sample.tolist())
        future=self.y.copy();future[:,2000:]=1
        future_candidates=self.detect(future)
        self.assertEqual(full.iloc[0].to_dict(),future_candidates.iloc[0].to_dict())
    def test_raw_faults_and_channels(self):
        raw=np.random.default_rng(1).normal(0,1e-6,(16,1024))
        raw[0,10:110]=0;raw[1,200]=np.inf;raw[2,500:]+=200e-6
        q=basic_quality_candidates(raw,self.fs,self.channels,self.cfg['signal_quality'])
        self.assertIn('CANDIDATE_zero',q.label.to_list())
        self.assertIn('CANDIDATE_nonfinite',q.label.to_list())
        self.assertTrue(((q.label=='CANDIDATE_jump')&(q.channel==self.channels[2])).any())
    def test_association_does_not_control_simulation(self):
        y=np.tile(np.sin(np.arange(3000)/25)[None,:]*10e-6,(16,1))
        meta=pd.DataFrame({'event_index':[0,1],'onset_sample':[1000,1064],'label':['target','non-target']})
        q=basic_quality_candidates(y,self.fs,self.channels,self.cfg['signal_quality'])
        b=pd.DataFrame([dict(blink_candidate_id='B1',onset_sample=1020,offset_sample_exclusive=1200,decision_seconds=5)],columns=['blink_candidate_id','onset_sample','offset_sample_exclusive','decision_seconds'])
        t,p,links,qlinks=associate_epochs(meta,y,y,np.zeros_like(y,dtype=bool),q,b,self.fs,self.channels,self.cfg)
        self.assertEqual(len(links),2);self.assertTrue(t.has_blink_candidate.all())
        self.assertNotIn('valid_epoch',t.columns)
        self.assertTrue((links.earliest_joint_decision_seconds>=links.blink_decision_seconds).all())
        a=simulate_thresholds(t,p,qlinks,self.channels,self.cfg,self.fs)
        changed=t.copy();changed['has_blink_candidate']=False;changed['label']='unknown'
        c=simulate_thresholds(changed,p,qlinks,self.channels,self.cfg,self.fs)
        pd.testing.assert_frame_equal(a,c)
        self.assertFalse(a.would_reject.any())
    def test_sparse_manual_is_not_ground_truth(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tests') as temp:
            root=Path(temp);cfg=copy.deepcopy(self.cfg)
            cfg['manual_validation']['manual_root']='manual'
            cfg['manual_validation']['reviewed_intervals_path']='reviewed.csv'
            cfg['manual_validation']['sparse_blink_files']=['synthetic.edf']
            path=root/'manual/manual_review_tables';path.mkdir(parents=True)
            candidate=self.detect(self.y);r=candidate.iloc[0]
            pd.DataFrame([dict(manual_id='M1',start_sec=r.onset_seconds-.01,end_sec=r.offset_seconds+.01,type='MANUAL_bad_span')]).to_csv(path/'synthetic_manual_review.csv',index=False)
            table=pd.DataFrame([dict(onset_sample=1000,epoch_stop_sample_exclusive=1205,filter_startup=False)])
            table.attrs['duration_seconds']=12
            output=root/'output';output.mkdir()
            status=manual_comparison(Path('synthetic.edf'),candidate,table,np.ones((1,16)),self.channels,self.fs,cfg,root,output)
            self.assertIsNone(status['reviewed_event_recall']);self.assertEqual(status['evaluated_candidates'],0)
            pd.DataFrame([dict(edf_name='synthetic.edf',start_sec=0,end_sec=12,review_status='complete')]).to_csv(root/'reviewed.csv',index=False)
            status=manual_comparison(Path('synthetic.edf'),candidate,table,np.ones((1,16)),self.channels,self.fs,cfg,root,output)
            self.assertEqual(status['evaluated_manual_events'],1);self.assertEqual(status['reviewed_event_recall'],1)

if __name__=='__main__':unittest.main()
