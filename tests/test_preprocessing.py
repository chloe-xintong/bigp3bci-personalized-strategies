"""Regression checks for causal state, event bookkeeping, and QC leakage."""
import copy
import json
import os
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('_MNE_FAKE_HOME_DIR',str(ROOT))
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pandas as pd
import mne
from bigp3bci_preprocessing import design_filter,causal_filter,continuous_quality,epoch_quality,event_metadata

class PreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.cfg=json.loads((ROOT/'configs/causal_pilot.json').read_text())
        self.fs=256
        self.cfg['epoch_qc']['channels']=['EEG_Fz']
        self.x=np.random.default_rng(9).normal(0,10e-6,(1,4096))
    def test_streaming_and_future_independence(self):
        sos=design_filter(self.fs,self.cfg['filter'])
        a,_,_=causal_filter(self.x,sos,self.fs,self.cfg['filter'])
        cfg=copy.deepcopy(self.cfg['filter']); cfg['chunk_seconds']=.137
        b,_,_=causal_filter(self.x,sos,self.fs,cfg)
        np.testing.assert_allclose(a,b,rtol=0,atol=0)
        changed=self.x.copy(); changed[:,2000:]*=100
        c,_,_=causal_filter(changed,sos,self.fs,cfg)
        np.testing.assert_array_equal(a[:,:2000],c[:,:2000])
    def test_nonfinite_does_not_poison_file(self):
        x=self.x.copy(); x[0,10]=np.nan
        y,mask,_=causal_filter(x,design_filter(self.fs,self.cfg['filter']),self.fs,self.cfg['filter'])
        self.assertTrue(np.isfinite(y).all()); self.assertTrue(mask[0,10]); self.assertFalse(mask[0,-1])
    def test_labels_do_not_affect_qc_and_tail_retained(self):
        self.cfg['filter']['startup_seconds']=0
        windows=continuous_quality(self.x,self.fs,['EEG_Fz'],self.cfg['continuous_qc'])
        meta=pd.DataFrame({'onset_sample':[256,512,4000],'label':['target','non-target','target']})
        args=(self.x,self.x,np.zeros_like(self.x,dtype=bool))
        t,p,e=epoch_quality(*args,meta,windows,['EEG_Fz'],self.fs,self.cfg,[.5,250])
        swapped=meta.copy(); swapped['label']='unknown'
        u,_,_=epoch_quality(*args,swapped,windows,['EEG_Fz'],self.fs,self.cfg,[.5,250])
        self.assertEqual(t.valid_epoch.tolist(),u.valid_epoch.tolist())
        self.assertEqual(e.shape,(3,1,205)); self.assertIn('incomplete',t.iloc[-1].rejection_reason)
        self.assertTrue(t.valid_epoch.iloc[0]); self.assertTrue(t.valid_epoch.iloc[1])
    def test_persistent_channel_is_deferred(self):
        x=np.zeros_like(self.x)
        windows=continuous_quality(x,self.fs,['EEG_Fz'],self.cfg['continuous_qc'])
        meta=pd.DataFrame({'onset_sample':[3000]})
        t,_,_=epoch_quality(x,x,np.zeros_like(x,dtype=bool),meta,windows,['EEG_Fz'],self.fs,self.cfg,[.5,250])
        self.assertTrue(pd.isna(t.valid_epoch.iloc[0])); self.assertEqual(t.epoch_qc_status.iloc[0],'pending_channel_policy')
    def test_original_events_and_repeated_character_targets(self):
        x=np.zeros((5,1000)); x[1,[100,200,500]]=1; x[2,[100,500]]=1; x[3,[100,200,500]]=[1,2,1]; x[4,[300,700]]=1
        raw=mne.io.RawArray(x,mne.create_info(['EEG_Fz','StimulusBegin','StimulusType','StimulusCode','SelectedTarget'],256,['eeg']+['stim']*4),verbose=False)
        # event_metadata needs CurrentTarget, which may repeat across trials.
        target=mne.io.RawArray(np.ones((1,1000)),mne.create_info(['CurrentTarget'],256,['stim']),verbose=False)
        raw.add_channels([target])
        t=event_metadata(raw,Path('F_03/SE001/Train/CBCol/test.edf'),'ETSU')
        self.assertEqual(t.onset_sample.tolist(),[100,200,500]); self.assertEqual(t.character_trial_id.tolist(),[1,1,2]); self.assertEqual(t.label.tolist(),['target','non-target','target'])

if __name__=='__main__': unittest.main()
