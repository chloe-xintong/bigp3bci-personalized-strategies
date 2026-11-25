"""Human candidate-review persistence must not overwrite source annotations."""
from pathlib import Path
import json
import os
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('_MNE_FAKE_HOME_DIR',str(ROOT))
sys.path.insert(0,str(ROOT/'src'))
import mne
import numpy as np
import pandas as pd
from bigp3bci_candidate_review import save_review_spans

class ReviewSaveTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory(dir=ROOT/'tests');self.root=Path(self.temp.name)
  self.out=self.root/'outputs/blink_candidate_review/demo';self.out.mkdir(parents=True)
  self.rec=self.root/'record';self.rec.mkdir()
  (self.rec/'processing_record.json').write_text(json.dumps({'n_samples':500,'sfreq':100}))
  c=pd.DataFrame({'blink_candidate_id':['B1','B2'],'onset_seconds':[.8,2.8],'peak_seconds':[1.,3.],'offset_seconds':[1.3,3.3]})
  c.to_csv(self.out/'candidate_details.csv',index=False)
  c.assign(review_decision='',review_notes='').to_csv(self.out/'review_decisions.csv',index=False)
  pd.DataFrame([dict(edf_name='demo.edf',processing_results='record',candidate_details_csv='outputs/blink_candidate_review/demo/candidate_details.csv',review_decisions_csv='outputs/blink_candidate_review/demo/review_decisions.csv')]).to_csv(self.out.parent/'file_index.csv',index=False)
  self.raw=mne.io.RawArray(np.zeros((1,500)),mne.create_info(['EEG_Fz'],100,['eeg']),verbose=False)
  self.raw._blink_review_edf='demo.edf'
 def tearDown(self):
  self.raw.close();self.temp.cleanup()
 def test_only_review_spans_saved_and_peak_mapping(self):
  before=(self.out/'candidate_details.csv').read_bytes()
  self.raw.set_annotations(mne.Annotations([.8,.5,.9,2.81],[.5,.3,.3,.05],['CANDIDATE_blink','MANUAL_bad_span','REVIEW_not_blink','REVIEW_not_blink']))
  csv,fif,spans=save_review_spans(self.root,'demo.edf',self.raw)
  d=pd.read_csv(csv,keep_default_na=False)
  self.assertEqual(d.review_decision.tolist(),['not_blink',''])
  self.assertEqual(len(mne.read_annotations(fif)),2)
  self.assertEqual(spans.blink_candidate_ids.tolist(),['B1',''])
  self.assertEqual(before,(self.out/'candidate_details.csv').read_bytes())
 def test_conflict_is_uncertain(self):
  self.raw.set_annotations(mne.Annotations([.8,.8],[.5,.5],['REVIEW_not_blink','REVIEW_confirmed_blink']))
  csv,_,_=save_review_spans(self.root,'demo.edf',self.raw)
  d=pd.read_csv(csv,keep_default_na=False)
  self.assertEqual(d.review_decision.tolist(),['uncertain',''])
 def test_default_bad_is_saved_as_not_blink(self):
  self.raw.set_annotations(mne.Annotations([.8,2.8],[.5,.5],['BAD','BAD_blink']))
  csv,fif,spans=save_review_spans(self.root,'demo.edf',self.raw)
  d=pd.read_csv(csv,keep_default_na=False)
  self.assertEqual(d.review_decision.tolist(),['not_blink','not_blink'])
  self.assertEqual(list(mne.read_annotations(fif).description),['REVIEW_not_blink','REVIEW_not_blink'])
 def test_empty_window_preserves_saved_reviews(self):
  self.raw.set_annotations(mne.Annotations([.8],[.5],['BAD']))
  csv,fif,_=save_review_spans(self.root,'demo.edf',self.raw)
  self.raw.set_annotations(mne.Annotations([],[],[]))
  with self.assertRaises(ValueError):save_review_spans(self.root,'demo.edf',self.raw)
  self.assertEqual(len(mne.read_annotations(fif)),1)
 def test_main_only_plot_preserves_flash_events_and_valid_color_mapping(self):
  from unittest.mock import patch
  from bigp3bci_candidate_review import open_candidate_review
  with patch('bigp3bci_candidate_review.load_candidate_review',return_value=(self.raw,np.empty((0,3),int),pd.DataFrame(),Path('unused.csv'))), patch.object(self.raw,'plot',return_value='browser') as plot:
   result=open_candidate_review(self.root,'demo.edf',main_only=True)
  self.assertEqual(plot.call_args.kwargs['events'].shape,(0,3))
  self.assertEqual(plot.call_args.kwargs['event_color'],{})
  self.assertEqual(result[1],'browser')
 def test_wrong_edf_not_saved(self):
  self.raw._blink_review_edf='other.edf'
  with self.assertRaises(ValueError):save_review_spans(self.root,'demo.edf',self.raw)

if __name__=='__main__':unittest.main()
