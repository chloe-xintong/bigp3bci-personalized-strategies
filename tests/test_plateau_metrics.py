"""Local-platform features must ignore DC offset and reject pure level steps."""
from pathlib import Path
import json,sys,unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from bigp3bci_plateau_metrics import platform_excursion

class PlateauTests(unittest.TestCase):
 def setUp(self):
  self.cfg=json.loads((ROOT/'configs/local_plateau_metrics.json').read_text(encoding='utf-8-sig'))
  self.fs=256;self.peak=512;t=np.arange(1024)/256
  self.wave=np.tile(80e-6*np.exp(-.5*((t-2)/.05)**2),(3,1))
 def test_offset_invariance_and_both_polarities(self):
  a=platform_excursion(self.wave,self.peak,self.fs,self.cfg)
  b=platform_excursion(self.wave+150e-6,self.peak,self.fs,self.cfg)
  c=platform_excursion(-self.wave,self.peak,self.fs,self.cfg)
  self.assertAlmostEqual(a['local_excursion_uv'],b['local_excursion_uv'])
  self.assertAlmostEqual(a['local_excursion_uv'],c['local_excursion_uv'])
  self.assertTrue(a['meets_excursion_candidate']);self.assertTrue(c['meets_excursion_candidate'])
 def test_level_step_is_not_an_independent_lobe(self):
  x=np.zeros((3,1024));x[:,500:]=100e-6
  r=platform_excursion(x,self.peak,self.fs,self.cfg)
  self.assertEqual(r['local_excursion_uv'],0.)
  self.assertFalse(r['meets_excursion_candidate'])
 def test_future_after_observation_window_cannot_change_metric(self):
  a=platform_excursion(self.wave,self.peak,self.fs,self.cfg)
  cut=a['required_decision_sample']+1
  changed=self.wave.copy();changed[:,cut:]=1.
  b=platform_excursion(changed,self.peak,self.fs,self.cfg)
  self.assertEqual(a,b)
  missing=platform_excursion(self.wave[:,:self.peak+1],self.peak,self.fs,self.cfg)
  self.assertFalse(missing['platform_context_complete']);self.assertFalse(missing['meets_excursion_candidate'])

if __name__=='__main__':unittest.main()
