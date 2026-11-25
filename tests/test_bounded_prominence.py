"""Bounded prominence should be shift invariant, causal and uncertainty preserving."""
from pathlib import Path
import json,sys,unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from bigp3bci_bounded_prominence import bounded_prominence,confirmation_status

class ProminenceTests(unittest.TestCase):
 def setUp(self):
  self.cfg=json.loads((ROOT/'configs/bounded_prominence_secondary.json').read_text(encoding='utf-8-sig'))
  self.fs=256;self.t=np.arange(1024)/256;self.anchor=512
 def wave(self,amplitude=80e-6):
  return np.tile(amplitude*np.exp(-.5*((self.t-2)/.04)**2),(3,1))
 def test_four_uv_bump_on_minus_forty_platform_is_weak(self):
  f=bounded_prominence(self.wave(4e-6)-40e-6,self.anchor,self.fs,self.cfg)
  self.assertAlmostEqual(f['prominence_uv'],4.,places=3)
  self.assertEqual(confirmation_status(f,8,self.cfg)[0],'not_supported')
 def test_dc_shift_and_polarity_invariance(self):
  a=bounded_prominence(self.wave(),self.anchor,self.fs,self.cfg)
  b=bounded_prominence(self.wave()+100e-6,self.anchor,self.fs,self.cfg)
  c=bounded_prominence(-self.wave(),self.anchor,self.fs,self.cfg)
  self.assertAlmostEqual(a['prominence_uv'],b['prominence_uv'])
  self.assertAlmostEqual(a['prominence_uv'],c['prominence_uv'])
  self.assertEqual(confirmation_status(a,18,self.cfg)[0],'supported')
  self.assertEqual(confirmation_status(c,18,self.cfg)[0],'supported')
 def test_future_outside_declared_observation_irrelevant(self):
  x=self.wave();a=bounded_prominence(x,self.anchor,self.fs,self.cfg)
  cut=a['required_decision_sample']+1
  y=x.copy();y[:,cut:]=1.
  b=bounded_prominence(y,self.anchor,self.fs,self.cfg)
  self.assertEqual(a,b)
  prefix=bounded_prominence(x[:,:cut],self.anchor,self.fs,self.cfg)
  self.assertEqual(a,prefix)
 def test_missing_or_nonfinite_reference_is_uncertain(self):
  f=bounded_prominence(self.wave()[:,:520],self.anchor,self.fs,self.cfg)
  self.assertEqual(confirmation_status(f,12,self.cfg)[0],'uncertain')
  x=self.wave();x[0,500]=np.nan
  f=bounded_prominence(x,self.anchor,self.fs,self.cfg)
  self.assertEqual(confirmation_status(f,12,self.cfg)[0],'uncertain')
 def test_adjacent_true_peaks_never_forced_negative_without_background(self):
  x=np.tile(80e-6*(np.exp(-.5*((self.t-1.9)/.055)**2)+np.exp(-.5*((self.t-2.1)/.055)**2)),(3,1))
  for center in [1.9,2.1]:
   f=bounded_prominence(x,round(center*256),self.fs,self.cfg)
   self.assertIn(confirmation_status(f,18,self.cfg)[0],['supported','uncertain'])
 def test_neighbor_contaminated_reference_not_negative_evidence(self):
  f=bounded_prominence(self.wave(),self.anchor,self.fs,self.cfg,[(400,620)])
  self.assertTrue(f['reference_neighbor_overlap'])
  self.assertEqual(confirmation_status(f,18,self.cfg)[0],'uncertain')

if __name__=='__main__':unittest.main()
