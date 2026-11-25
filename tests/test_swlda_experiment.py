"""SWLDA and label-free character evidence regression checks."""
from pathlib import Path
import json,sys,unittest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from scipy import stats
from bigp3bci_swlda import fit_swlda,score_swlda,window_features,auc_rank,decode_character


def reference_stepwise(x,y,cfg):
 """Slow direct-design OLS reference, independent from Gram/projection path."""
 n,p=x.shape;selected=[]
 def ols(indices):
  d=np.column_stack([np.ones(n),x[:,indices]])
  coef=np.linalg.lstsq(d,y,rcond=1e-10)[0]
  rss=float(np.sum((y-d@coef)**2))
  return d,coef,rss
 for _ in range(100):
  changed=False;_,_,rss=ols(selected)
  candidates=[]
  if len(selected)<cfg['max_features']:
   for j in range(p):
    if j in selected:continue
    _,_,added=ols(selected+[j]);df=n-len(selected)-2
    f=max(0.,rss-added)/(added/df)
    candidates.append((stats.f.sf(f,1,df),j))
  if candidates:
   probability,j=min(candidates)
   if probability<cfg['p_enter']:selected.append(j);changed=True
  while selected:
   d,coef,rss=ols(selected)
   var=(rss/(n-len(selected)-1))*np.diag(np.linalg.pinv(d.T@d))[1:]
   probs=2*stats.t.sf(np.abs(coef[1:]/np.sqrt(var)),n-len(selected)-1)
   worst=int(np.argmax(probs))
   if probs[worst]<=cfg['p_remove']:break
   selected.pop(worst);changed=True
  if not changed or len(selected)>=cfg['max_features']:break
 return selected

class SWLDATests(unittest.TestCase):
 def setUp(self):
  self.cfg=json.loads((ROOT/'configs/swlda_rejection_experiment.json').read_text())
 def test_matches_direct_ols_stepwise_reference(self):
  rng=np.random.default_rng(20261125);x=rng.normal(size=(250,9))
  y=(1.6*x[:,0]-.9*x[:,3]+rng.normal(size=250)>.8).astype(int)
  c=self.cfg['swlda'].copy();c['max_features']=6
  expected=reference_stepwise(x,y,c);model=fit_swlda(x,y,c)
  self.assertEqual(model['selected_features'],expected)
  d=np.column_stack([np.ones(len(x)),x[:,expected]])
  score=d@np.linalg.lstsq(d,y,rcond=1e-10)[0]
  np.testing.assert_allclose(score_swlda(model,x),score,rtol=1e-8,atol=1e-9)
 def test_collinear_and_constant_features(self):
  rng=np.random.default_rng(4);a=rng.normal(size=200)
  x=np.column_stack([a,a,np.ones(200),rng.normal(size=200)])
  y=(a+rng.normal(size=200)*.1>0).astype(int)
  m=fit_swlda(x,y,self.cfg['swlda'])
  self.assertFalse(2 in m['selected_features'])
  self.assertLessEqual(len(set(m['selected_features'])&{0,1}),1)
  self.assertTrue(np.isfinite(score_swlda(m,x)).all())
 def test_feature_bins_integer_endpoint_and_channel_order(self):
  fs=256;time=np.arange(205)/fs
  x=np.stack([np.tile(time,(16,1))+np.arange(16)[:,None]])*1e-6
  features,edges=window_features(x,fs,self.cfg['features'])
  self.assertEqual(features.shape,(1,256));self.assertEqual(edges[-1],205)
  for ch in [0,7,15]:
   for t in range(16):
    expected=x[0,ch,(time>=t*.05-1e-12)&(time<(t+1)*.05-1e-12)].mean()*1e6
    self.assertAlmostEqual(features[0,ch*16+t],expected)
 def test_rejected_flash_is_missing_not_zero(self):
  c=self.cfg['character_evidence'].copy()
  d=decode_character([.8,np.nan],[[True,False],[False,True]],c)
  self.assertIsNone(d['predicted_index']);self.assertEqual(d['status'],'insufficient_evidence')
 def test_evidence_normalization_and_tie(self):
  c=self.cfg['character_evidence']
  a=decode_character([.8,.2],[[1,0],[0,1]],c)
  b=decode_character([.8]*10+[.2],[[1,0]]*10+[[0,1]],c)
  self.assertEqual(a['predicted_index'],0)
  np.testing.assert_allclose(a['evidence'],b['evidence'])
  t=decode_character([.3,.3],[[1,0],[0,1]],c)
  self.assertIsNone(t['predicted_index']);self.assertEqual(t['status'],'evidence_tie')
 def test_auc_ties(self):
  self.assertAlmostEqual(auc_rank([0,0,1,1],[0,1,1,2]),.875)

if __name__=='__main__':unittest.main()
