"""Frozen full-coverage Train selection; reuse identical cached preprocessing."""
from pathlib import Path
import os,sys,json
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('_MNE_FAKE_HOME_DIR',str(ROOT))
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pandas as pd
from bigp3bci_expanded_pilot import process_expanded_file
from bigp3bci_policy_comparison import evaluate_policies
from bigp3bci_preprocessing import file_hash
OUT=ROOT/'outputs/studyf_all_preprocessing'

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 cfg=json.loads((ROOT/'configs/expanded_pilot.json').read_text())
 policy_cfg=json.loads((ROOT/'configs/three_policy_comparison.json').read_text())
 existing=pd.read_csv(ROOT/'outputs/full_three_session_preprocessing/input_manifest.csv').set_index('edf_name')
 sites={'F_03':'ETSU','F_05':'Duke','F_06':'Duke','F_07':'Duke','F_08':'Duke','F_20':'ETSU','F_21':'ETSU','F_23':'ETSU','F_24':'ETSU','F_25':'ETSU'}
 rows=[]
 for subject,site in sites.items():
  for session in ['SE001','SE002','SE003']:
   paths=sorted((ROOT/'data/raw/bigP3BCI-data/StudyF'/subject/session/'Train/CBCol').glob('*.edf'))
   if len(paths)!=3:raise ValueError(f'Expected 3 Train EDFs: {subject} {session}')
   rows.extend(dict(edf_name=p.name,path=str(p),subject_id=subject,session_id=session,site=site,site_status='inferred') for p in paths)
 available=sorted((ROOT/'data/raw/bigP3BCI-data/StudyF').glob('*/SE*/Train/*/*.edf'))
 if {str(p) for p in available}!={r['path'] for r in rows}:raise ValueError('StudyF selection is not exhaustive')
 selection=pd.DataFrame(rows)
 target=OUT/'selection_manifest.csv'
 if target.exists() and not pd.read_csv(target).equals(selection):raise ValueError('Frozen selection changed')
 selection.to_csv(target,index=False)
 records=[];frames=[]
 for i,r in enumerate(selection.itertuples(),1):
  folder=ROOT/existing.loc[r.edf_name,'results_path'] if r.edf_name in existing.index else OUT/Path(r.edf_name).stem
  record_path=folder/'processing_record.json'
  if not record_path.exists():
   print(f'[{i}/90] processing {r.edf_name}',flush=True)
   process_expanded_file(Path(r.path),r.site,cfg,folder,ROOT)
  else:print(f'[{i}/90] reuse {r.edf_name}',flush=True)
  rec=json.loads(record_path.read_text())
  # Manual annotation file lists do not enter filtering, detection or policies.
  if {k:v for k,v in rec['config'].items() if k!='manual_validation'}!={k:v for k,v in cfg.items() if k!='manual_validation'}:
   raise ValueError('Cached preprocessing differs: '+r.edf_name)
  if file_hash(Path(r.path))!=rec['source_sha256_after']:raise ValueError('Source changed')
  e=pd.read_csv(folder/'epoch_candidates.csv',keep_default_na=False)
  q=pd.read_csv(folder/'raw_quality_epoch_links.csv',keep_default_na=False)
  b=pd.read_csv(folder/'blink_epoch_links.csv',keep_default_na=False)
  long=pd.read_csv(folder/'epoch_ptp_long.csv')
  ptp=long.pivot(index='event_index',columns='channel',values='ptp_uv').reindex(index=e.event_index,columns=rec['channels']).to_numpy()
  policies=evaluate_policies(e,ptp,q,b,rec['channels'],rec['sfreq'],rec['n_samples'],cfg['blink_detector'],policy_cfg)
  in_pilot=bool(existing.loc[r.edf_name,'in_frozen_pilot']) if r.edf_name in existing.index else False
  policies['in_frozen_pilot']=in_pilot
  frames.append(policies)
  records.append(dict(edf_name=r.edf_name,results_path=str(folder.relative_to(ROOT)),source_record_sha256=file_hash(record_path),epoch_table_sha256=file_hash(folder/'epoch_candidates.csv'),blink_links_sha256=file_hash(folder/'blink_epoch_links.csv'),subject_id=r.subject_id,session_id=r.session_id,site=r.site,in_frozen_pilot=in_pilot))
 pd.DataFrame(records).to_csv(OUT/'input_manifest.csv',index=False)
 pd.concat(frames,ignore_index=True).to_csv(OUT/'all_epoch_policy_decisions.csv',index=False)
 print('DONE: all StudyF, 90 Train EDFs, 30 subject/session groups',flush=True)

if __name__=='__main__':main()
