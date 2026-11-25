"""Run representative training pilot; explicit EDF inputs support other sites."""
import argparse
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('_MNE_FAKE_HOME_DIR',str(ROOT))
sys.path.insert(0,str(ROOT/'src'))
import pandas as pd
from bigp3bci_preprocessing import process_edf


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/causal_pilot.json')
    parser.add_argument('--data-root',type=Path,default=ROOT/'data/raw/bigP3BCI-data/StudyF')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/causal_pilot')
    args=parser.parse_args()
    cfg=json.loads(args.config.read_text(encoding='utf-8-sig'))
    tables=[]
    for site,subject in cfg['pilot_subjects'].items():
        files=sorted((args.data_root/subject).glob('SE*/Train/CBCol/*.edf'))
        if not files: raise FileNotFoundError(f'No training EDF for {site}: {subject}')
        path=files[0]
        print(f'{site} (inferred): {path}',flush=True)
        tables.append(process_edf(path,site,cfg,args.output/path.stem))
    all_events=pd.concat(tables,ignore_index=True)
    all_events.to_csv(args.output/'all_epoch_metadata_quality.csv',index=False)
    for level,keys in [('edf',['subject_id','session_id','edf_name']),('session',['subject_id','session_id']),('subject',['subject_id'])]:
        summary=all_events.groupby(keys).valid_epoch.agg(total_flashes='size',valid_epochs='sum',retention='mean')
        summary.to_csv(args.output/f'{level}_summary.csv')
        for column,name in [('rejection_reason','reasons'),('trigger_channels','channels')]:
            expanded=all_events[keys].join(all_events[column].str.get_dummies(sep=';'))
            expanded.groupby(keys).sum().to_csv(args.output/f'{level}_{name}.csv')
        all_events.groupby(keys+['label']).valid_epoch.agg(['size','sum','mean']).to_csv(args.output/f'{level}_label_retention.csv')
    (args.output/'config_snapshot.json').write_text(json.dumps(cfg,indent=2),encoding='utf8')
    print(all_events.groupby(['site','edf_name']).valid_epoch.agg(['size','sum','mean']).to_string())

if __name__=='__main__': main()
