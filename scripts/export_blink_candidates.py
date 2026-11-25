"""Export fixed blink candidate annotations, preserving original EDF/manual files."""
from pathlib import Path
import json
import os
import sys
import shutil
import zipfile
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('_MNE_FAKE_HOME_DIR',str(ROOT))
sys.path.insert(0,str(ROOT/'src'))
import mne
import numpy as np
import pandas as pd
from bigp3bci_preprocessing import file_hash


def main():
 out=ROOT/'outputs/blink_candidate_review';out.mkdir(parents=True,exist_ok=True)
 cfg_path=ROOT/'configs/expanded_pilot.json';config=json.loads(cfg_path.read_text());before=file_hash(cfg_path)
 files=pd.read_csv(ROOT/'outputs/blink_annotation_update/all_processed_detection_summary.csv')
 supplements=pd.read_csv(ROOT/'outputs/blink_annotation_update/annotation_manifest.csv').set_index('edf_name')
 index=[];all_candidates=[]
 for row in files.itertuples():
  stem=Path(row.edf_name).stem
  base=ROOT/'outputs/expanded_pilot'/stem if row.in_frozen_pilot else ROOT/supplements.loc[row.edf_name,'results_path']
  rec=json.loads((base/'processing_record.json').read_text())
  if rec['config']['blink_detector']!=config['blink_detector']:raise ValueError('Different detector parameters')
  c=pd.read_csv(base/'blink_candidates.csv',keep_default_na=False)
  dest=out/stem;dest.mkdir(exist_ok=True)
  a=mne.Annotations(onset=c.onset_seconds.to_numpy(float),duration=(c.offset_seconds-c.onset_seconds).to_numpy(float),description=['CANDIDATE_blink']*len(c))
  fif=dest/(stem+'-blink-candidates-annot.fif');a.save(fif,overwrite=True,verbose='ERROR')
  # Standard MNE CSV and complete measurement table have distinct names.
  a.save(dest/(stem+'-blink-candidates-annot.csv'),overwrite=True,verbose='ERROR')
  loaded=mne.read_annotations(fif)
  assert len(loaded)==len(c)
  np.testing.assert_allclose(loaded.onset,c.onset_seconds.to_numpy(float),atol=1/rec['sfreq'])
  np.testing.assert_allclose(loaded.duration,c.offset_seconds.to_numpy(float)-c.onset_seconds.to_numpy(float),atol=1/rec['sfreq'])
  c.to_csv(dest/'candidate_details.csv',index=False)
  review=dest/'review_decisions.csv'
  if review.exists():
   prior=pd.read_csv(review,keep_default_na=False)
   if prior.blink_candidate_id.tolist()!=c.blink_candidate_id.tolist():raise ValueError('Candidate IDs changed; existing review decisions preserved; choose another export version')
  else:
   template=c[['edf_name','blink_candidate_id','onset_seconds','peak_seconds','offset_seconds','decision_seconds','filter_startup']].copy()
   template['review_decision']='';template['review_notes']=''
   template.to_csv(review,index=False)
  index.append(dict(edf_name=row.edf_name,subject_id=row.subject_id,session_id=row.session_id,site=row.site,site_status='inferred',
                    candidate_count=len(c),nonstartup_candidates=int(c.filter_startup.eq(False).sum()),
                    annotation_fif=str(fif.relative_to(ROOT)),candidate_details_csv=str((dest/'candidate_details.csv').relative_to(ROOT)),
                    review_decisions_csv=str(review.relative_to(ROOT)),processing_results=str(base.relative_to(ROOT)),source_edf=rec['source_edf'],
                    annotation_reload_verified=True))
  all_candidates.append(c)
 pd.DataFrame(index).to_csv(out/'file_index.csv',index=False)
 all_c=pd.concat(all_candidates,ignore_index=True);all_c.to_csv(out/'all_candidate_details.csv',index=False)
 (out/'detector_parameters.json').write_text(json.dumps(config['blink_detector'],indent=2),encoding='utf8')
 (out/'READ_ME.txt').write_text('''当前固定参数的疑似眨眼候选，不是人工真值。共12份训练EDF，参数未调整。

每份文件包含：
1. *-blink-candidates-annot.fif：MNE annotation，标签CANDIDATE_blink。
2. *-blink-candidates-annot.csv：MNE标准annotation CSV。
3. candidate_details.csv：候选ID、起止/峰/判定时间、指标、启动期标记。
4. review_decisions.csv：可填写confirmed_blink / not_blink / uncertain及review_notes；再次导出不会覆盖已有决定。

使用notebooks/11_bigp3bci_blink_candidate_review.ipynb查看。默认关闭GUI，可先选文件/模式再打开浏览器。
支持原始EEG、正式因果分析信号和辅助因果分支；标注区间始终来自辅助检测分支，原始信号上可能有相位错位。绿色为已有人工标注，橙色为自动候选。不要把图上显示标尺当成检测阈值。

onset/offset是候选波形区间，decision_seconds才是确认时刻；前10秒启动期候选也保留供检查。没有按延迟移动事件。未打标片段不等于正常，候选不意味着必须拒绝epoch。

文件不包含EDF信号本身；需要项目原始数据。原始EDF和人工标注没有修改。检测幅值25微伏不是epoch PTP上限250微伏。
''',encoding='utf8')
 assert before==file_hash(cfg_path)
 (out/'export_record.json').write_text(json.dumps(dict(files=len(index),candidates=len(all_c),config_sha256=before,parameters_unchanged=True,labels=['CANDIDATE_blink'],raw_edf_modified=False,manual_annotations_modified=False),indent=2),encoding='utf8')
 archive=out/'blink_candidate_annotations.zip'
 with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
  for path in sorted(out.rglob('*')):
   if path.is_file() and path!=archive:z.write(path,path.relative_to(out))
 print(f'Exported {len(index)} EDF annotation sets, {len(all_c)} candidates; FIF reload verified; {archive}')

if __name__=='__main__':main()
