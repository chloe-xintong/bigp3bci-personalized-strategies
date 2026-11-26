"""CPU SWLDA experiment: frozen character split, A/B/C policies and full-trial decoding."""
from pathlib import Path
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
 os.environ[key]='1'
import json,re,sys,time
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('_MNE_FAKE_HOME_DIR',str(ROOT));sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pandas as pd
from scipy import stats
from bigp3bci_preprocessing import file_hash
from bigp3bci_data_policy import require_train_edf,require_train_record
from bigp3bci_continuous_qc import read_bigp3bci_edf
from bigp3bci_swlda import fit_swlda,score_swlda,window_features,auc_rank,decode_character
OUT=ROOT/'outputs/swlda_rejection_experiment_v1'


def freeze_csv(frame,path):
 if path.exists():
  previous=pd.read_csv(path,keep_default_na=False)
  current=pd.read_csv(__import__('io').StringIO(frame.to_csv(index=False)),keep_default_na=False)
  if not previous.equals(current):raise ValueError(f'Frozen manifest changed: {path}; choose a new experiment version')
 frame.to_csv(path,index=False)


def prepare(cfg):
 manifest=pd.read_csv(ROOT/cfg['input_manifest'])
 policy=pd.read_csv(ROOT/cfg['policy_decisions'],low_memory=False,usecols=['edf_name','event_index','strategy','would_keep','rejection_reasons','trigger_channels','policy_decision_seconds','decision_data_available','common_fault'])
 reverse={value:key for key,value in cfg['strategies'].items()}
 policy=policy[policy.strategy.isin(reverse)].copy();policy['policy']=policy.strategy.map(reverse)
 feature_arrays=[];memberships=[];metadata=[];trials=[];audit=[];sources=[];characters=None;edge_rows=[];group_dictionaries=[]
 row_count=0
 for item in manifest.itertuples():
  folder=ROOT/item.results_path
  rec=json.loads((folder/'processing_record.json').read_text())
  if rec['config']['blink_detector']!=cfg['blink_detector']:raise ValueError('Frozen 25uV detector parameters differ')
  for source_file,expected in [('processing_record.json',item.source_record_sha256),('epoch_candidates.csv',item.epoch_table_sha256),('blink_epoch_links.csv',item.blink_links_sha256)]:
   if file_hash(folder/source_file)!=expected:raise ValueError('Preprocessing provenance changed: '+str(folder/source_file))
  group_dictionaries.append(pd.read_csv(folder/'stimulus_group_dictionary.csv').assign(edf_name=item.edf_name))
  path=Path(rec['source_edf'])
  require_train_edf(path)
  if rec['channels']!=cfg['features']['channels']:raise ValueError('Different channel order')
  source_before=file_hash(path)
  e=pd.read_csv(folder/'epoch_candidates.csv')
  with np.load(folder/'epochs_all.npz') as saved:
   if saved['channels'].tolist()!=cfg['features']['channels']:raise ValueError('Epoch channel order differs')
   if not np.array_equal(saved['event_index'],e.event_index):raise ValueError('Epoch/event ID mismatch')
   x,edges=window_features(saved['eeg_v'],float(saved['sfreq']),cfg['features'])
  feature_arrays.append(x)
  e['feature_row']=np.arange(row_count,row_count+len(e));row_count+=len(e)
  e['model_id']=e.subject_id+'_'+e.session_id
  raw=read_bigp3bci_edf(path)
  names=sorted([n for n in raw.ch_names if re.match(r'^.+_\d+_\d+$',n)],key=lambda n:tuple(map(int,n.rsplit('_',2)[1:])))
  if len(names)!=36 or len(set(tuple(map(int,n.rsplit('_',2)[1:])) for n in names))!=36:raise ValueError('Cannot reconstruct 36-character matrix')
  if characters is not None and names!=characters:raise ValueError('Different character dictionaries across files')
  characters=names
  active=raw.get_data(picks=names)[:,e.onset_sample.to_numpy(int)].T
  if not np.isin(active,[0.,1.]).all():raise ValueError('Unexpected character channel states')
  active=active>.5;memberships.append(active)
  truth=e.current_target.to_numpy(int)
  if not np.isin(truth,np.arange(1,37)).all():raise ValueError('CurrentTarget outside verified matrix')
  raw_type=e.stimulus_type_raw.to_numpy()
  membership=active[np.arange(len(e)),truth-1]
  disagreement=int((membership!=raw_type).sum())
  if disagreement:raise ValueError('CurrentTarget row-major mapping inconsistent with actual flashed characters/StimulusType')
  state_begin=raw.get_data(picks=['StimulusBegin'])[0]
  onset_values=state_begin[e.onset_sample.to_numpy(int)]
  if not (onset_values>.5).all():raise ValueError('Onset alignment mismatch')
  raw.close()
  if source_before!=file_hash(path):raise ValueError('Raw EDF changed')
  state_audit=pd.read_csv(folder/'character_event_audit.csv').set_index('character_trial_id')
  fs=rec['sfreq']
  for trial,g in e.groupby('character_trial_id',sort=True):
   if trial<1 or trial not in state_audit.index:raise ValueError('Unassigned character events; mapping incomplete')
   a=state_audit.loc[trial]
   if len(g)!=int(a.n_flashes) or g.current_target.nunique()!=1:raise ValueError('Character boundary/current target inconsistent')
   if not g.character_boundary_known.all():raise ValueError('Character has unknown boundary')
   target=int(g.current_target.iloc[0])-1
   trials.append(dict(model_id=g.model_id.iloc[0],subject_id=item.subject_id,session_id=item.session_id,site=item.site,
    edf_name=item.edf_name,character_trial_id=int(trial),character_key=f'{Path(item.edf_name).stem}_T{int(trial):02d}',
    trial_start_sample=max(0,int(a.start_boundary_exclusive)),trial_end_sample=int(a.end_boundary_exclusive),
    trial_start_seconds=max(0,float(a.start_boundary_exclusive))/fs,trial_end_seconds=float(a.end_boundary_exclusive)/fs,
    first_flash_seconds=float(g.onset_seconds.min()),last_flash_seconds=float(g.onset_seconds.max()),
    flashes=len(g),target_flashes=int(g.label.eq('target').sum()),nontarget_flashes=int(g.label.eq('non-target').sum()),
    true_character_index=target,true_character=names[target],observed_closed_trial=True,canonical_126_flashes=len(g)==126,
    in_frozen_pilot=item.in_frozen_pilot,sampling_hz=fs))
  audit.append(dict(edf_name=item.edf_name,subject_id=item.subject_id,session_id=item.session_id,n_flashes=len(e),
      n_character_state_channels=len(names),target_membership_label_disagreements=disagreement,
      source_truth='CurrentTarget 1-based row-major index empirically verified against character states and raw StimulusType',
      prediction_mapping='actual character-channel membership at each onset; no StimulusCode or true labels in decoder',
      true_ids_observed=json.dumps(sorted(set(truth.tolist())))))
  sources.append(dict(edf_name=item.edf_name,source_edf_sha256=source_before,epoch_npz_sha256=file_hash(folder/'epochs_all.npz'),
                      source_events_sha256=file_hash(folder/'epoch_candidates.csv'),source_blinks_sha256=file_hash(folder/'blink_candidates.csv')))
  for bin_index,(start,stop) in enumerate(zip(edges[:-1],edges[1:])):
   edge_rows.append(dict(edf_name=item.edf_name,bin_index=bin_index,start_sample=int(start),stop_sample_exclusive=int(stop),n_samples=int(stop-start),sampling_hz=fs))
  metadata.append(e)
  print('Prepared',item.edf_name,len(e),'flashes',flush=True)
 e=pd.concat(metadata,ignore_index=True);x=np.vstack(feature_arrays);active=np.vstack(memberships)
 trial_table=pd.DataFrame(trials)
 # Fixed before any epoch rejection or model fitting. No label-based split.
 split_rows=[]
 for model_id,g in trial_table.groupby('model_id',sort=True):
  g=g.sort_values(['edf_name','character_trial_id']).copy()
  ntrain=int(np.floor(len(g)*cfg['split']['train_fraction']))
  if ntrain<cfg['split']['minimum_train_characters'] or len(g)-ntrain<cfg['split']['minimum_validation_characters']:raise ValueError('Insufficient closed trials for fixed split')
  g['set']=['train']*ntrain+['validation']*(len(g)-ntrain)
  g['acquisition_order']=np.arange(len(g))
  split_rows.append(g)
 split=pd.concat(split_rows,ignore_index=True)
 freeze_csv(split,OUT/'character_split.csv')
 split_keys=split[['edf_name','character_trial_id','character_key','set','acquisition_order']]
 e=e.merge(split_keys,on=['edf_name','character_trial_id'],validate='many_to_one')
 policy=policy.merge(e[['edf_name','event_index','feature_row','model_id','set','character_key']],on=['edf_name','event_index'],validate='many_to_one')
 # Validate exact A/B/C semantics against the original frozen epoch table.
 for edf,rows in e.groupby('edf_name'):
  q=policy[policy.edf_name.eq(edf)].pivot(index='event_index',columns='policy',values='would_keep').reindex(rows.event_index)
  if not np.array_equal(q.C.to_numpy(),q.A.to_numpy() & ~rows.has_blink_candidate.to_numpy()):raise ValueError('C mixes interval versions or QC baselines')
  if not np.array_equal(q.B.to_numpy(),q.A.to_numpy() & ~(rows.max_ptp_uv.to_numpy()>150.)):raise ValueError('B is not A plus 150uV PTP')
 leak=[]
 for edf,g in e.groupby('edf_name'):
  train=g[g.set.eq('train')];val=g[g.set.eq('validation')]
  if not len(train) or not len(val):continue
  gap=int(val.onset_sample.min()-train.epoch_stop_sample_exclusive.max())
  if gap<0:raise ValueError('Training/validation epochs share source samples')
  for strategy in cfg['strategies']:
   q=policy[(policy.edf_name==edf)&(policy.policy==strategy)&policy.set.eq('train')]
   future_gap=float(val.onset_seconds.min()-q.policy_decision_seconds.max())
   if future_gap<=0:raise ValueError('Training policy confirmation consumes validation-period data')
   leak.append(dict(edf_name=edf,policy=strategy,raw_sample_gap=gap,training_policy_to_first_validation_seconds=future_gap,
                    shared_raw_samples=False,training_confirmation_enters_validation=False))
 if e.groupby(['edf_name','character_trial_id']).set.nunique().max()!=1:raise ValueError('Character split leakage')
 freeze_csv(e[['edf_name','event_index','character_key','model_id','set','feature_row']],OUT/'epoch_split.csv')
 # Full three-EDF sessions split exactly between files. Independent recordings
 # have separate sample clocks and filter state; never invent a cross-file gap.
 for model_id,g in e.groupby('model_id'):
  train_files=set(g.loc[g.set.eq('train'),'edf_name'])
  val_files=set(g.loc[g.set.eq('validation'),'edf_name'])
  if train_files.isdisjoint(val_files):
   for strategy in cfg['strategies']:
    leak.append(dict(edf_name='',model_id=model_id,policy=strategy,raw_sample_gap=np.nan,
      training_policy_to_first_validation_seconds=np.nan,shared_raw_samples=False,
      training_confirmation_enters_validation=False,scope='independent_EDFs_separate_sample_clocks',
      train_edfs=';'.join(sorted(train_files)),validation_edfs=';'.join(sorted(val_files))))
 pd.DataFrame(leak).to_csv(OUT/'split_leakage_audit.csv',index=False)
 pd.DataFrame(audit).to_csv(OUT/'mapping_audit.csv',index=False)
 pd.DataFrame(sources).to_csv(OUT/'input_provenance.csv',index=False)
 pd.DataFrame(edge_rows).to_csv(OUT/'feature_sample_bins.csv',index=False)
 pd.concat(group_dictionaries,ignore_index=True).to_csv(OUT/'stimulus_group_dictionary.csv',index=False)
 dictionary=pd.DataFrame([dict(character_index=i,current_target_value=i+1,character_channel=n,row=int(n.rsplit('_',2)[1]),column=int(n.rsplit('_',2)[2])) for i,n in enumerate(characters)])
 dictionary.to_csv(OUT/'character_dictionary.csv',index=False)
 np.savez_compressed(OUT/'features_and_membership.npz',features_uv=x,flashed=active,characters=characters)
 feature_order=pd.DataFrame([dict(feature_index=ch*16+t,channel=name,bin_index=t,start_ms=t*50,end_ms=(t+1)*50) for ch,name in enumerate(cfg['features']['channels']) for t in range(16)])
 feature_order.to_csv(OUT/'feature_dictionary.csv',index=False)
 return manifest,e,x,active,split,policy,characters,feature_order


def run_models(cfg,e,x,active,split,policy,characters,feature_order):
 scores=[];predictions=[];character_evidence=[];group_counts=[];models=[];auc_rows=[];budgets=[]
 model_dir=OUT/'models';model_dir.mkdir(exist_ok=True)
 for model_id,g in e.groupby('model_id',sort=True):
  meta=split[split.model_id.eq(model_id)]
  train_trials=meta[meta.set.eq('train')];val_trials=meta[meta.set.eq('validation')]
  acquired=sum(float(h.trial_end_seconds.max()-h.trial_start_seconds.min()) for _,h in train_trials.groupby('edf_name'))
  file_groups=g[['subject_id','session_id','site']].iloc[0].to_dict()
  fold_scores={}
  for strategy in cfg['strategies']:
   q=policy[(policy.model_id==model_id)&(policy.policy==strategy)].copy()
   q=q.merge(e.drop(columns=['model_id','set','feature_row','character_key']),on=['edf_name','event_index'],validate='one_to_one')
   q=q.sort_values('feature_row').reset_index(drop=True)
   train=q.set.eq('train')&q.would_keep
   y=q.label.map({'non-target':0,'target':1}).to_numpy(float)
   finite=np.isfinite(x[q.feature_row.to_numpy(int)]).all(axis=1)
   train &= np.isfinite(y)&finite
   fit_rows=q.loc[train,'feature_row'].to_numpy(int)
   start=time.perf_counter()
   try:
    model=fit_swlda(x[fit_rows],y[train],cfg['swlda'])
    fit_status=model['status']
   except ValueError as exc:
    model=None;fit_status='unavailable: '+str(exc)
   elapsed=time.perf_counter()-start
   score=np.full(len(q),np.nan)
   can_score=q.would_keep.to_numpy(bool)&finite
   if model is not None:score[can_score]=score_swlda(model,x[q.loc[can_score,'feature_row'].to_numpy(int)])
   q['target_score']=score;q['has_score']=np.isfinite(score)
   q['score_status']=np.where(~q.would_keep,'rejected_no_score',np.where(q.has_score,'scored','model_unavailable_or_nonfinite'))
   q['policy']=strategy;q['fit_status']=fit_status
   scores.append(q)
   fold_scores[strategy]=q
   nselected=len(model['selected_features']) if model else 0
   info=dict(model_id=model_id,policy=strategy,**file_groups,fit_status=fit_status,n_selected=nselected,
      training_characters=len(train_trials),validation_characters=len(val_trials),training_epochs=len(fit_rows),
      training_target_epochs=int(y[train].sum()),fit_cpu_wall_seconds=elapsed,acquired_calibration_seconds=acquired)
   models.append(info)
   budgets.append(dict(**info,training_collected_flashes=int(q.set.eq('train').sum()),validation_collected_flashes=int(q.set.eq('validation').sum()),
       training_kept_epochs=int((q.set.eq('train')&q.would_keep).sum()),validation_kept_epochs=int((q.set.eq('validation')&q.would_keep).sum())))
   if model is not None:
    prefix=model_dir/(model_id+'_'+strategy)
    np.savez_compressed(str(prefix)+'.npz',weights_uv=model['weights'],intercept=model['intercept'],train_mean=model['train_mean'],train_scale=model['train_scale'],selected_features=model['selected_features'])
    serial={key:value for key,value in model.items() if key not in ['weights','train_mean','train_scale']}
    serial.update(info)
    Path(str(prefix)+'.json').write_text(json.dumps(serial,indent=2),encoding='utf8')
    selected=feature_order[feature_order.feature_index.isin(model['selected_features'])].copy()
    selected['weight_per_uv']=model['weights'][selected.feature_index.to_numpy(int)]
    selected.to_csv(str(prefix)+'_selected_features.csv',index=False)
   valid=q[q.set.eq('validation')]
   labels=valid.label.map({'non-target':0,'target':1}).to_numpy(float)
   auc_rows.append(dict(model_id=model_id,policy=strategy,**file_groups,scope='own_valid_scored_epochs',n_epochs=int(valid.has_score.sum()),auc=auc_rank(labels,valid.target_score)))
   for trial in val_trials.itertuples():
    h=valid[valid.character_key.eq(trial.character_key)].sort_values('onset_sample')
    ids=h.feature_row.to_numpy(int)
    d=decode_character(h.target_score.to_numpy(float),active[ids],cfg['character_evidence'])
    index=d['predicted_index'];name=characters[index] if index is not None else ''
    deadline=max(float(trial.trial_end_seconds),float(h.policy_decision_seconds.max()))
    data_available=bool(h.decision_data_available.all())
    # Refuse to claim an online output if confirmation data never arrived.
    status=d['status'] if data_available else 'confirmation_pending_at_eof'
    if not data_available:index=None;name=''
    correct=bool(index is not None and index==trial.true_character_index)
    predictions.append(dict(model_id=model_id,policy=strategy,**file_groups,edf_name=trial.edf_name,
      character_key=trial.character_key,character_trial_id=trial.character_trial_id,true_character=trial.true_character,
      predicted_character=name,prediction_status=status,correct=correct,output_available=index is not None,
      collected_flashes=len(h),valid_flashes=int(h.would_keep.sum()),scored_flashes=d['n_scored'],
      min_active_evidence=int(d['n_active'].min()),max_active_evidence=int(d['n_active'].max()),
      trial_start_seconds=trial.trial_start_seconds,trial_end_seconds=trial.trial_end_seconds,
      first_flash_seconds=trial.first_flash_seconds,last_flash_seconds=trial.last_flash_seconds,
      evidence_ready_seconds=deadline,prediction_available_seconds=deadline if index is not None else np.nan,
      decision_latency_after_trial_end_seconds=deadline-trial.trial_end_seconds,
      output_latency_after_trial_end_seconds=deadline-trial.trial_end_seconds if index is not None else np.nan,
      elapsed_since_first_flash_seconds=deadline-trial.first_flash_seconds,
      elapsed_since_trial_start_seconds=deadline-trial.trial_start_seconds,
      training_characters=len(train_trials),calibration_acquisition_seconds=acquired))
    for k,ch in enumerate(characters):
     character_evidence.append(dict(model_id=model_id,policy=strategy,edf_name=trial.edf_name,character_key=trial.character_key,
       candidate_character=ch,active_valid_flashes=int(d['n_active'][k]),inactive_valid_flashes=int(d['n_inactive'][k]),normalized_evidence=d['evidence'][k]))
    for stim,gg in h.groupby('stimulus_group_id'):
     group_counts.append(dict(model_id=model_id,policy=strategy,edf_name=trial.edf_name,character_key=trial.character_key,
       stimulus_group_id=stim,total_flashes=len(gg),valid_flashes=int(gg.would_keep.sum()),scored_flashes=int(gg.has_score.sum())))
   print('SWLDA',model_id,strategy,'train',len(fit_rows),'features',nselected,fit_status,flush=True)
  # Fair auxiliary AUC on exactly the same scored validation event IDs.
  common=None
  for strategy,q in fold_scores.items():
   keys=set(zip(q.loc[q.set.eq('validation')&q.has_score,'edf_name'],q.loc[q.set.eq('validation')&q.has_score,'event_index']))
   common=keys if common is None else common&keys
  for strategy,q in fold_scores.items():
   selected=q[q.set.eq('validation')].copy()
   keep=np.array([(a,b) in common for a,b in zip(selected.edf_name,selected.event_index)])
   selected=selected[keep]
   auc_rows.append(dict(model_id=model_id,policy=strategy,**file_groups,scope='common_valid_scored_epochs_A_B_C',n_epochs=len(selected),
       auc=auc_rank(selected.label.map({'non-target':0,'target':1}).to_numpy(float),selected.target_score)))
 result=dict(scores=pd.concat(scores,ignore_index=True),predictions=pd.DataFrame(predictions),evidence=pd.DataFrame(character_evidence),groups=pd.DataFrame(group_counts),
             models=pd.DataFrame(models),auc=pd.DataFrame(auc_rows),budgets=pd.DataFrame(budgets))
 return result


def main():
 global OUT
 import argparse
 parser=argparse.ArgumentParser()
 parser.add_argument('--config',default='configs/swlda_rejection_experiment.json')
 parser.add_argument('--output',default='outputs/swlda_rejection_experiment_v1')
 args=parser.parse_args()
 OUT=ROOT/args.output
 OUT.mkdir(parents=True,exist_ok=True)
 config_path=ROOT/args.config;cfg=json.loads(config_path.read_text())
 snapshot=OUT/'config_snapshot.json'
 if snapshot.exists() and json.loads(snapshot.read_text())!=cfg:raise ValueError('Experiment settings changed; use a new version')
 snapshot.write_text(json.dumps(cfg,indent=2),encoding='utf8')
 manifest,e,x,active,split,policy,characters,feature_order=prepare(cfg)
 inventory=split.groupby(['subject_id','session_id','set']).agg(character_trials=('character_key','size'),flashes=('flashes','sum')).reset_index()
 inventory.to_csv(OUT/'character_inventory_and_split.csv',index=False)
 # Membership data used by decoder is frozen independently of all true labels.
 result=run_models(cfg,e,x,active,split,policy,characters,feature_order)
 for key,name in [('scores','epoch_scores'),('predictions','character_predictions'),('evidence','character_evidence'),('groups','stimulus_group_flash_counts'),('models','model_summary'),('auc','validation_auc'),('budgets','calibration_and_retention')]:
  result[key].to_csv(OUT/(name+'.csv'),index=False)
 pred=result['predictions']
 expected=set(split.loc[split.set.eq('validation'),'character_key'])
 for strategy in cfg['strategies']:
  if set(pred.loc[pred.policy.eq(strategy),'character_key'])!=expected:raise AssertionError('Validation characters differ between strategies')
 for level,keys in [('overall',[]),('subject',['subject_id']),('session',['subject_id','session_id']),('edf',['edf_name'])]:
  summary=pred.groupby(['policy']+keys).agg(validation_characters=('correct','size'),correct_characters=('correct','sum'),character_accuracy=('correct','mean'),
   output_characters=('output_available','sum'),mean_decision_wait_after_trial_seconds=('decision_latency_after_trial_end_seconds','mean'),mean_output_wait_after_trial_seconds=('output_latency_after_trial_end_seconds','mean'),mean_valid_flashes=('valid_flashes','mean')).reset_index()
  summary['no_output_characters']=summary.validation_characters-summary.output_characters
  summary.to_csv(OUT/(level+'_character_summary.csv'),index=False)
 own=result['scores'][result['scores'].set.eq('validation')]
 own.groupby(['policy','subject_id','session_id']).agg(collected=('would_keep','size'),kept=('would_keep','sum'),retention=('would_keep','mean'),scored=('has_score','sum')).reset_index().to_csv(OUT/'validation_retention.csv',index=False)
 result['auc'].groupby(['policy','scope']).agg(model_folds=('auc','count'),macro_auc=('auc','mean')).reset_index().to_csv(OUT/'overall_macro_auc.csv',index=False)
 paired=pred.pivot(index=['model_id','edf_name','character_key','true_character'],columns='policy',values=['correct','predicted_character','output_available','valid_flashes','prediction_available_seconds','evidence_ready_seconds'])
 paired.columns=['_'.join(c) for c in paired.columns];paired=paired.reset_index();paired.to_csv(OUT/'paired_character_comparison.csv',index=False)
 comparisons=[]
 for first,second in [('A','B'),('A','C'),('B','C')]:
  a=paired['correct_'+first].astype(bool);b=paired['correct_'+second].astype(bool)
  gains=int((~a&b).sum());losses=int((a&~b).sum())
  comparisons.append(dict(first=first,second=second,characters=len(paired),second_only_correct=gains,first_only_correct=losses,
    accuracy_difference=float(b.mean()-a.mean()),paired_exact_binomial_p=float(stats.binomtest(gains,gains+losses,.5).pvalue) if gains+losses else 1.,
    mean_extra_wait_seconds=float((paired['evidence_ready_seconds_'+second]-paired['evidence_ready_seconds_'+first]).mean()),
    note='descriptive paired development comparison; no preprocessing retuning from this result'))
 pd.DataFrame(comparisons).to_csv(OUT/'paired_strategy_summary.csv',index=False)
 (OUT/'implementation_record.json').write_text(json.dumps(dict(classifier='SWLDA, project-owned OLS forward/backward stepwise implementation',
   implementation_file='src/bigp3bci_swlda.py',implementation_sha256=file_hash(ROOT/'src/bigp3bci_swlda.py'),
   reference=cfg['swlda']['reference'],source_config_sha256=file_hash(config_path),split_sha256=file_hash(OUT/'character_split.csv'),
   feature_count=256,models=len(result['models']),training_characters=int(split.set.eq('train').sum()),validation_characters=int(split.set.eq('validation').sum()),
   used_final_test_data=False,used_gpu=False,feature_selection_training_only=True,ordinary_or_regularized_lda_substitution=False,
   mapping_available=True,mapping_disagreement_count=0,baseline=None,
   timing_limit='sample-clock simulation includes epoch/confirmation/character waits, excludes unmeasured hardware communication and per-epoch compute latency',
   C_interval_version=cfg['blink_interval_version'],
   model_statuses=result['models'].fit_status.value_counts().to_dict(),software=dict(numpy=np.__version__,pandas=pd.__version__,scipy=__import__('scipy').__version__),
   limitation='small chronological development holdout; prior preprocessing development used these Train records, not independent final evaluation'),indent=2),encoding='utf8')
 print(pd.read_csv(OUT/'overall_character_summary.csv').to_string(index=False),flush=True)
 print(pd.DataFrame(comparisons).to_string(index=False),flush=True)

if __name__=='__main__':main()
