"""Frozen-trigger blink main/recovery localization and epoch-loss comparison."""
from pathlib import Path
import json
import os
import sys
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('_MNE_FAKE_HOME_DIR',str(ROOT))
sys.path.insert(0,str(ROOT/'src'))
import mne
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from bigp3bci_preprocessing import file_hash
from bigp3bci_continuous_qc import read_bigp3bci_edf
from bigp3bci_blink_boundaries import localization_signal,localize_boundaries,epoch_boundary_links

OUT=ROOT/'outputs/blink_boundary_localization_v1'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((ROOT/'configs/blink_boundary_localization.json').read_text())
    detector_path=ROOT/'configs/expanded_pilot.json';detector=json.loads(detector_path.read_text());detector_hash=file_hash(detector_path)
    manifest_path=ROOT/'outputs/blink_candidate_review/file_index.csv';manifest=pd.read_csv(manifest_path)
    old_policy=pd.read_csv(ROOT/'outputs/three_policy_comparison/all_epoch_policy_decisions.csv',usecols=['edf_name','event_index','strategy','common_fault','common_fault_reasons'])
    base=old_policy[old_policy.strategy.eq('faults_only')].drop(columns='strategy')
    all_rows=[];all_flags=[];all_links=[];all_decisions=[];audits=[];manual_pairs=[];lobes=[]
    before_hashes={str((ROOT/r.processing_results/'blink_candidates.csv').relative_to(ROOT)):file_hash(ROOT/r.processing_results/'blink_candidates.csv') for r in manifest.itertuples()}
    old_results_record=ROOT/'outputs/three_policy_comparison/comparison_record.json';policy_hash=file_hash(old_results_record)
    for r in manifest.itertuples():
        print(r.edf_name,flush=True)
        folder=ROOT/r.processing_results;dest=OUT/Path(r.edf_name).stem;dest.mkdir(exist_ok=True)
        rec=json.loads((folder/'processing_record.json').read_text())
        if rec['config']['blink_detector']!=detector['blink_detector']:raise ValueError('Detector parameters changed')
        old=pd.read_csv(folder/'blink_candidates.csv')
        events=pd.read_csv(folder/'epoch_candidates.csv')
        with np.load(folder/'continuous_eeg.npz') as data:
            formal=data['eeg_v'];aux=data['auxiliary_eeg_v'];channels=data['channels'].tolist();fs=float(data['sfreq'])
        edf=Path(r.source_edf);edf_before=file_hash(edf)
        raw=read_bigp3bci_edf(edf);x=raw.get_data(picks=channels);raw.close()
        local,contaminated,sos,state=localization_signal(x,fs,cfg['localization_branch'])
        boundaries,possible_lobes=localize_boundaries(x,formal,local,old,channels,fs,cfg,contaminated)
        boundaries.to_csv(dest/'boundaries.csv',index=False)
        possible_lobes.to_csv(dest/'possible_merged_lobes.csv',index=False)
        flags,links=epoch_boundary_links(events,boundaries,fs,cfg)
        if not np.array_equal(flags.has_old_overlap.to_numpy(),events.has_blink_candidate.to_numpy()):raise AssertionError('Old candidate overlap changed')
        flags.to_csv(dest/'epoch_boundary_flags.csv',index=False);links.to_csv(dest/'epoch_boundary_links.csv',index=False)
        np.savez_compressed(dest/'localization_branch.npz',eeg_v=local,channels=channels,sfreq=fs,sos=sos,final_state=state)
        # Distinct annotation files; no overwrite of previous review exports.
        for region,ac,bc in [('main','main_onset_seconds','main_offset_seconds'),('recovery','recovery_onset_seconds','recovery_offset_seconds')]:
            onset=boundaries[ac].to_numpy(float);duration=boundaries[bc].to_numpy(float)-onset
            keep=duration>0
            labels=np.array(['CANDIDATE_blink_'+region+('_uncertain' if not row.localization_usable_for_provisional_comparison else '_unconfirmed' if region=='recovery' and row.recovery_status!='quiet_criterion_met_unvalidated' else '') for row in boundaries.itertuples()],dtype=str)
            annotations=mne.Annotations(onset[keep],duration[keep],labels[keep])
            path=dest/(Path(r.edf_name).stem+'-'+region+'-annot.fif');annotations.save(path,overwrite=True,verbose='ERROR')
            loaded=mne.read_annotations(path)
            np.testing.assert_allclose(loaded.onset,np.sort(onset[keep]),atol=1/fs)
        combined=events.merge(flags,on='event_index',validate='one_to_one').merge(base[base.edf_name.eq(r.edf_name)][['event_index','common_fault','common_fault_reasons']],on='event_index',validate='one_to_one')
        # A localized start can precede the old trigger. A negative decision
        # must wait long enough for a future trigger to back-localize into it.
        backward=cfg['peak_search_before_trigger_seconds']+cfg['onset_lookback_seconds']
        forward=max(detector['blink_detector']['max_duration_seconds']+max(detector['blink_detector']['recovery_seconds'],detector['blink_detector']['merge_gap_seconds']),
                    cfg['peak_search_after_trigger_seconds']+cfg['main_end_search_after_peak_seconds']+cfg['recovery']['maximum_after_main_seconds']+cfg['recovery']['quiet_seconds'])
        guard=(int(np.ceil(backward*fs))+int(np.ceil(forward*fs)))/fs
        old_guard=(int(np.ceil(detector['blink_detector']['max_duration_seconds']*fs))+max(int(np.ceil(detector['blink_detector']['recovery_seconds']*fs)),int(np.ceil(detector['blink_detector']['merge_gap_seconds']*fs))))/fs
        decisions=[]
        for scheme,marker in [('faults_only',None),('old_candidate_overlap','has_old_overlap'),('main_only_with_old_fallback','has_main_policy_overlap'),('main_plus_recovery_with_old_fallback','has_main_recovery_policy_overlap')]:
            copy=combined.copy()
            hit=np.zeros(len(copy),bool) if marker is None else copy[marker].to_numpy(bool)
            copy['scheme']=scheme;copy['interval_hit']=hit;copy['would_keep']=~copy.common_fault & ~hit
            copy['extra_interval_rejection']=~copy.common_fault & hit
            copy['policy_decision_seconds']=copy.epoch_decision_seconds+(0 if marker is None else old_guard if scheme=='old_candidate_overlap' else guard)
            copy['pending_at_eof']=copy.policy_decision_seconds>(x.shape[1]-1)/fs
            copy['comparison_only']=True
            decisions.append(copy)
        decisions=pd.concat(decisions,ignore_index=True);decisions.to_csv(dest/'policy_comparison_epochs.csv',index=False)
        all_decisions.append(decisions)
        audits.append(dict(edf_name=r.edf_name,n_candidates=len(old),localized_main_usable=int(boundaries.localization_usable_for_provisional_comparison.sum()),
                           old_fallback_candidates=int((~boundaries.localization_usable_for_provisional_comparison.astype(bool)).sum()),
                           possible_multi_lobe_candidates=int(boundaries.possible_merged_lobes.sum()),
                           near_duplicate_peak_candidates=int(boundaries.near_duplicate_localized_peak.sum()),
                           main_onset_order_inversions=int((boundaries.main_onset_sample.astype(float).diff()<0).sum()),
                           recovery_not_confirmed=int(boundaries.recovery_status.ne('quiet_criterion_met_unvalidated').sum()),
                           new_policy_conservative_extra_wait_seconds=guard))
        boundaries['subject_id']=r.subject_id;boundaries['session_id']=r.session_id
        all_rows.append(boundaries);all_flags.append(combined);all_links.append(links.assign(edf_name=r.edf_name));lobes.append(possible_lobes.assign(edf_name=r.edf_name))
        manual_path=ROOT/'outputs/artifact_qc/continuous_manual_qc/manual/manual_review_tables'/(Path(r.edf_name).stem+'_manual_review.csv')
        if manual_path.exists():
            manual=pd.read_csv(manual_path)
            for m in manual.itertuples():
                # Compare the same frozen parent/manual associations; no rematch
                # used to inflate apparent performance. Domains are explicit.
                hits=boundaries[(boundaries.old_onset_seconds<m.end_sec)&(boundaries.old_offset_seconds>m.start_sec)] if len(boundaries) else boundaries
                for b in hits.itertuples():
                    manual_pairs.append(dict(edf_name=r.edf_name,manual_id=m.manual_id,blink_candidate_id=b.blink_candidate_id,
                                             manual_start_sec=m.start_sec,manual_end_sec=m.end_sec,manual_domain='original raw EEG',
                                             old_domain='official+causal5Hz',new_domain='raw+causal10Hz',
                                             old_onset_delta=b.old_onset_seconds-m.start_sec,old_offset_delta=b.old_offset_seconds-m.end_sec,
                                             new_onset_delta=b.main_onset_seconds-m.start_sec,new_offset_delta=b.main_offset_seconds-m.end_sec,
                                             new_main_usable=b.localization_usable_for_provisional_comparison,
                                             interpretation='descriptive domain mismatch; not boundary accuracy, no constant shift correction'))
        assert edf_before==file_hash(edf)
        (dest/'processing_record.json').write_text(json.dumps(dict(source_record_sha256=file_hash(folder/'processing_record.json'),source_edf_sha256=edf_before,
                    source_candidates_sha256=before_hashes[str((folder/'blink_candidates.csv').relative_to(ROOT))],old_candidates_unchanged=True,
                    localization_sos=sos.tolist(),cfg=cfg,n_candidates=len(old),new_policy_conservative_extra_wait_seconds=guard),indent=2),encoding='utf8')
    boundaries=pd.concat(all_rows,ignore_index=True);flags=pd.concat(all_flags,ignore_index=True);decisions=pd.concat(all_decisions,ignore_index=True)
    boundaries.to_csv(OUT/'all_boundaries.csv',index=False);flags.to_csv(OUT/'all_epoch_boundary_flags.csv',index=False)
    pd.concat(all_links,ignore_index=True).to_csv(OUT/'all_epoch_boundary_links.csv',index=False)
    pd.concat(lobes,ignore_index=True).to_csv(OUT/'possible_merged_lobes.csv',index=False)
    pd.DataFrame(audits).to_csv(OUT/'localization_audit.csv',index=False);pd.DataFrame(manual_pairs).to_csv(OUT/'legacy_manual_domain_comparison.csv',index=False)
    decisions.to_csv(OUT/'all_policy_comparison_epochs.csv',index=False)
    for name,keys in [('overall',[]),('edf',['edf_name']),('subject',['subject_id']),('session',['subject_id','session_id']),('label',['edf_name','label']),('character',['edf_name','character_trial_id']),('stimulus_group',['edf_name','character_trial_id','stimulus_group_id'])]:
        summary=decisions.groupby(['scheme']+keys).agg(total=('would_keep','size'),kept=('would_keep','sum'),retention=('would_keep','mean'),extra_interval_rejections=('extra_interval_rejection','sum'),pending_at_eof=('pending_at_eof','sum')).reset_index()
        summary.to_csv(OUT/(name+'_summary.csv'),index=False)
    wide=decisions.pivot(index=['edf_name','event_index'],columns='scheme',values='would_keep')
    changes=[]
    for scheme in ['main_only_with_old_fallback','main_plus_recovery_with_old_fallback']:
        changes.append(dict(scheme=scheme,newly_kept_vs_old=int((wide[scheme]&~wide.old_candidate_overlap).sum()),newly_flagged_vs_old=int((~wide[scheme]&wide.old_candidate_overlap).sum())))
    pd.DataFrame(changes).to_csv(OUT/'epoch_set_changes.csv',index=False)
    assert (wide.main_plus_recovery_with_old_fallback<=wide.main_only_with_old_fallback).all()
    if int((~wide.old_candidate_overlap&wide.faults_only).sum())!=2105:raise AssertionError('Old 2105-overlap reference changed')
    for relative,value in before_hashes.items():assert file_hash(ROOT/relative)==value
    assert file_hash(detector_path)==detector_hash and file_hash(old_results_record)==policy_hash
    (OUT/'config_snapshot.json').write_text(json.dumps(cfg,indent=2),encoding='utf8')
    (OUT/'run_record.json').write_text(json.dumps(dict(files=len(manifest),frozen_candidates=len(boundaries),detector_config_sha256=detector_hash,
             frozen_trigger_ids_preserved=True,old_candidate_files_unchanged=True,old_policy_record_unchanged=True,actual_epoch_deletion=False,
             time_shift_correction=False,recovery_cleanliness_validated=False,main_boundaries_validated=False,
             localization_usable_candidates=int(boundaries.localization_usable_for_provisional_comparison.sum()),
             config=cfg),indent=2),encoding='utf8')
    from bigp3bci_boundary_review import render_boundary_examples
    render_boundary_examples(ROOT,OUT)
    print(pd.read_csv(OUT/'overall_summary.csv').to_string(index=False),flush=True)
    print(pd.DataFrame(audits).to_string(index=False),flush=True)

if __name__=='__main__':main()
