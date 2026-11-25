"""Causal, non-destructive Study F preprocessing. No fitting or label-based QC."""
from __future__ import annotations
import json
import hashlib
import html
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from bigp3bci_continuous_qc import read_bigp3bci_edf, get_all_eeg_channels, stimulus_begin_events


def design_filter(fs, cfg):
    if not 0 < cfg['band_hz'][0] < cfg['band_hz'][1] < fs / 2 or not 0 < cfg['notch_hz'] < fs / 2:
        raise ValueError('Band and notch must be below Nyquist')
    if cfg['family'] != 'butter':
        raise ValueError('Only butter is implemented; do not silently substitute families')
    b, a = signal.iirnotch(cfg['notch_hz'], cfg['notch_q'], fs=fs)
    return np.vstack([signal.tf2sos(b, a), signal.butter(cfg['order'], cfg['band_hz'], btype='bandpass', fs=fs, output='sos')])


def causal_filter(x, sos, fs, cfg):
    """Maintain SOS state across blocks; nonfinite inputs use last finite input.

    Replacement is ONLY to keep the recursive state finite. Affected data and
    the following startup-length recovery are invalid, never silently imputed.
    """
    if cfg['initialization'] not in ('steady_first_sample', 'zero'):
        raise ValueError('Unknown initialization')
    x = np.asarray(x, float)
    y = np.empty_like(x)
    contaminated = np.zeros_like(x, dtype=bool)
    final_states = []
    block = max(1, round(fs * cfg['chunk_seconds']))
    recovery = round(fs * cfg['startup_seconds'])
    for ch in range(len(x)):
        clean = x[ch].copy()
        last = clean[0] if np.isfinite(clean[0]) else 0.
        for t in np.flatnonzero(~np.isfinite(clean)):
            last = clean[t - 1] if t and np.isfinite(clean[t - 1]) else last
            clean[t] = last
            contaminated[ch, t:min(x.shape[1], t + recovery + 1)] = True
        zi = signal.sosfilt_zi(sos) * clean[0] if cfg['initialization'] == 'steady_first_sample' else np.zeros((len(sos), 2))
        for start in range(0, x.shape[1], block):
            stop = min(x.shape[1], start + block)
            y[ch, start:stop], zi = signal.sosfilt(sos, clean[start:stop], zi=zi)
        final_states.append(zi)
    return y, contaminated, np.array(final_states)


def continuous_quality(x, fs, channels, cfg):
    """Fixed, trailing windows. Persistence uses only completed window history."""
    n = x.shape[1]
    w, step = round(fs * cfg['window_seconds']), round(fs * cfg['step_seconds'])
    rows, count = [], np.zeros(len(channels), int)
    starts = list(range(0, max(0, n - w + 1), step))
    if not starts or starts[-1] + w < n:
        starts.append(max(0, n - w))
    for i, start in enumerate(starts):
        stop = min(n, start + w)
        for ch, name in enumerate(channels):
            z = x[ch, start:stop] * 1e6
            finite = np.isfinite(z)
            ptp = float(np.ptp(z[finite])) if finite.any() else np.nan
            std = float(np.std(z[finite])) if finite.any() else np.nan
            reasons = []
            if not finite.all(): reasons.append('nonfinite')
            if finite.all() and ptp == 0: reasons.append('constant')
            if finite.all() and np.all(z == 0): reasons.append('zero')
            if finite.all() and (ptp <= cfg['flat_ptp_uv'] or std <= cfg['low_std_uv']): reasons.append('low_variation')
            if finite.any() and np.max(np.abs(z[finite])) >= cfg['large_abs_uv']: reasons.append('large_amplitude')
            if ptp >= cfg['large_ptp_uv']: reasons.append('large_variation')
            count[ch] += bool(reasons)
            persistent = (stop / fs >= cfg['persistent_min_seconds'] and count[ch] / (i + 1) >= cfg['persistent_fraction'])
            rows.append(dict(channel=name, start_sample=start, stop_sample_exclusive=stop, available_sample=stop-1,
                             ptp_uv=ptp, std_uv=std, nonfinite_samples=int((~finite).sum()), reason=';'.join(reasons),
                             persistent_as_of_window=bool(persistent), history_anomaly_fraction=count[ch]/(i+1)))
    return pd.DataFrame(rows)


def event_metadata(raw, path, site):
    """Reuse verified flash helper and notebook SelectedTarget boundary rule."""
    events = stimulus_begin_events(raw)
    onsets = events[:, 0] - raw.first_samp
    selected = raw.get_data(picks=['SelectedTarget'])[0]
    boundaries = np.flatnonzero((selected > .5) & ~np.r_[False, selected[:-1] > .5])
    states = raw.get_data(picks=['StimulusCode', 'CurrentTarget'])
    rows = []
    counters = {}
    for index, (onset, event) in enumerate(zip(onsets, events)):
        trial = int(np.searchsorted(boundaries, onset, side='right')) + 1
        assigned = trial <= len(boundaries) and onset not in boundaries
        key = trial if assigned else -1
        counters[key] = counters.get(key, 0) + 1
        rows.append(dict(event_index=index, subject_id=path.parts[-5], session_id=path.parts[-4], source='StudyF', site=site,
                         site_evidence='inferred mapping from notebook 02', edf_name=path.name, split=path.parts[-3], condition=path.parts[-2],
                         character_trial_id=trial if assigned else -1, character_boundary_known=assigned, flash_index=counters[key],
                         onset_sample=int(onset), original_event_sample=int(event[0]), onset_seconds=onset/raw.info['sfreq'],
                         stimulus_code=states[0, onset], current_target=states[1, onset], event_code=int(event[2]),
                         label={1:'non-target', 2:'target', 3:'unknown'}[event[2]]))
    return pd.DataFrame(rows)


def epoch_quality(x, y, contaminated, metadata, windows, channels, fs, cfg, bounds):
    # [onset, onset + ceil(.8*fs)) includes samples with t < .8 seconds.
    size = int(np.ceil(cfg['epoch']['duration_seconds'] * fs))
    qc = cfg['epoch_qc']['channels']
    if not qc or len(qc) != len(set(qc)) or not set(qc) <= set(channels):
        raise ValueError(f'QC channels must be unique existing EEG channels: {qc}')
    picks = [channels.index(c) for c in qc]
    lo, hi = np.asarray(bounds) * 1e-6
    rows, ptps = [], []
    epochs = np.full((len(metadata), len(channels), size), np.nan)
    for i, row in metadata.iterrows():
        start = int(row.onset_sample)
        stop = start + size
        complete = stop <= y.shape[1]
        end = min(stop, y.shape[1])
        epochs[i, :, :end-start] = y[:, start:end]
        z = epochs[i, picks]
        ptp = np.ptp(z, axis=1)
        ptps.append(ptp)
        reasons, triggers = [], set()
        def flag(reason, names=()):
            reasons.append(reason)
            triggers.update(names)
        if not complete: flag('incomplete')
        if start < round(fs*cfg['filter']['startup_seconds']): flag('filter_startup')
        nonfinite = ~np.isfinite(x[picks, start:end]).all(axis=1)
        if nonfinite.any(): flag('raw_nonfinite', np.array(qc)[nonfinite])
        recover = contaminated[picks, start:end].any(axis=1)
        if recover.any(): flag('filter_recovery', np.array(qc)[recover])
        if complete and not np.isfinite(z).all(): flag('filtered_nonfinite', np.array(qc)[~np.isfinite(z).all(axis=1)])
        # Only windows completed by epoch availability can affect this decision.
        history = windows[windows.available_sample < stop]
        overlap = history[(history.start_sample < stop) & (history.stop_sample_exclusive > start)]
        flat = overlap[overlap.reason.str.contains('constant|zero|low_variation', regex=True) & overlap.channel.isin(qc)]
        if len(flat): flag('raw_flat', flat.channel.unique())
        # Persistent channel uncertainty is distinct from ordinary epoch rejection.
        # Keep all 16 inputs; this file is held for a channel policy decision.
        persistent = history[history.persistent_as_of_window & history.channel.isin(qc)].channel.unique()
        high, low = np.isfinite(ptp) & (ptp > hi), np.isfinite(ptp) & (ptp < lo)
        if high.any(): flag('ptp_high', np.array(qc)[high])
        if low.any(): flag('ptp_low', np.array(qc)[low])
        rows.append(dict(**row.to_dict(), valid_epoch=not reasons, rejection_reason=';'.join(reasons), trigger_channels=';'.join(sorted(triggers)),
                         epoch_qc_status='pending_channel_policy' if len(persistent) else ('pass' if not reasons else 'reject'),
                         max_ptp_uv=float(np.nanmax(ptp))*1e6 if np.isfinite(ptp).any() else np.nan,
                         min_ptp_uv=float(np.nanmin(ptp))*1e6 if np.isfinite(ptp).any() else np.nan,
                         complete_epoch=complete, available_sample=stop-1, decision_seconds=(start+fs*.8)/fs,
                         channel_policy_pending=bool(len(persistent)), persistent_channels_as_of_epoch=';'.join(persistent)))
    table = pd.DataFrame(rows)
    table['valid_epoch'] = table.valid_epoch.astype('boolean')
    table.loc[table.channel_policy_pending, 'valid_epoch'] = pd.NA
    return table, np.array(ptps)*1e6, epochs


def savefig(fig, path):
    fig.set_layout_engine('constrained')
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(fig)


def filter_figures(x, y, sos, fs, channels, out):
    f, before = signal.welch(x*1e6, fs, nperseg=min(x.shape[1], round(4*fs)), axis=-1)
    _, after = signal.welch(y*1e6, fs, nperseg=min(x.shape[1], round(4*fs)), axis=-1)
    fig, axes = plt.subplots(5, 4, figsize=(16, 16), sharex=True, sharey=True)
    for i, ax in enumerate(axes.flat):
        if i > 16: ax.axis('off'); continue
        p, q = (before[i], after[i]) if i < 16 else (before.mean(0), after.mean(0))
        ax.plot(f, 10*np.log10(np.maximum(p, 1e-30)), label='raw')
        ax.plot(f, 10*np.log10(np.maximum(q, 1e-30)), label='causal')
        ax.set(title=channels[i] if i < 16 else 'Channel mean power', xlim=(0, min(125, fs/2)), xlabel='Hz', ylabel='dB (uV^2/Hz)')
        ax.axvline(60, color='gray', linestyle=':'); ax.legend()
    savefig(fig, out/'psd.png')
    fig, ax=plt.subplots(figsize=(10,4))
    ax.plot(f,10*np.log10(np.maximum(before.mean(0),1e-30)),label='raw')
    ax.plot(f,10*np.log10(np.maximum(after.mean(0),1e-30)),label='causal')
    ax.set(xlim=(0,3),xlabel='Hz',ylabel='dB (uV^2/Hz)',title='Low frequency mean PSD (0.25 Hz resolution)'); ax.legend()
    savefig(fig,out/'psd_low_frequency.png')
    pd.DataFrame({'frequency_hz':np.tile(f, 16), 'channel':np.repeat(channels, len(f)), 'raw_uv2_hz':before.ravel(), 'causal_uv2_hz':after.ravel()}).to_csv(out/'psd.csv', index=False)
    for name, start in [('startup', 0), ('signal', min(round(30*fs), max(0, x.shape[1]-round(10*fs)))), ('transient', max(0, int(np.argmax(np.nanmax(np.abs(np.diff(x, axis=1)),axis=0)))-round(2*fs)))]:
        stop=min(x.shape[1], start+round(10*fs))
        fig, axes=plt.subplots(8, 2, figsize=(16, 16), sharex=True)
        for c, ax in enumerate(axes.flat):
            ax.plot(np.arange(start,stop)/fs, x[c,start:stop]*1e6, label='raw', alpha=.7)
            ax.plot(np.arange(start,stop)/fs, y[c,start:stop]*1e6, label='causal', alpha=.8)
            ax.set(title=channels[c], xlabel='seconds', ylabel='uV'); ax.legend()
        savefig(fig, out/f'waveform_{name}.png')
    freq=np.linspace(.01, fs/2-.01, 16000)
    _, h=signal.sosfreqz(sos, worN=freq, fs=fs)
    delay=-np.gradient(np.unwrap(np.angle(h)), 2*np.pi*freq)
    pd.DataFrame(dict(frequency_hz=freq, gain_db=20*np.log10(np.maximum(abs(h),1e-30)), group_delay_seconds=delay)).to_csv(out/'filter_response.csv',index=False)
    fig, axes=plt.subplots(2,1,figsize=(10,8))
    axes[0].plot(freq,20*np.log10(np.maximum(abs(h),1e-30))); axes[0].set(xlim=(0,125),ylim=(-120,5),ylabel='gain dB')
    axes[1].plot(freq,delay*1000); axes[1].set(xlim=(.5,30),ylabel='group delay ms',xlabel='Hz')
    savefig(fig,out/'filter_response.png')


def diagnostic_figures(epochs, table, ptp, channels, fs, out):
    fig, axes=plt.subplots(4,4,figsize=(16,12))
    for c,ax in enumerate(axes.flat):
        z=ptp[:,c]; z=z[np.isfinite(z)]
        ax.hist(z,bins=60); ax.set(title=channels[c],xlabel='PTP uV',ylabel='epochs')
    savefig(fig,out/'ptp_distributions.png')
    fig,axes=plt.subplots(4,4,figsize=(16,12),sharex=True,sharey=True)
    for c,ax in enumerate(axes.flat):
        for label,color in [('target','tab:red'),('non-target','tab:blue')]:
            for accepted,style in [(False,':'),(True,'-')]:
                mask=(table.label==label)&table.complete_epoch
                if accepted: mask &= table.valid_epoch.fillna(False)
                a=epochs[mask.to_numpy(),c]
                if len(a): ax.plot(np.arange(epochs.shape[-1])/fs*1000,np.nanmean(a,axis=0)*1e6,color=color,ls=style,label=f'{label} {"after" if accepted else "before"} n={len(a)}')
        ax.set(title=channels[c],xlabel='ms',ylabel='uV'); ax.legend(fontsize=5)
    savefig(fig,out/'erp_before_after.png')


def acquisition_header(path):
    # Read exact EDF per-signal prefilter strings, not just MNE's aggregated limits.
    with path.open('rb') as f:
        fixed=f.read(256); ns=int(fixed[252:256]); block=f.read(ns*256)
    labels=[block[i*16:(i+1)*16].decode('ascii',errors='replace').strip() for i in range(ns)]
    offset=ns*(16+80+8+8+8+8+8)
    return dict(zip(labels,[block[offset+i*80:offset+(i+1)*80].decode('ascii',errors='replace').strip() for i in range(ns)]))


def file_hash(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def process_edf(path, site, cfg, out):
    out.mkdir(parents=True,exist_ok=True)
    source_hash=file_hash(path)
    raw=read_bigp3bci_edf(path)
    channels=get_all_eeg_channels(raw); fs=float(raw.info['sfreq'])
    if set(channels)!=set(cfg['epoch_qc']['channels']):
        raise ValueError(f'Configured 16 channels differ from EDF: {channels}')
    x=raw.get_data(picks=channels)
    metadata=event_metadata(raw,path,site)
    sos=design_filter(fs,cfg['filter'])
    y, contaminated, state=causal_filter(x,sos,fs,cfg['filter'])
    assert y.shape == x.shape
    assert np.array_equal(stimulus_begin_events(raw)[:,0], metadata.original_event_sample.to_numpy())
    windows=continuous_quality(x,fs,channels,cfg['continuous_qc'])
    windows.to_csv(out/'continuous_windows.csv',index=False)
    summary=[]
    for ch, g in windows.groupby('channel',sort=False):
        summary.append(dict(channel=ch, flat_window_fraction=float(g.reason.str.contains('constant|zero|low_variation').mean()),
                            anomaly_window_fraction=float(g.reason.ne('').mean()), persistent_candidate=bool(g.persistent_as_of_window.any()),
                            reasons=';'.join(sorted({r for s in g.reason for r in s.split(';') if r}))))
    pd.DataFrame(summary).to_csv(out/'channel_quality.csv',index=False)
    comparisons=[]
    for index,bounds in enumerate(cfg['epoch_qc']['candidate_bounds_uv']):
        t,p,e=epoch_quality(x,y,contaminated,metadata,windows,channels,fs,cfg,bounds)
        comparisons.append(dict(candidate=index,low_uv=bounds[0],high_uv=bounds[1],n_total=len(t),n_valid=int(t.valid_epoch.sum()),n_pending=int(t.valid_epoch.isna().sum()),retention=float(t.valid_epoch.fillna(False).sum()/len(t))))
        if index==cfg['epoch_qc']['selected_candidate']: table,ptp,epochs=t,p,e
    pd.DataFrame(comparisons).to_csv(out/'threshold_comparison.csv',index=False)
    table['classification_eligible']=table.valid_epoch.fillna(False) & (not any(r['persistent_candidate'] for r in summary))
    table.to_csv(out/'epoch_metadata_quality.csv',index=False)
    for idx,bounds in enumerate(cfg['epoch_qc']['candidate_bounds_uv']):
        candidate,_,_=epoch_quality(x,y,contaminated,metadata,windows,channels,fs,cfg,bounds)
        candidate.groupby('label').valid_epoch.agg(['size','sum','mean']).to_csv(out/f'candidate_{idx}_label_retention.csv')
        candidate.trigger_channels.str.get_dummies(sep=';').sum().rename('count').to_csv(out/f'candidate_{idx}_channel_triggers.csv')
    pd.DataFrame(ptp,columns=cfg['epoch_qc']['channels']).rename_axis('event_index').to_csv(out/'epoch_ptp_uv.csv')
    trials=table.groupby('character_trial_id').agg(total_flashes=('valid_epoch','size'), valid_flashes=('valid_epoch','sum'))
    trials['pending_flashes']=table.groupby('character_trial_id').valid_epoch.apply(lambda a: int(a.isna().sum()))
    trials['evidence_insufficient']=trials.valid_flashes==0
    trials['scoring_policy']='pending; only zero valid flashes is flagged as insufficient'
    trials.to_csv(out/'character_trials.csv')
    table.groupby('label').valid_epoch.agg(['size','sum','mean']).to_csv(out/'label_retention.csv')
    reasons=table.rejection_reason.str.get_dummies(sep=';').sum().rename('count')
    reasons.to_csv(out/'rejection_counts.csv')
    table.trigger_channels.str.get_dummies(sep=';').sum().rename('count').to_csv(out/'channel_trigger_counts.csv')
    np.savez_compressed(out/'continuous_causal_eeg.npz',eeg_v=y,channels=channels,sfreq=fs,sos=sos,final_state=state)
    np.savez_compressed(out/'epochs_all.npz',eeg_v=epochs,event_index=table.event_index.to_numpy(),sfreq=fs,channels=channels,valid_epoch=table.valid_epoch.fillna(False).to_numpy(dtype=bool),validity_pending=table.valid_epoch.isna().to_numpy())
    assert file_hash(path)==source_hash, 'Source EDF changed during processing'
    record=dict(source_sha256=source_hash,config=cfg,source_edf=str(path.resolve()),sfreq=fs,n_samples=int(raw.n_times),channels=channels,
                acquisition_prefilter=acquisition_header(path),mne_acquisition_highpass=raw.info['highpass'],mne_acquisition_lowpass=raw.info['lowpass'],
                added_filter_sos=sos.tolist(),filter_design_basis='Butterworth: maximally flat passband; prototype order 4 (8th-order bandpass), second-order notch Q30 (~2Hz bandwidth); engineering candidates, not BCI2000 reproduction',
                initialization_basis='first-sample steady state reduces artificial DC step; separate EDF states; conservative 10s startup guard is a configurable development choice',
                initial_state='sosfilt_zi * first sample' if cfg['filter']['initialization']=='steady_first_sample' else 'zeros',
                startup_interval_samples=[0,round(fs*cfg['filter']['startup_seconds'])],startup_policy='overlapping epochs invalid',
                epoch_samples=epochs.shape[-1],epoch_interval='[onset, onset + ceil(0.8*sfreq))',baseline=None,
                event_samples_unchanged=True,signal_delay='frequency dependent; see filter_response.csv; events never shifted',
                threshold_selection='fixed development candidates; no distribution fitting; no validation/test/calibration threshold estimation',
                file_needs_channel_policy=any(r['persistent_candidate'] for r in summary),
                inference_ready=False,original_edf_modified=False,
                software=dict(numpy=np.__version__,pandas=pd.__version__,scipy=__import__('scipy').__version__,mne=__import__('mne').__version__))
    (out/'processing_record.json').write_text(json.dumps(record,indent=2),encoding='utf8')
    filter_figures(x,y,sos,fs,channels,out)
    diagnostic_figures(epochs,table,ptp,cfg['epoch_qc']['channels'],fs,out)
    sections=[f'<h1>{html.escape(path.stem)}</h1>', '<p>Development pilot; inferred site mapping. No final threshold selection. Baseline=None. Events unchanged; causal signal delay is frequency dependent. Pending channel policy prevents classification.</p>',
              pd.DataFrame(comparisons).to_html(index=False),'<h2>Rejection reasons</h2>',reasons.to_frame().to_html(),
              '<h2>Channels</h2>',pd.DataFrame(summary).to_html(index=False),'<h2>Character trials</h2>',trials.to_html()]
    for image_name in ['psd','psd_low_frequency','waveform_startup','waveform_signal','waveform_transient','filter_response','ptp_distributions','erp_before_after']:
        sections.append(f'<h2>{image_name}</h2><img style="max-width:100%" src="{image_name}.png">')
    (out/'report.html').write_text('<!doctype html><html><meta charset="utf-8"><title>EEG pilot</title><body>'+''.join(sections)+'</body></html>',encoding='utf8')
    raw.close()
    return table
