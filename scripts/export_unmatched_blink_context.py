"""Export unmatched old-detector candidates for every saved manual blink file."""
from pathlib import Path
import hashlib
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/unmatched_blink_context'


def covered(start, end, intervals):
    right = start
    for a, b in sorted(intervals):
        if a > right:
            break
        right = max(right, b)
        if right >= end:
            return True
    return False


def context_rows(candidates, manual, intervals, decisions):
    """One row per unmatched candidate and manual match of its predecessor."""
    previous = None
    rows = []
    for candidate in candidates.sort_values(['onset_seconds', 'blink_candidate_id']).itertuples():
        hits = manual[(manual.start_sec < candidate.offset_seconds) &
                      (manual.end_sec > candidate.onset_seconds)]
        if len(hits):
            previous = (candidate, hits)
            continue
        complete = covered(candidate.onset_seconds, candidate.offset_seconds, intervals)
        base = dict(blink_candidate_id=candidate.blink_candidate_id,
                    onset_seconds=candidate.onset_seconds, offset_seconds=candidate.offset_seconds,
                    peak_seconds=candidate.peak_seconds, filter_startup=bool(candidate.filter_startup),
                    fully_within_complete_review=complete,
                    interpretation='unmatched_in_complete_review' if complete else 'unmatched_outside_complete_review_unknown',
                    saved_candidate_review_decision=decisions.get(candidate.blink_candidate_id, ''),
                    previous_matched_candidate_id='', previous_candidate_onset_seconds=None,
                    previous_candidate_offset_seconds=None, gap_from_previous_candidate_end_seconds=None,
                    previous_match_manual_id='', previous_match_manual_start_seconds=None,
                    previous_match_manual_end_seconds=None,
                    nearest_manual_id='', nearest_manual_start_seconds=None, nearest_manual_end_seconds=None)
        # A separate nearest annotation helps when there is no previous matched candidate.
        if len(manual):
            distances = (manual.start_sec - candidate.offset_seconds).clip(lower=0) + (candidate.onset_seconds - manual.end_sec).clip(lower=0)
            near = manual.loc[distances.idxmin()]
            base.update(nearest_manual_id=near.manual_id, nearest_manual_start_seconds=near.start_sec,
                        nearest_manual_end_seconds=near.end_sec)
        if previous is None:
            rows.append(base)
        else:
            prior, annotations = previous
            base.update(previous_matched_candidate_id=prior.blink_candidate_id,
                        previous_candidate_onset_seconds=prior.onset_seconds,
                        previous_candidate_offset_seconds=prior.offset_seconds,
                        gap_from_previous_candidate_end_seconds=candidate.onset_seconds-prior.offset_seconds)
            for annotation in annotations.itertuples():
                rows.append(dict(base, previous_match_manual_id=annotation.manual_id,
                                 previous_match_manual_start_seconds=annotation.start_sec,
                                 previous_match_manual_end_seconds=annotation.end_sec))
    return rows


def main():
    config = json.loads((ROOT / 'configs/expanded_pilot.json').read_text())
    manual_root = ROOT / config['manual_validation']['manual_root']
    sources = sorted((manual_root / 'manual_review_tables').glob('*_manual_review.csv'))
    manifest = pd.read_csv(ROOT / 'outputs/blink_annotation_update/annotation_manifest.csv').set_index('edf_name')
    review = pd.read_csv(manual_root / 'reviewed_intervals.csv')
    frames, summary, provenance = [], [], []
    OUT.mkdir(parents=True, exist_ok=True)
    for source in sources:
        stem = source.name.removesuffix('_manual_review.csv')
        edf = stem + '.edf'
        candidate_path = ROOT / manifest.loc[edf, 'results_path'] / 'blink_candidates.csv'
        candidates = pd.read_csv(candidate_path)
        manual = pd.read_csv(source).sort_values(['start_sec', 'end_sec'])
        assert candidates.blink_candidate_id.is_unique and manual.manual_id.is_unique
        assert (manual.end_sec > manual.start_sec).all()
        assert manual.type.isin(config['manual_validation']['legacy_blink_types']).all()
        ranges = review[review.edf_name.eq(edf) & review.review_status.eq('complete')]
        intervals = list(zip(ranges.start_sec, ranges.end_sec))
        decisions = {}
        decision_path = ROOT / 'outputs/blink_candidate_review' / stem / 'review_decisions.csv'
        if decision_path.exists():
            table = pd.read_csv(decision_path, keep_default_na=False)
            decisions = dict(zip(table.blink_candidate_id, table.review_decision))
        rows = context_rows(candidates, manual, intervals, decisions)
        for row in rows:
            row.update(edf_name=edf, subject_id='_'.join(stem.split('_')[:2]))
        frame = pd.DataFrame(rows)
        assert len(frame), 'Keep an explicit empty schema if future files have zero unmatched candidates'
        cols = ['edf_name', 'subject_id'] + [c for c in frame.columns if c not in ['edf_name', 'subject_id']]
        frame = frame[cols]
        frame.to_csv(OUT / (stem + '_unmatched.csv'), index=False, encoding='utf-8-sig')
        unique = frame.drop_duplicates('blink_candidate_id')
        # Independent intersection matrix verifies the unmatched inventory.
        expected = {c.blink_candidate_id for c in candidates.itertuples()
                    if not ((manual.start_sec < c.offset_seconds) & (manual.end_sec > c.onset_seconds)).any()}
        assert set(unique.blink_candidate_id) == expected
        frames.append(frame)
        summary.append(dict(edf_name=edf, manual_spans=len(manual), candidates=len(candidates),
                            unmatched_candidates=len(unique), export_rows=len(frame),
                            unmatched_in_complete_review=int(unique.fully_within_complete_review.sum()),
                            nonstartup_unmatched_in_complete_review=int((unique.fully_within_complete_review & ~unique.filter_startup).sum()),
                            unmatched_without_previous_match=int(unique.previous_matched_candidate_id.eq('').sum())))
        provenance.append(dict(edf_name=edf, annotation_source=str(source.relative_to(ROOT)),
                               annotation_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                               candidate_source=str(candidate_path.relative_to(ROOT)),
                               candidate_sha256=hashlib.sha256(candidate_path.read_bytes()).hexdigest()))
    all_rows = pd.concat(frames, ignore_index=True)
    all_rows.to_csv(OUT / 'all_unmatched_candidates.csv', index=False, encoding='utf-8-sig')
    complete = all_rows[all_rows.fully_within_complete_review]
    complete.to_csv(OUT / 'complete_review_unmatched_candidates.csv', index=False, encoding='utf-8-sig')
    summary = pd.DataFrame(summary)
    summary.to_csv(OUT / 'file_summary.csv', index=False, encoding='utf-8-sig')
    payload = dict(summary=summary.to_dict('records'), rows=all_rows.astype(object).where(pd.notna(all_rows), None).to_dict('records'),
                   provenance=provenance,
                   definitions=dict(unmatched='No positive-duration interval overlap with any saved manual blink annotation; old_candidate_overlap_v1.',
                                    previous='Most recent matched candidate by onset in the same EDF, not the immediately previous unmatched candidate.',
                                    manual='previous_match_manual_* belongs to the previous matched candidate; nearest_manual_* is separate temporal context, never an asserted match.',
                                    rows='Multiple manual overlaps create multiple rows for one unmatched candidate. Count unique EDF/candidate IDs.',
                                    time='Seconds from EDF start; full precision retained. Startup candidates included and flagged.',
                                    coverage='Outside explicitly complete reviewed intervals, unmatched does not establish false positive. Negative candidate reviews do not establish complete coverage.'))
    (OUT / 'export_data.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(summary.to_string(index=False))
    print('Export rows:', len(all_rows))


if __name__ == '__main__':
    main()
