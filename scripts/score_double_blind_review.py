"""Compare locked independent forms; keep adjudication separate."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from build_double_blind_review import LABELS, ROOT


def agreement(a, b):
    table = pd.crosstab(pd.Categorical(a, categories=LABELS), pd.Categorical(b, categories=LABELS), dropna=False)
    counts = table.to_numpy(float)
    n = counts.sum()
    observed = float(np.trace(counts)/n)
    expected = float(np.dot(counts.sum(axis=0),counts.sum(axis=1))/n**2)
    return dict(n=int(n), agreement=observed, cohens_kappa=(observed-expected)/(1-expected) if expected < 1 else None), table


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--packet',type=Path,default=ROOT/'outputs/double_blind_review_v1')
    parser.add_argument('--phase',choices=['pilot_30','all'],default='pilot_30')
    args=parser.parse_args()
    key=pd.read_csv(args.packet/'coordinator_only/sampling_key.csv')
    forms=[]; reviewer_codes=[]
    for reviewer in ['A','B']:
        form=pd.read_csv(args.packet/f'reviewer_{reviewer}/review_form.csv',keep_default_na=False)
        if not form.blind_id.is_unique or set(form.blind_id)!=set(key.blind_id):
            raise ValueError('Review IDs changed, duplicated or missing')
        phase_lookup=key.set_index('blind_id').phase
        if not form.phase.eq(form.blind_id.map(phase_lookup)).all():
            raise ValueError('Phase assignment changed')
        form=form if args.phase=='all' else form[form.phase.eq('pilot_30')]
        if not form.label.isin(LABELS).all():
            raise ValueError(f'Reviewer {reviewer}: selected phase has blank or invalid labels; finish both independent forms first')
        codes=set(form.reviewer_code.str.strip())
        if len(codes)!=1 or '' in codes:raise ValueError('Use one nonempty reviewer_code per form')
        reviewer_codes.append(next(iter(codes)))
        forms.append(form[['blind_id','label','confidence','notes']].rename(columns={c:c+'_'+reviewer for c in ['label','confidence','notes']}))
    if reviewer_codes[0]==reviewer_codes[1]:raise ValueError('Two distinct reviewer codes are required')
    paired=forms[0].merge(forms[1],on='blind_id',validate='one_to_one').merge(key,on='blind_id',validate='one_to_one')
    result,table=agreement(paired.label_A,paired.label_B)
    result['scope']='Descriptive four-category agreement on the selected enriched sample, before adjudication.'
    result['phase']=args.phase
    result['weighted_label_proportions']={}
    if args.phase=='all':
        for reviewer in ['A','B']:
            result['weighted_label_proportions'][reviewer]={label:float(paired.loc[paired['label_'+reviewer].eq(label),'analysis_weight'].sum()/paired.analysis_weight.sum()) for label in LABELS}
    dest=args.packet/'coordinator_only'/('agreement_'+args.phase)
    if dest.exists():raise FileExistsError('Keep prior locked reports; choose a new packet copy to recompute')
    dest.mkdir()
    paired.to_csv(dest/'locked_independent_ratings.csv',index=False,encoding='utf-8-sig')
    table.to_csv(dest/'four_category_contingency.csv',encoding='utf-8-sig')
    conflicts=paired[paired.label_A.ne(paired.label_B)].copy()
    conflicts['adjudicated_label']='';conflicts['adjudicator_code']='';conflicts['adjudication_notes']=''
    conflicts.to_csv(dest/'disagreements_for_adjudication.csv',index=False,encoding='utf-8-sig')
    (dest/'agreement.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
