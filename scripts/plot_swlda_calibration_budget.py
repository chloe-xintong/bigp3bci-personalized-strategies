"""Redraw notebook 11 figures from saved CSVs; no model is refitted."""
from pathlib import Path
import argparse
import json
import sys
import pandas as pd
import matplotlib
matplotlib.use('Agg')

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from bigp3bci_budget_plot import plot_budget_results,plot_budget_small_multiples


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--experiment',default='outputs/swlda_calibration_budget_train_validation_v1')
    out=ROOT/parser.parse_args().experiment;k_values=json.loads((out/'config_snapshot.json').read_text())['k_values']
    overall=pd.read_csv(out/'overall_results.csv');summary=pd.read_csv(out/'subject_results.csv')
    plot_budget_results(overall,out/'calibration_curves.png',k_values,metric='macro_flash_roc_auc')
    plot_budget_results(overall,out/'character_accuracy_curves.png',k_values,metric='mean_subject_accuracy')
    plot_budget_small_multiples(overall,summary,out/'auc_small_multiples.png',k_values)
    plot_budget_small_multiples(overall,summary,out/'character_accuracy_small_multiples.png',k_values,metric='mean_subject_accuracy')
    loss=out/'weighted_logloss'
    if (loss/'completion.json').exists():
        loss_overall=pd.read_csv(loss/'overall_metrics.csv')
        subjects=summary.merge(pd.read_csv(loss/'subject_metrics.csv'),on=['subject_id','scheme','k'],validate='one_to_one')
        plot_budget_results(loss_overall,loss/'weighted_logloss_curves.png',k_values,metric='macro_weighted_log_loss')
        plot_budget_small_multiples(loss_overall,subjects,loss/'logloss_small_multiples.png',k_values,metric='macro_weighted_log_loss')
        plot_budget_small_multiples(loss_overall,subjects,loss/'logloss_small_multiples_zoom.png',k_values,metric='macro_weighted_log_loss',zoom=True)
    print('Redrew figures in',out)


if __name__=='__main__':main()
