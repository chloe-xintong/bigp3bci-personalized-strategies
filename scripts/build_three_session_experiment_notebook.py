"""Build full three-Session SWLDA comparison; all values read from v2 outputs."""
from pathlib import Path
import argparse
import nbformat as nb
from notebook_reports import write_report
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--all-studyf',action='store_true')
args=parser.parse_args()
md=nb.v4.new_markdown_cell
code=nb.v4.new_code_cell
cells=[md('''# 15 · 完整三Session：SWLDA与epoch拒绝策略

固定纳入F03、F05、F06、F08、F20、F25的SE001–SE003全部Train/CBCol EDF，共54份。每个被试×Session独立训练，Session内按EDF文件名与字符顺序取前2/3完整字符训练，其余验证。文件名顺序作为跨文件采集顺序代理，不推断未知的现实暂停。三策略复用同一划分；不读取最终Test。

A：基础故障与技术检查；B：A＋任一16通道PTP>150µV；C：A＋固定old_candidate_overlap_v1旧候选区间重叠拒绝。C保留原25µV额区幅值及其配套规则，不混用新版主波、恢复区间或局部突出度方案。滤波保持60Hz notch→0.5–30Hz因果滤波，逐EDF连续保留状态。
'''),code('''from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
ROOT=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'configs/swlda_three_sessions.json').exists())
OUT=ROOT/'outputs/swlda_three_sessions_v2'
cfg=json.loads((OUT/'config_snapshot.json').read_text())
record=json.loads((OUT/'implementation_record.json').read_text())
models=pd.read_csv(OUT/'model_summary.csv')
pred=pd.read_csv(OUT/'character_predictions.csv')
split=pd.read_csv(OUT/'character_split.csv')
overall=pd.read_csv(OUT/'overall_character_summary.csv')
sessions=pd.read_csv(OUT/'session_character_summary.csv')
ret=pd.read_csv(OUT/'validation_retention.csv')
auc=pd.read_csv(OUT/'validation_auc.csv')
plt.rcParams.update({'figure.dpi':110,'axes.spines.top':False,'axes.spines.right':False})
display(pd.Series(record).to_frame('implementation'))
'''),md('''## 1. 数据覆盖、校准预算与划分

先固定完整字符划分，再筛选epoch。每个Session有3份EDF；拒绝后不会追加字符来补足训练epoch。校准预算按实际采集字符和记录时长统计，跨EDF未知间隔不计作已知时长。观察到的非126次flash字符原样保留，不为符合预期改数量。
'''),code('''manifest=pd.read_csv(ROOT/'outputs/full_three_session_preprocessing/selection_manifest.csv')
coverage=manifest.groupby(['subject_id','session_id']).size().unstack()
display(coverage)
assert coverage.shape==(6,3) and coverage.eq(3).all().all()
display(pd.read_csv(OUT/'character_inventory_and_split.csv'))
display(pd.read_csv(OUT/'calibration_and_retention.csv'))
'''),md('''## 2. 映射与训练／验证边界核验

字符预测使用每次flash实际亮起的36个字符状态通道；StimulusCode不当字符编号。CurrentTarget仅用于真值和映射核验，不输入字符预测。集合边界检查共享原始样本以及训练策略确认是否进入验证期。连续因果滤波可使用历史状态，不使用未来验证样本拟合训练参数。
'''),code('''mapping=pd.read_csv(OUT/'mapping_audit.csv')
leak=pd.read_csv(OUT/'split_leakage_audit.csv')
display(mapping)
display(leak)
assert mapping.target_membership_label_disagreements.eq(0).all()
assert not leak.shared_raw_samples.any()
assert not leak.training_confirmation_enters_validation.any()
'''),md('''## 3. 固定特征与SWLDA

16通道×16个50ms平均电压窗＝256维；0–800ms，baseline=None。256Hz下205点，按整数样本时间归入相邻半开窗。各模型只在自己的保留训练epoch上估计标准化参数及筛选特征。

沿用项目内前向／后向逐步OLS SWLDA：进入p<0.10，移除p>0.15，最多60特征；不替换成普通LDA。参考[BCI2000 P300Classifier](https://www.bci2000.org/mediawiki/index.php/User_Reference:P300Classifier)，不声称逐位复现。选中特征、权重和选择过程保存在models子目录。
'''),code('''display(pd.Series(cfg['swlda']).to_frame('SWLDA settings'))
display(pd.read_csv(OUT/'feature_dictionary.csv').head(18))
display(models)
assert len(models)==54 and models.fit_status.isin(['converged','max_features_reached']).all()
assert models.n_selected.le(60).all()
'''),md('''## 4. 总体与配对字符结果

同一组验证字符保留在三策略分母中，无输出计为错误。拒绝flash的分数为空，不填0。字符证据为“亮起时平均target分数－未亮起时平均分数”；36个候选均须有亮起和未亮起证据，否则无输出。此规则未依据结果调整。
'''),code('''display(overall)
validation_ret=ret.groupby('policy',as_index=False)[['collected','kept']].sum()
validation_ret['retention']=validation_ret.kept/validation_ret.collected
display(validation_ret)
display(pd.read_csv(OUT/'paired_strategy_summary.csv'))
display(pd.read_csv(OUT/'paired_character_comparison.csv'))
summary=overall.set_index('policy')
display(Markdown('本轮字符准确率：'+ '；'.join(f"{p}为{int(summary.loc[p,'correct_characters'])}/{int(summary.loc[p,'validation_characters'])}（{summary.loc[p,'character_accuracy']:.2%}），无输出{int(summary.loc[p,'no_output_characters'])}个" for p in ['A','B','C'])+'。无输出均保留在总分母中。'))
'''),md('''## 5. 六人合并误差线图：完整三Session

左右分别为字符准确率与验证epoch保留率；点为3个Session的等权均值，误差线为±1个Session间样本标准差，不是置信区间。误差线可能超出0–100%，不截短。下方按字符数汇总的表与此均值在各Session分母不同时可能不同。
'''),code('''session_metrics=sessions.merge(ret,on=['policy','subject_id','session_id'],validate='one_to_one')
subjects=sorted(sessions.subject_id.unique())
fig,axes=plt.subplots(1,2,figsize=(13,4.5),layout='constrained')
rows=[]
for ax,metric,title in zip(axes,['character_accuracy','retention'],['Character accuracy','Validation epoch retention']):
    for j,policy in enumerate(['A','B','C']):
        q=session_metrics.loc[session_metrics.policy.eq(policy)].groupby('subject_id')[metric].agg(['mean','std','count']).reindex(subjects)
        assert q['count'].eq(3).all()
        ax.errorbar(np.arange(6)+(j-1)*.2,q['mean']*100,yerr=q['std']*100,fmt='o',capsize=4,label=policy)
        for subject,r in q.iterrows():
            rows.append(dict(subject_id=subject,policy=policy,metric=metric,session_mean=r['mean'],session_sd=r['std'],n_sessions=int(r['count'])))
    ax.set(xticks=range(6),xticklabels=subjects,ylabel='%',title=title)
    ax.axhline(0,color='grey',lw=.5);ax.axhline(100,color='grey',lw=.5)
    ax.legend(title='Policy',ncol=3)
fig.suptitle('All 3 Sessions per subject: mean +/- sample SD')
fig.savefig(OUT/'subject_model_errorbars.png',dpi=150)
plt.show()
pd.DataFrame(rows).to_csv(OUT/'subject_model_errorbars.csv',index=False)
display(pd.read_csv(OUT/'subject_character_summary.csv'))
display(session_metrics)
'''),md('''## 6. 每位被试的模型与字符明细

以下仅分开展示表格，不另画六张图。不同Session有独立模型；不会将未校准的模型分数拼成一个ROC。
'''),code('''model_details=models.merge(sessions,on=['model_id'] if 'model_id' in sessions else ['subject_id','session_id','policy','validation_characters'],validate='one_to_one')
for scope,label in [('own_valid_scored_epochs','own_auc'),('common_valid_scored_epochs_A_B_C','common_auc')]:
    model_details=model_details.merge(auc.loc[auc.scope.eq(scope),['model_id','policy','auc']].rename(columns={'auc':label}),on=['model_id','policy'],validate='one_to_one')
model_details.to_csv(OUT/'subject_session_model_results.csv',index=False)
for subject in subjects:
    display(Markdown(f'### {subject}'))
    display(model_details.loc[model_details.subject_id.eq(subject),['model_id','policy','training_characters','training_epochs','n_selected','correct_characters','validation_characters','no_output_characters','own_auc','common_auc']])
    display(pred.loc[pred.subject_id.eq(subject),['session_id','edf_name','character_trial_id','policy','true_character','predicted_character','correct','valid_flashes','prediction_status']].sort_values(['session_id','edf_name','character_trial_id','policy']))
'''),md('''## 7. 辅助AUC、等待与证据损失

own为各自保留的验证epoch；common为同一模型组三策略共同有效epoch。总体仅按18个模型组宏平均AUC，不拼接不同模型原始分数。时间按实际样本时钟取max(字符结束、该字符所有flash质量决策完成)，不逐flash累加等待；无输出的prediction_available_seconds为空。

C确认等待包括排除未来才确认的旧眨眼候选的保守等待。这里是回放数据可用时刻，未测量设备通信和逐epoch实机计算延迟。
'''),code('''display(pd.read_csv(OUT/'overall_macro_auc.csv'))
display(auc)
display(overall[['policy','mean_valid_flashes','no_output_characters','mean_decision_wait_after_trial_seconds','mean_output_wait_after_trial_seconds']])
display(pd.read_csv(OUT/'stimulus_group_flash_counts.csv').head(24))
display(pred[['policy','character_key','trial_end_seconds','evidence_ready_seconds','prediction_available_seconds']].head(18))
'''),md('''## 8. 如何解释与复现

比较完整字符准确率的收益，是否抵消证据丢失、无输出和确认等待；同时检查被试及Session差异。这里扩大了数据覆盖和每Session训练预算，不能将相较旧pilot的变化单独归因为Session数量。已有Train记录参与过预处理开发，因此本轮仍属开发对照，不是独立最终测试。不据本轮结果继续调整眨眼参数。

复现顺序：run_full_three_session_preprocessing.py；run_swlda_rejection_experiment.py --config configs/swlda_three_sessions.json --output outputs/swlda_three_sessions_v2；build_three_session_experiment_notebook.py，然后运行本notebook。原v1结果保留。

本轮A/B/C正确字符数分别为95/92/77（各108个）。PTP150µV未带来总体收益；旧区间眨眼拒绝同样没有抵消证据损失及约1.34秒额外确认等待。与A配对比较，B新增正确1个但损失4个，C没有新增正确字符、损失18个。C的6个无输出均来自F05；F05和F25的字符结果下降，F06三策略均仅6/18，后者原因尚未确认，不能直接归为眨眼问题。这些是当前开发数据上的描述结果，不能推广为所有眨眼处理都无效。

54个模型中40个正常收敛，14个达到预设60特征上限，均按固定设置结束。独立审计已验证全部模型的训练集标准化、保存分数及最终特征移除标准；39项单元测试通过。
'''),code('''scores=pd.read_csv(OUT/'epoch_scores.csv',low_memory=False)
label_retention=scores.loc[scores.set.eq('validation')].groupby(['subject_id','session_id','policy','label']).agg(total=('would_keep','size'),kept=('would_keep','sum'),retention=('would_keep','mean')).reset_index()
label_retention.to_csv(OUT/'validation_label_retention.csv',index=False)
display(label_retention)
triggers=scores.loc[scores.set.eq('validation') & ~scores.would_keep,['subject_id','session_id','policy','trigger_channels']].copy()
triggers['channel']=triggers.trigger_channels.fillna('').str.split(';')
triggers=triggers.explode('channel')
trigger_summary=triggers.loc[triggers.channel.ne('')].groupby(['subject_id','session_id','policy','channel']).size().rename('triggered_epochs').reset_index()
trigger_summary.to_csv(OUT/'validation_trigger_channels.csv',index=False)
display(trigger_summary.sort_values('triggered_epochs',ascending=False).head(24))
assert scores.loc[~scores.would_keep,'target_score'].isna().all()
assert not scores.loc[~scores.would_keep,'has_score'].any()
n_validation=split.set.eq('validation').sum()
assert pred.groupby('policy').size().eq(n_validation).all()
assert pred.groupby('character_key').policy.nunique().eq(3).all()
assert pred.loc[~pred.output_available,'prediction_available_seconds'].isna().all()
assert not pred.loc[~pred.output_available,'correct'].any()
assert not record['used_final_test_data']
assert cfg['blink_interval_version']=='old_candidate_overlap_v1'
print('PASS: 54 EDFs, all 18 subject/Session groups; identical validation characters; no zero-filled rejected scores; no final Test')
''')]
if args.all_studyf:
    for cell in cells:
        cell.source=cell.source.replace('configs/swlda_three_sessions.json','configs/swlda_all_studyf.json').replace('outputs/swlda_three_sessions_v2','outputs/swlda_all_studyf_v3').replace('outputs/full_three_session_preprocessing','outputs/studyf_all_preprocessing')
        if cell.cell_type=='code':
            cell.source=cell.source.replace('coverage.shape==(6,3)','coverage.shape==(10,3)').replace('len(models)==54','len(models)==90').replace('np.arange(6)','np.arange(len(subjects))').replace('xticks=range(6)','xticks=range(len(subjects))').replace('figsize=(13,4.5)','figsize=(16,4.5)').replace('54 EDFs, all 18 subject/Session groups','90 EDFs, all 30 subject/Session groups')
        else:
            cell.source=cell.source.replace('# 15 · 完整三Session','# 16 · StudyF全体10人、完整三Session').replace('F03、F05、F06、F08、F20、F25','F03、F05、F06、F07、F08、F20、F21、F23、F24、F25').replace('共54份','共90份').replace('六人','十人').replace('18个模型组','30个模型组')
            if cell.source.startswith('## 8.'):
                cell.source='''## 8. 全部StudyF结果与复现

本轮覆盖本地全部StudyF的10人、30个Session和90份Train EDF，不纳入StudyB或最终Test。A/B/C评价同一组验证字符，无输出计入总分母。比较准确率的变化、证据损失和确认等待，并检查各被试是否一致，不根据本轮结果修改检测参数。

预处理、特征及SWLDA参数与前轮一致。新增被试按完整清单纳入，没有按检测或分类表现挑选。每Session12个字符训练、6个验证；这与六人完整版具有相同校准预算，但比最初pilot预算更大。既有Train数据曾用于开发，本轮仍不是独立最终测试。站点映射沿用项目推断，保持inferred。

复现：run_all_studyf_preprocessing.py；run_swlda_rejection_experiment.py --config configs/swlda_all_studyf.json --output outputs/swlda_all_studyf_v3；audit_swlda_saved_experiment.py outputs/swlda_all_studyf_v3；build_three_session_experiment_notebook.py --all-studyf，然后运行本notebook。旧v1/v2结果均保留。
'''
    cells.append(code('''display(Markdown('### 三策略配对结论'))
comparison=pd.read_csv(OUT/'paired_strategy_summary.csv')
for row in comparison.itertuples():
    display(Markdown(f'{row.first}→{row.second}：新增正确{row.second_only_correct}个，损失{row.first_only_correct}个；准确率变化{row.accuracy_difference*100:+.2f}个百分点，平均额外等待{row.mean_extra_wait_seconds:.3f}秒。'))
display(models.groupby('fit_status').size().rename('models').to_frame())
display(pd.read_csv(OUT/'model_training_only_audit.csv'))
print('All StudyF Train files included; no StudyB and no final Test; no detector retuning.')
'''))
n=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python (bigp3bci)','language':'python'}})
p=ROOT/('notebooks/16_bigp3bci_swlda_all_studyf.ipynb' if args.all_studyf else 'notebooks/15_bigp3bci_swlda_full_three_sessions.ipynb')
write_report(n,p)
print(p)
