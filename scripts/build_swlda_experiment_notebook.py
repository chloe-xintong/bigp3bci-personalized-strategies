"""Build the executed SWLDA A/B/C development experiment report notebook."""
from pathlib import Path
import nbformat as nb
ROOT=Path(__file__).resolve().parents[1]
md=nb.v4.new_markdown_cell;code=nb.v4.new_code_cell
cells=[md('''# 14 · SWLDA：三种epoch拒绝策略的字符识别对照

本轮实际训练并验证33个SWLDA模型，使用现有12份Train/CBCol EDF、6名被试、11个被试×Session模型组。没有使用最终Test数据，没有训练PTP 250µV重复组，也没有加入8/12/18µV突出度试验。

A：仅基础故障与技术有效性检查。B：A＋16通道任一PTP>150µV拒绝。C：A＋**old_candidate_overlap_v1旧候选区间**重叠拒绝。C使用原25µV额区幅值及原配套触发条件，不使用新版主波或主波＋恢复区间。它是曾得到2105个额外区间关联epoch的旧版本；本轮结论仅适用于这一固定版本。

固定48个训练字符和24个验证字符。所有策略评价同一组24个验证字符；无输出仍计入准确率分母。结果属于小样本开发对照，不是独立最终评测。
'''),code('''from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
ROOT=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'configs/swlda_rejection_experiment.json').exists())
OUT=ROOT/'outputs/swlda_rejection_experiment_v1'
config=json.loads((OUT/'config_snapshot.json').read_text())
record=json.loads((OUT/'implementation_record.json').read_text())
split=pd.read_csv(OUT/'character_split.csv')
pred=pd.read_csv(OUT/'character_predictions.csv')
models=pd.read_csv(OUT/'model_summary.csv')
auc=pd.read_csv(OUT/'validation_auc.csv')
overall=pd.read_csv(OUT/'overall_character_summary.csv')
plt.rcParams.update({'figure.dpi':110,'axes.spines.top':False,'axes.spines.right':False})
display(pd.Series(record,name='implementation and scope').to_frame())
'''),md('''## 1. 先整理字符数量，再固定划分

每个被试×Session单独建立模型，不混合其他被试。Session内按EDF文件名字典序、character_trial_id顺序排列完整关闭的字符试验；前floor(2N/3)个用于训练，其余验证。通常4个训练＋2个验证；F05 SE002有两份EDF，共8个训练＋4个验证。

这里“完整”指有SelectedTarget结束边界、目标一致、全部flash已归属，不强制126。F20 SE002首字符实际108次、其余正常126次，该字符按观察结果保留在训练清单。文件名顺序是本轮固定的采集顺序代理；跨EDF未记录的现实暂停不被当作已知零时长。

划分在任何策略筛选和模型拟合之前保存。char_split文件重跑若改变会报错，三种策略复用它。没有为了补足拒绝后的epoch数量增加校准字符。
'''),code('''display(pd.read_csv(OUT/'character_inventory_and_split.csv'))
display(split[['model_id','edf_name','character_trial_id','set','flashes','canonical_126_flashes']])
print('训练字符：',split.set.eq('train').sum(),'验证字符：',split.set.eq('validation').sum())
'''),md('''## 2. 交叠epoch与策略确认时间检查

同一字符内的重叠epoch可以保留，但训练和验证不能共享原始样本。本轮集合边界处原始样本间隔均为755点，约2.949秒；C在完成额外确认后，距离第一条验证flash仍至少1.594秒。

连续因果滤波允许历史训练信号状态传入后续验证信号；没有未来验证信号回流训练。这里的无泄漏检查针对本轮划分和拟合流程；此前预处理开发曾浏览这些Train记录，因此不能把它称为完全盲测。
'''),code('''leak=pd.read_csv(OUT/'split_leakage_audit.csv')
display(leak)
assert not leak.shared_raw_samples.any()
assert not leak.training_confirmation_enters_validation.any()
assert leak.training_policy_to_first_validation_seconds.gt(0).all()
'''),md('''## 3. 字符与flash映射核查

直接读取原始EDF中36个字符状态通道，按通道名的行列位置建立候选字典。每次flash实际亮起哪些字符，直接由该时刻的字符通道状态决定。

CurrentTarget的1-based行优先索引在全部9054次flash上与实际亮起集合、StimulusType标签相容，冲突数为0。该核查用于确认映射和评价真值，不进入字符预测函数。预测只接收分类分数和亮起集合，不接收CurrentTarget或target标签。

StimulusCode保留为原始计数，绝不拿它当字符编号。观察到的随机字符集合不是假定的固定行列刺激组。
'''),code('''display(pd.read_csv(OUT/'mapping_audit.csv'))
display(pd.read_csv(OUT/'character_dictionary.csv'))
display(pd.read_csv(OUT/'stimulus_group_dictionary.csv').head(8))
'''),md('''## 4. 固定256维特征

16个EEG通道，0–800ms，baseline=None。每通道16个50ms窗求平均，按“通道优先、窗时间递增”拼接。NPZ信号单位V，特征转换为µV。

256Hz下epoch为205点，各窗按sample/fs落入[窗起点,窗终点)分配，因此每窗12或13点，不重复或丢弃采样点。第一个窗13点、最后一个窗13点；最终样本为796.875ms。没有搜索其他通道、时间窗或baseline设置。
'''),code('''display(pd.DataFrame(config['features'].items(),columns=['setting','value']))
display(pd.read_csv(OUT/'feature_sample_bins.csv').query("edf_name=='F_03_SE001_CBCol_Train01.edf'"))
display(pd.read_csv(OUT/'feature_dictionary.csv').head(20))
'''),md('''## 5. 真正的逐步SWLDA实现与训练内筛选

本项目Python实现经典前进／后退逐步OLS二分类回归：条件partial-F检验p<0.10进入，加入后检查系数F（t平方）检验p>0.15移除，最多60个特征。响应为non-target=0、target=1，线性输出越高表示更偏target；输出不是校准概率。

实现参考：[BCI2000 P300Classifier官方说明](https://www.bci2000.org/mediawiki/index.php/User_Reference:P300Classifier)。这是在本轮固定特征上的项目实现，**不是普通LDA或正则化LDA，也不宣称逐位复现BCI2000程序**。常数／共线特征做数值秩检查；没有岭惩罚。选完特征后以OLS生成对应线性权重，这是SWLDA流程的一部分。

每组只用本策略保留的训练epoch估计均值、标准差和选特征；验证数据不参与。三个方案有相同进入／移除条件与上限，最终特征可不同。每模型保存npz系数、缩放参数、选中特征表和JSON逐步进入/移除日志。33个模型全部converged，最终20–45个特征。
'''),code('''display(pd.Series(config['swlda'],name='fixed SWLDA settings').to_frame())
display(models[['model_id','policy','training_characters','validation_characters','training_epochs','n_selected','fit_status','fit_cpu_wall_seconds']])
print('逐模型训练参数核验：',OUT/'model_training_only_audit.csv')
'''),md('''## 6. 校准预算与epoch保留率

校准预算按采集的字符数固定，不按拒绝后相同epoch数匹配。表内acquired_calibration_seconds为EDF样本时钟上的采集区间时长之和，包含区间中的刺激等待；不声称包含未知的跨文件现实暂停。

A/B/C训练样本分别5766/5590/4469；验证样本分别3024/2797/2216。被拒绝的flash保留记录，但target_score为空，不填0。没有额外训练PTP 250µV组。
'''),code('''budgets=pd.read_csv(OUT/'calibration_and_retention.csv')
display(budgets[['model_id','policy','training_characters','validation_characters','acquired_calibration_seconds','training_collected_flashes','training_kept_epochs','validation_collected_flashes','validation_kept_epochs']])
r=pd.read_csv(OUT/'validation_retention.csv')
pooled=r.groupby('policy').agg(collected=('collected','sum'),kept=('kept','sum'))
pooled['retention']=pooled.kept/pooled.collected
display(pooled)
'''),md('''## 7. 拒绝后字符证据如何累计？

每个保留且有分数的flash，按真实亮起集合更新36个候选的证据。候选字符c的证据为：

**mean(score | c被点亮) − mean(score | c未被点亮)**。

分别按有效次数归一化，避免仅因某字符获得更多有效flash而总和更大。被拒绝flash不更新任一分子或分母。所有36个候选均至少需要1个有效点亮、1个有效未点亮证据；任一候选覆盖不足，或最高分并列，则字符无输出。该规则在拟合前固定，没有按真值放宽难字符。

首轮只在完整character trial结束并等全部质量确认到齐后预测，没有动态停止。CurrentTarget只用于最后计算correct，不参与打分、覆盖条件或候选选择。
'''),code('''display(pd.read_csv(OUT/'character_evidence.csv').head(12))
display(pd.read_csv(OUT/'stimulus_group_flash_counts.csv').head(12))
display(pred.loc[~pred.output_available,['policy','edf_name','character_trial_id','prediction_status','valid_flashes','min_active_evidence']])
'''),md('''## 8. 主要结果：无输出字符仍在分母中

同一24个验证字符：A=18/24，B=17/24，C=12/24。A/B均能输出24个；C只能输出20个，4个证据覆盖不足仍算不正确，不能从分母剔除。

这轮C平均每字符只剩92.3次有效flash，且F05部分字符有候选完全没有点亮证据。A/B/C完整预测与每个候选的证据都保存，便于检查无输出而不是猜一个字符。
'''),code('''display(overall)
fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
axes[0].bar(overall.policy,overall.character_accuracy*100,color=['tab:blue','tab:orange','tab:green'])
axes[0].set(ylim=(0,105),ylabel='character accuracy (%)',title='Same 24 validation characters; no output counts as incorrect')
for i,row in enumerate(overall.itertuples()):axes[0].text(i,row.character_accuracy*100+2,f'{row.correct_characters}/{row.validation_characters}',ha='center')
axes[1].bar(pooled.index,pooled.retention*100,color=['tab:blue','tab:orange','tab:green'])
axes[1].set(ylim=(0,105),ylabel='validation epoch retention (%)',title='Different evidence retained, same character denominator')
fig.savefig(OUT/'accuracy_and_retention.png',dpi=120);plt.show()
'''),md('''## 9. 被试与Session：不能只看总体

F05在A下8/8、B下7/8、C下2/8，C有4个无输出；F20在C下由3/4变为4/4。F06与F25基础表现较低，不能由F03/F05的高结果掩盖。

这些差异可能涉及信号质量、非平稳性、类别相关伪迹和证据损失等，当前实验不能判定具体来源。候选剔除改变准确率也不能证明保留下来的特征一定是纯P300。
'''),code('''subject=pd.read_csv(OUT/'subject_character_summary.csv')
display(subject)
display(pd.read_csv(OUT/'session_character_summary.csv'))
# The combined subject error-bar figure is shown in section 14.
'''),md('''## 10. 配对比较与小样本限制

A→B没有新增正确字符、损失1个；A→C新增正确1个、损失7个。表内精确配对binomial检验只描述不一致字符，样本很小，没有做策略再调参或据p值确定最终方案。

当前不能说差异已形成可靠总体规律；也不能因C总体下降就说所有眨眼处理都无效。该结论严格限定旧区间版本与当前完整字符末端评分。
'''),code('''display(pd.read_csv(OUT/'paired_strategy_summary.csv'))
paired=pd.read_csv(OUT/'paired_character_comparison.csv')
display(paired[['edf_name','character_key','true_character','predicted_character_A','predicted_character_B','predicted_character_C','correct_A','correct_B','correct_C','valid_flashes_A','valid_flashes_B','valid_flashes_C']])
'''),md('''## 11. Epoch ROC-AUC：各自有效与共同有效要分开

own_valid_scored_epochs使用各模型自己保留的验证epoch，评价对象不同；common_valid_scored_epochs_A_B_C只用同一被试×Session中三组共同有分数的验证事件，三组样本相同。

总体仅报告11个模型组AUC的宏平均，不把不同模型未校准分数直接拼起来算一个ROC。AUC为辅助指标，不代替字符准确率或无输出率。
'''),code('''display(auc)
macro=pd.read_csv(OUT/'overall_macro_auc.csv')
display(macro)
fig,ax=plt.subplots(figsize=(11,4),layout='constrained')
macro.pivot(index='policy',columns='scope',values='macro_auc').plot.bar(ax=ax,rot=0)
ax.set(ylim=(0,1),ylabel='macro ROC-AUC over model folds');ax.legend(fontsize=7)
fig.savefig(OUT/'auxiliary_auc.png',dpi=120);plt.show()
'''),md('''## 12. 实际可用时间：按记录时间戳取最大，不逐flash相加

epoch使用256Hz下205点，等待完整800ms数据；现有样本端点约定使表中数据就绪时刻最多多0.781ms。C还使用旧区间策略的统一保守确认窗口，约1.355秒，用于确定“没有稍后确认的候选会重叠”。

字符证据就绪时刻为max(SelectedTarget字符结束边界、该字符全部flash的policy_decision_seconds)，不把各flash等待逐个相加。本轮A/B在记录的字符结束时刻已具备数据；C平均再等1.34375秒（字符结束边界已经覆盖了一小段epoch等待）。

evidence_ready_seconds始终记录完成此次判定的样本时钟；无输出时prediction_available_seconds留空。时间是数据回放模拟，不包含未测量的设备通信和逐epoch计算时延；CPU训练耗时另存，不能冒称实机端到端延迟。
'''),code('''display(pred[['policy','edf_name','character_trial_id','output_available','last_flash_seconds','trial_end_seconds','evidence_ready_seconds','prediction_available_seconds','decision_latency_after_trial_end_seconds']].head(12))
display(overall[['policy','mean_decision_wait_after_trial_seconds','mean_output_wait_after_trial_seconds','no_output_characters']])
key=pred.character_key.iloc[0]
t=pred[pred.character_key.eq(key)].set_index('policy')
fig,ax=plt.subplots(figsize=(11,3),layout='constrained')
for i,policy in enumerate(['A','B','C']):
    row=t.loc[policy]
    ax.plot([row.last_flash_seconds,row.evidence_ready_seconds],[i,i],lw=3)
    ax.scatter([row.last_flash_seconds,row.trial_end_seconds,row.evidence_ready_seconds],[i,i,i],marker='o')
ax.set(yticks=[0,1,2],yticklabels=['A','B','C'],xlabel='EDF sample-clock seconds',title=f'{key}: last flash → full-character/QC evidence ready')
fig.savefig(OUT/'one_character_timing.png',dpi=120);plt.show()
'''),md('''## 13. 本轮结论与复现

在当前开发划分中：PTP150µV没有提高字符准确率，少了1个正确字符；旧眨眼区间拒绝总体下降，并增加无输出和约1.34秒的确认等待，收益没有抵消本轮证据损失。但F20有个别改善，样本24个字符也很少，因此不据此继续修改检测参数或断言普遍结论。

复现：`scripts/run_swlda_rejection_experiment.py`；固定配置：`configs/swlda_rejection_experiment.json`。保存划分、输入hash、映射、特征、33个模型、选中特征与步骤、逐epoch分数、逐字符预测、刺激集合/字符有效flash数、AUC与配对比较。

39项单元检查通过，包含SWLDA与独立直接OLS逐步参考的一致性、共线特征处理、整数采样窗、拒绝flash为缺失而非0、证据次数归一化及AUC。另核验33个模型的均值/标准差确实来自各自训练保留集，最终特征显著性和保存分数一致。

既有预处理开发与人工复核曾使用这些Train记录，且4份补充文件因有标注而选入；这不是独立最终测试。最终Test没有读取，策略也未自动启用或部署。
'''),code('''scores=pd.read_csv(OUT/'epoch_scores.csv',usecols=['policy','set','would_keep','target_score','has_score','character_key'])
assert scores.loc[~scores.would_keep,'target_score'].isna().all()
assert not scores.loc[~scores.would_keep,'has_score'].any()
assert pred.groupby('policy').size().eq(24).all()
assert pred.groupby('character_key').policy.nunique().eq(3).all()
assert pred.loc[~pred.output_available,'prediction_available_seconds'].isna().all()
assert not pred.loc[~pred.output_available,'correct'].any()
assert models.shape[0]==33 and models.fit_status.eq('converged').all()
assert models.n_selected.le(60).all()
assert record['feature_selection_training_only'] and not record['ordinary_or_regularized_lda_substitution']
assert not record['used_final_test_data']
assert config['blink_interval_version']=='old_candidate_overlap_v1'
print('PASS: same 24 validation characters, missing rejected scores, fixed C version, true train-only SWLDA, no final Test data')
''')]
cells += [md('''## 14. 每位被试的独立模型结果

六位被试合并为一张误差线图，左右分别展示字符准确率和验证epoch保留率。点为该被试各Session的等权均值，误差线为Session间样本标准差（±1 SD），不是置信区间；只有一个Session时仅显示点，标准差记为缺失。均值±SD可能超出0–100%，不截短误差线。此图的Session等权均值与下方按字符数汇总的准确率可能不同，尤其F05各Session验证字符数量不同。

以下按被试分别展示A/B/C，不重新训练或改变划分。模型按“被试×Session”独立训练；同一被试多个Session的结果汇总展示，不表示拟合了跨Session合并模型。

每个人依次展示：字符准确率、无输出数、epoch保留率和确认等待；各Session模型的训练预算、选中特征数与验证AUC；逐字符真值、预测及有效flash数。无输出计为错误。AUC在各Session分别计算，不拼接不同模型的分数。每Session仅有2–4个验证字符，不应将结果高低视为稳定个体差异。

数值另存为subject_model_results.csv和subject_session_model_results.csv。所选特征与权重见models目录下对应model_id和策略的文件。
'''),code('''from IPython.display import Markdown
subject_results=pd.read_csv(OUT/'subject_character_summary.csv')
retention=pd.read_csv(OUT/'validation_retention.csv')
subject_retention=retention.groupby(['subject_id','policy'],as_index=False)[['collected','kept']].sum()
subject_retention['epoch_retention']=subject_retention.kept/subject_retention.collected
subject_results=subject_results.merge(subject_retention,on=['subject_id','policy'],validate='one_to_one')
subject_results.to_csv(OUT/'subject_model_results.csv',index=False)
session_results=models.merge(pd.read_csv(OUT/'session_character_summary.csv'),on=['policy','subject_id','session_id','validation_characters'],validate='one_to_one')
session_results=session_results.merge(retention,on=['policy','subject_id','session_id'],validate='one_to_one')
for scope,label in [('own_valid_scored_epochs','own_auc'),('common_valid_scored_epochs_A_B_C','common_auc')]:
    session_results=session_results.merge(auc.loc[auc.scope.eq(scope),['model_id','policy','auc']].rename(columns={'auc':label}),on=['model_id','policy'],validate='one_to_one')
session_results.to_csv(OUT/'subject_session_model_results.csv',index=False)
# Session is the replicate; SD is undefined for a single Session.
subjects=sorted(subject_results.subject_id.unique())
fig,axes=plt.subplots(1,2,figsize=(13,4.5),layout='constrained')
error_rows=[]
for ax,metric,title in zip(axes,['character_accuracy','retention'],['Character accuracy','Validation epoch retention']):
    for j,policy in enumerate(['A','B','C']):
        stats=session_results.loc[session_results.policy.eq(policy)].groupby('subject_id')[metric].agg(['mean','std','count']).reindex(subjects)
        x=np.arange(len(subjects))+(j-1)*0.2
        ax.errorbar(x,stats['mean']*100,yerr=stats['std'].fillna(0)*100,fmt='o',capsize=4,label=policy,color=['tab:blue','tab:orange','tab:green'][j])
        for subject,row in stats.iterrows():
            error_rows.append(dict(subject_id=subject,policy=policy,metric=metric,session_mean=row['mean'],session_sd=row['std'],n_sessions=int(row['count'])))
    ax.set(xticks=np.arange(len(subjects)),xticklabels=[f'{sub}\\n(n={session_results.loc[session_results.subject_id.eq(sub),"session_id"].nunique()})' for sub in subjects],ylabel='%',title=title,ylim=(-40,140))
    ax.axhline(0,color='grey',lw=.5);ax.axhline(100,color='grey',lw=.5)
    ax.legend(title='Policy',ncol=3)
fig.suptitle('Session mean +/- sample SD; n = Sessions; single Session: point only')
fig.savefig(OUT/'subject_model_errorbars.png',dpi=150)
plt.show()
pd.DataFrame(error_rows).to_csv(OUT/'subject_model_errorbars.csv',index=False)
for subject in sorted(subject_results.subject_id.unique()):
    display(Markdown(f'### {subject}'))
    sub=subject_results.loc[subject_results.subject_id.eq(subject)].sort_values('policy')
    display(sub[['policy','correct_characters','validation_characters','character_accuracy','no_output_characters','kept','collected','epoch_retention','mean_decision_wait_after_trial_seconds']].round(4))

    display(Markdown('**各Session独立模型**：own_auc与common_auc分别对应各自有效epoch和三策略共同有效epoch。'))
    rows=session_results.loc[session_results.subject_id.eq(subject)].sort_values(['session_id','policy'])
    display(rows[['model_id','policy','training_characters','training_epochs','n_selected','fit_status','correct_characters','validation_characters','no_output_characters','retention','own_auc','common_auc']].round(4))
    display(Markdown('**逐字符结果**：predicted_character为空表示无输出；相同character_key对应三策略评价的同一个验证字符。'))
    display(pred.loc[pred.subject_id.eq(subject),['session_id','edf_name','character_trial_id','policy','true_character','predicted_character','prediction_status','correct','valid_flashes','collected_flashes','decision_latency_after_trial_end_seconds']].sort_values(['session_id','edf_name','character_trial_id','policy']))
assert len(session_results)==33
assert subject_results.groupby('policy').validation_characters.sum().eq(24).all()
assert subject_results.groupby('policy').correct_characters.sum().to_dict()==overall.set_index('policy').correct_characters.to_dict()
print('PASS: individual tables reconcile with all 33 models and the overall 24-character comparison')
''')]
n=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python (bigp3bci)','language':'python'}})
p=ROOT/'notebooks/14_bigp3bci_swlda_rejection_experiment.ipynb';nb.write(n,p);print(p)
