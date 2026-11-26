"""Build the frozen-trigger boundary localization / recovery diagnostic notebook."""
from pathlib import Path
import textwrap
import nbformat as nb
from notebook_reports import write_report
ROOT=Path(__file__).resolve().parents[1]
cells=[]
def md(s):cells.append(nb.v4.new_markdown_cell(textwrap.dedent(s).strip()))
def code(s):cells.append(nb.v4.new_code_cell(textwrap.dedent(s).strip()))
md('''
# 08 · 冻结触发条件，分离主波边界、恢复区间与判定时间

旧结果完整保留；旧595个候选ID及触发参数不变。本轮**只新增定位与恢复的工程候选规则**，不重新调20/25µV触发条件，也不对旧人工边界统一减114/225ms。

主波：在独立原始域因果分支上定位。恢复：单独检查正式0.5–30Hz信号。主波结束不表示信号恢复可用，恢复安静条件满足也不证明信号干净。

原始EDF、11号交互窗口、旧自动标注和人工复核文件不修改。所有新产物位于outputs/blink_boundary_localization_v1。
''')
code('''
from pathlib import Path
import json,sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display,Image
ROOT=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'configs/blink_boundary_localization.json').exists())
OUT=ROOT/'outputs/blink_boundary_localization_v1'
config=json.loads((OUT/'config_snapshot.json').read_text())
record=json.loads((OUT/'run_record.json').read_text())
b=pd.read_csv(OUT/'all_boundaries.csv')
audit=pd.read_csv(OUT/'localization_audit.csv')
summary=pd.read_csv(OUT/'overall_summary.csv')
display(pd.Series(record).drop('config').to_frame('run record'))
''')
md('''
## 1. 什么保持不变，什么新增？

旧触发器使用正式分析信号之后的5Hz因果低通：共同额区进入/退出20/10µV、额区幅值≥25µV，以及原有共同变化、持续时间、间距和合并条件，全部保持不变。

新定位分支：**原始EEG → 60Hz notch Q30 → 二阶10Hz因果低通，无新增高通**。它独立于正式分析和旧触发分支，减少再用同一高通反向恢复波定义主波边界的问题。采集端原有滤波已经在原始EDF中，新增分支不能还原采集前的波形，也不能消除所有时序差异。

新分支仍有自身因果延迟，因此新边界不等同于生理真值。后面提供相同分支上的人工标注入口；旧原始域人工标注只作并列参考。
''')
code("display(pd.Series(config,name='provisional localization settings').to_frame())")
md('''
## 2. 主波定位：历史回溯不等于提前输出

旧触发点之前0.9–0.55秒的历史估计局部基线；在触发前0.4秒至后0.3秒范围内选主要额区偏离峰，允许正负极性。相对该峰回溯最多0.7秒寻找安静后开始偏离的起点；峰后寻找主波回到基线附近的时刻。边界阈值为峰幅15%、下限5µV，安静确认分别0.04/0.06秒。

这些是待验证定位参数，不是新的检测门槛。为检查相邻同极性主波，还查看峰后至0.8秒的局部波形，相关观察时间包含在main_decision_seconds中。回溯起点仅描述波形位置，绝不替代判定时间。

参考区不足、峰在搜索边缘、原始域波形太弱、主波端点找不到、持续时间异常、疑似相邻多波峰或非有限恢复影响时，都标记不确定。没有因为旧检测成功就强行宣布边界可信。
''')
code('''
fields=['edf_name','blink_candidate_id','trigger_crossing_seconds','old_onset_seconds','old_offset_seconds',
        'main_onset_seconds','main_peak_seconds','main_offset_seconds','main_decision_seconds',
        'recovery_onset_seconds','recovery_offset_seconds','decision_seconds','decision_available','actual_decision_seconds',
        'localization_usable_for_provisional_comparison','localization_flags']
display(b[fields].head(15))
''')
md('''
## 3. 恢复区间：在正式分析信号上单独判断

从主波结束起，在正式0.5–30Hz额区共同波形上观察。相对局部历史基线的偏离降到max(10µV,正式主波峰值20%)以内，并持续0.12秒，才标记quiet_criterion_met_unvalidated。最大观察1.5秒；没满足则标not_confirmed_within_horizon，文件结束不足则另标。

这只是一套有限时间、额区范围内的安静指标，**不证明16通道恢复可用**。反向波、尾波、真实脑电与眼动影响无法靠此规则完全区分。保存全部16通道在恢复区间的PTP，供后续核查；绝不因主波结束就把尾波自动视为正常。

恢复区间末端是安静窗口开始，decision仍需等待安静窗口结束及其他所需观察。若判定所需数据尚未到齐，decision_available=False、actual_decision_seconds留空；预计所需时间不能被称为已实际完成判断。
''')
code('''
display(b.groupby('recovery_status').size().rename('candidates').to_frame())
display(b[['blink_candidate_id','recovery_duration_seconds','recovery_criterion_uv','recovery_channels','recovery_ptp_by_channel_uv']].head(8))
assert not b.recovery_proves_clean_signal.any()
''')
md('''
## 4. 不确定定位与旧区间回退

595个触发中，428个通过本轮定位的内部形态检查，167个标记不确定；“通过内部检查”不代表已被人工验证准确。

用于epoch损失对照时，不确定候选**回退旧区间**，而不是因找不到边界而把该候选删除。原始main/recovery估计也保留作诊断，图上不确定主波用红色，不能误当作已采纳的橙色主波。

相邻主波、多触发定位到接近同一峰、主波起点相对旧候选次序倒置、恢复区间与下一主波相交，分别报告。跨候选审计为回顾性诊断，不参与实时边界或拒绝判定，不静默合并/拆分旧ID。
''')
code('''
display(audit)
flags=b.localization_flags.fillna('').str.get_dummies(sep=';').sum().sort_values(ascending=False)
display(flags.rename('candidates').to_frame())
display(pd.read_csv(OUT/'possible_merged_lobes.csv').head(15))
''')
md('''
## 5. 检查是否把主波切得太短

下表将同一父候选的新主波时长与旧区间时长并列。仅对内部检查通过的定位统计，并列出最短的若干例，不把“更短”当作成功标准。

持续时间至少0.08秒只是工程约束，不能证明0.09秒就足够；短主波、多个真实眨眼相近、局部基线受前次眨眼污染等情形，仍需要在同一分支上人工核对。
''')
code('''
good=b[b.localization_usable_for_provisional_comparison].copy()
good['old_duration_seconds']=good.old_offset_seconds-good.old_onset_seconds
display(good.groupby('edf_name')[['old_duration_seconds','main_duration_seconds','recovery_duration_seconds']].agg(['count','median','min','max']))
display(good.sort_values('main_duration_seconds')[['edf_name','blink_candidate_id','main_onset_seconds','main_offset_seconds','main_duration_seconds']].head(10))
fig,ax=plt.subplots(figsize=(9,4),layout='constrained')
ax.hist(good.old_duration_seconds,bins=30,alpha=.5,label='old')
ax.hist(good.main_duration_seconds,bins=30,alpha=.5,label='main provisional')
ax.set(xlabel='duration (s)',ylabel='candidates');ax.legend();plt.show()
''')
md('''
## 6. 同时看四个信号域，不做统一时间平移

下面固定展示两份完整复核文件的首个非启动候选、首个形态警告，以及此前检查的F05 B0010反向恢复疑似误报。

每图依次为原始EEG、新定位分支、正式0.5–30Hz、旧5Hz触发分支。同一主波/恢复边界叠加到四域便于辨别相位与形态影响，但**应在新定位分支上判断新主波边界是否合理，在正式分析信号上判断尾波是否仍有影响**。

绿色是历史原始域人工标注；灰色旧区间、橙色暂定主波、紫色恢复、红色不确定主波。虚线为判定时间，不是波形起点。图中有些判定晚于主波很多，这是实际用到后续观察信息的结果，不往前伪装时间。
''')
code('''
examples=pd.read_csv(OUT/'plotted_examples.csv')
display(examples)
for name in examples.figure:display(Image(filename=str(OUT/name),width=1200))
''')
md('''
## 7. 新旧epoch重叠量：不是污染epoch真值

所有策略共用先前faults_only的共同基础条件，12份训练EDF的9054条事件全部保存。比较旧候选区间、仅主波、主波+恢复；定位不确定者均按前述规则回退旧区间。

旧的2105条只能称为“通过基础条件、与旧候选区间重叠的epoch”。下面的新数量也只是候选区间重叠，不表示这些epoch必须拒绝。无任何真实删除、重参考、ICA或模型训练。
''')
code('''
display(summary)
display(pd.read_csv(OUT/'epoch_set_changes.csv'))
perfile=pd.read_csv(OUT/'edf_summary.csv')
display(perfile.pivot(index='edf_name',columns='scheme',values='extra_interval_rejections'))
''')
md('''
**本轮结果：**旧区间标记2105条；仅主波（不确定回退）1847条；主波+恢复（不确定回退）2401条。主波方案净减少258条，但因为回溯起点向前移动，新旧集合不是严格包含关系，需看newly_kept/newly_flagged两列。

加入恢复后比旧方案多标记296条，说明把主波缩短不能自动推导信号更早可用。这是固定恢复工程条件下的结果，不应按保留率反向选择参数。
''')
md('''
## 8. 类别、字符与刺激组

标签仅用于分组统计，不参与触发或定位。每字符有效flash仍只是对应方案下预计保留数，未定义字符评分/动态停止。刺激组沿用实际字符通道集合签名，不猜行列语义。
''')
code('''
display(pd.read_csv(OUT/'subject_summary.csv'))
labels=pd.read_csv(OUT/'label_summary.csv')
display(labels[labels.edf_name.eq('F_05_SE003_CBCol_Train04.edf')])
chars=pd.read_csv(OUT/'character_summary.csv')
display(chars[chars.edf_name.eq('F_05_SE003_CBCol_Train04.edf')])
print('完整刺激集合计数：',OUT/'stimulus_group_summary.csv')
''')
md('''
## 9. 在线可用时间与待定

main_decision_seconds包含旧检测确认、峰搜索、主波结束确认及峰后多波审计所需观察；decision_seconds再包含恢复观察。actual_decision_seconds仅在数据真正到齐时提供，不能用回溯main_onset_seconds替代。

对“没有候选重叠”也需要保守等待：因为稍后的触发可回溯到当前epoch。新对照方案采用固定3.824秒左右的额外确认窗口，覆盖最大回溯与前向观察。它只是避免偷看未来的保守对照，**不是推荐在线等待时间**。本轮尚未优化增量调度/最早判定，也没有训练分类器；EOF待定另存，不假装已知无眨眼。
''')
code('''
display(b[['blink_candidate_id','trigger_crossing_seconds','main_onset_seconds','main_decision_seconds','decision_seconds','decision_available','actual_decision_seconds']].head(12))
display(audit[['edf_name','new_policy_conservative_extra_wait_seconds']])
display(summary[['scheme','pending_at_eof']])
''')
md('''
## 10. 旧人工边界不能直接用作新分支定位真值

下面只沿用旧父候选与人工的相交配对，列出旧/新边界与历史人工边界之差；不通过重新匹配来提高表面成绩，不做平均偏移校正。

三者的信号域明确分开：manual=原始EDF，old=正式+5Hz，new=原始+10Hz。这些delta仅描述，不报告为定位误差或证明新边界准确。要得到可比真值，请在下一节选择localization模式，独立标注所见主波边界，并记录信号域；正式分析尾波需在analysis模式另核查。
''')
code('''
manual=pd.read_csv(OUT/'legacy_manual_domain_comparison.csv')
display(manual.head(12))
''')
md('''
## 11. 独立交互核查窗口（不影响11号复核）

设置文件、mode和起点后，将OPEN_BOUNDARY_BROWSER=True再运行。mode可为raw、localization、analysis、auxiliary。灰色旧、橙色暂定主波、紫色恢复、红色定位不确定；历史人工为绿色。

如果要标可比主波边界：选择localization，按a新建BOUNDARY_MANUAL_main；恢复可在analysis上标BOUNDARY_MANUAL_recovery，不确定用BOUNDARY_MANUAL_uncertain。保存单元单独写新目录，记录mode，不覆盖11中的多检决定或旧人工标注。没有标记会明确报错而不存空文件。这里不要用BAD表示主波边界，BAD仍是11号窗口的“多检”语义。
''')
code('''
if str(ROOT/'src') not in sys.path:sys.path.insert(0,str(ROOT/'src'))
from bigp3bci_boundary_review import open_boundary_review,save_domain_manual_boundaries
EDF='F_05_SE003_CBCol_Train04.edf'
BOUNDARY_MODE='localization'
BOUNDARY_START_SECONDS=10.5
OPEN_BOUNDARY_BROWSER=False
if OPEN_BOUNDARY_BROWSER:
    get_ipython().run_line_magic('matplotlib','qt')
    import mne
    mne.viz.set_browser_backend('matplotlib')
    boundary_raw,boundary_browser,boundary_table=open_boundary_review(ROOT,EDF,BOUNDARY_MODE,BOUNDARY_START_SECONDS)
else:
    print('需要时将OPEN_BOUNDARY_BROWSER设为True；无需关闭11号窗口或重新初始化其面板。')
''')
code('''
SAVE_DOMAIN_BOUNDARIES=False
if SAVE_DOMAIN_BOUNDARIES:
    print(save_domain_manual_boundaries(ROOT,boundary_raw))
''')
md('''
## 12. 一致性检查与后续证据

23项回归测试通过，新增检查历史回溯、未来数据不能改变已完成边界、正负极性、相邻峰只标记不静默合并、定位失败回退旧区间。源EDF及旧候选表hash前后一致；旧2105参考精确重现。

仍待验证：局部基线是否被前次眨眼影响、主波是否切短、真实相邻眨眼与反向波如何区分、恢复区间是否需要拒绝、分类效果以及同域人工真值。当前方法只是可复核的第一版定位，不能宣布完成伪迹去除。

运行：scripts/refine_blink_boundaries.py；配置：configs/blink_boundary_localization.json。所有时间端点均按样本半开区间保存；重叠时长按并集计算，不重复累计。
''')
code('''
assert record['frozen_trigger_ids_preserved'] and record['old_candidate_files_unchanged']
assert record['old_policy_record_unchanged'] and not record['time_shift_correction']
assert len(b)==595 and b.blink_candidate_id.is_unique
assert (b.main_decision_seconds>=b.frozen_detector_decision_seconds).all()
assert (b.decision_seconds>=b.main_decision_seconds).all()
assert (b.recovery_onset_sample==b.main_offset_sample_exclusive).all()
assert (b.recovery_offset_sample_exclusive>=b.recovery_onset_sample).all()
assert b.loc[~b.decision_available,'actual_decision_seconds'].isna().all()
assert int(summary.loc[summary.scheme.eq('old_candidate_overlap'),'extra_interval_rejections'].iloc[0])==2105
assert summary.total.eq(9054).all()
print('PASS: frozen triggers and old results preserved; domains and decision availability explicit; old overlap count reproduced')
''')
n=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python (bigp3bci)','language':'python'}})
p=ROOT/'notebooks/08_bigp3bci_blink_main_recovery_localization.ipynb';print(write_report(n,p))
