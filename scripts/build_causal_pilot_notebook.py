"""Build the explanatory notebook from saved causal pilot outputs."""
from pathlib import Path
import textwrap
import nbformat as nb
from notebook_reports import write_reportf
ROOT = Path(__file__).resolve().parents[1]
cells=[]
def md(s): cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(s).strip()))
def code(s): cells.append(nbf.v4.new_code_cell(textwrap.dedent(s).strip()))
md('''
# 05 · 因果滤波与自动 epoch QC：pilot 结果逐步解读

本 notebook **读取已经保存的预处理产物**，不重写 EDF，也不重新选择阈值。按“数据来源 → 滤波 → 连续通道质量 → epoch → 拒绝 → ERP”逐步核查。

Duke 为 F_05_SE001_CBCol_Train02，ETSU 为 F_03_SE001_CBCol_Train01。站点对应关系沿用 notebook 02 的**推断映射**，不是新增确认结果。每站仅一份训练文件，因此图形用于开发验证，不能据此确定最终阈值或推断全部受试者。

每个步骤先说明图表回答的问题，再输出结果及解释。可直接查看已保存的输出，也可在 `bigp3bci` 环境中 Run All。若需重跑预处理，先执行 `scripts/run_causal_pilot.py`。
''')
code('''
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Image, Markdown

# 支持从仓库根目录或 notebooks 目录启动。
ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents]
             if (p / 'configs/causal_pilot.json').exists()), None)
if ROOT is None:
    raise FileNotFoundError('请从项目目录或其 notebooks 子目录启动。')
OUT = ROOT / 'outputs' / 'causal_pilot'
FILES = {'Duke': 'F_05_SE001_CBCol_Train02', 'ETSU': 'F_03_SE001_CBCol_Train01'}
records = {site: json.loads((OUT / stem / 'processing_record.json').read_text(encoding='utf-8'))
           for site, stem in FILES.items()}
tables = {site: pd.read_csv(OUT / stem / 'epoch_metadata_quality.csv',
                          dtype={'valid_epoch': 'boolean', 'classification_eligible': 'boolean'})
          for site, stem in FILES.items()}
plt.rcParams.update({'figure.dpi': 110, 'axes.spines.top': False, 'axes.spines.right': False})
def show_saved(site, image_name, width=1250):
    display(Markdown(f'**{site}（站点推断） · {FILES[site]}**'))
    display(Image(filename=str(OUT / FILES[site] / image_name), width=width))
def read_table(site, name):
    return pd.read_csv(OUT / FILES[site] / name, keep_default_na=False)
print('产物目录：', OUT)
''')
md('''
## 1. 先确认输入和处理范围

下表检查：是否只处理 16 个 EEG 通道、采样率与连续样本数、epoch 样本数，以及采集时已有的滤波。

**采集滤波**来自 EDF 原始 header；**新增滤波**是本次执行的数字处理。即便采集端已做类似滤波，两份文件仍使用相同新增流程。相同新增流程不保证两站原始频响完全一致。
''')
code('''
rows=[]
for site, r in records.items():
    acquisition = sorted({v for k,v in r['acquisition_prefilter'].items() if k.startswith('EEG_')})
    rows.append({'site_inferred': site, 'edf': FILES[site], 'EEG_channels': len(r['channels']),
                 'sfreq_Hz': r['sfreq'], 'continuous_samples': r['n_samples'],
                 'epoch_samples': r['epoch_samples'], 'acquisition_prefilter': '; '.join(acquisition),
                 'added_filter': '60 Hz notch → 0.5–30 Hz causal SOS',
                 'file_needs_channel_policy': r['file_needs_channel_policy']})
display(pd.DataFrame(rows))
display(pd.DataFrame(records['Duke']['config']['filter'].items(), columns=['parameter', 'value']))
''')
md('''
**如何理解：**Duke header 为 HP 0.1 Hz / LP 60 Hz / notch 60 Hz；ETSU 为 HP 0.5 Hz / LP 30 Hz / notch 58–62 Hz。新增滤波为 Q=30 的二阶 notch，随后是原型四阶 Butterworth 带通（带通总八阶）。这是明确记录的工程候选，**不是 BCI2000 的精确复现**。每份 EDF 单独初始化，内部逐块保留滤波状态。
''')
md('''
## 2. 实际滤波器的频响与时间延迟

上图显示各频率被保留或衰减多少：0 dB 表示幅度基本保留，−20 dB 表示幅度约变为 1/10。0.5 Hz 以下、30 Hz 以上衰减，60 Hz 还叠加 notch。

下图是群延迟：因果滤波的延迟随频率变化。不能把它解释成所有 EEG 都晚了固定毫秒数。两站新增滤波器相同，因此展示一份设计响应，并列出代表频率的延迟。
''')
code('''
show_saved('Duke', 'filter_response.png', 1000)
response = pd.read_csv(OUT / FILES['Duke'] / 'filter_response.csv')
sample_response=[]
for hz in [0.5, 1, 5, 10, 20, 30, 60]:
    v=response.iloc[(response.frequency_hz-hz).abs().argmin()]
    sample_response.append({'frequency_Hz': hz, 'gain_dB': v.gain_db,
                            'group_delay_ms': v.group_delay_seconds * 1000})
display(pd.DataFrame(sample_response).round(3))
''')
md('''
**如何理解：**例如 10 Hz 约延迟 16.4 ms，1 Hz 约延迟 251.6 ms；这些是对应频率的群延迟，不是 ERP 峰值的统一修正量。notch 附近幅度极小，其数值群延迟不适合解释被保留信号的时序。**事件 onset 保持原样，没有按延迟移动**。
''')
md('''
## 3. 逐通道与平均 PSD：滤波改变了哪些频率？

蓝线是滤波前，橙线是因果滤波后。纵轴为绝对 PSD：`dB(µV²/Hz)`；不是每条曲线单独归一化。每份图内所有通道共享坐标，60 Hz 用竖线标出。最后一格先在线性功率域平均通道，再转 dB。

重点看 30 Hz 以上和 60 Hz 是否进一步衰减；不要因原始 60 Hz 已有凹口就判断新增 notch 未执行。原始 EDF 本身可能已带采集滤波。Welch PSD 使用 4 秒窗，频率分辨率为 0.25 Hz；全文件 PSD 只是开发诊断，不参与测试阈值估计。
''')
for site in ['Duke','ETSU']:
    code(f"show_saved('{site}', 'psd.png')")
md('''
**低频放大图：**下面专看 0–3 Hz 的平均 PSD。低频衰减受原始采集滤波、有限窗分辨率和文件中的瞬变影响，不能只凭一个频点估计滤波器截止频率；截止特性应结合步骤 2 的设计响应。
''')
code('''
for site in FILES:
    show_saved(site, 'psd_low_frequency.png', 1000)
''')
md('''
## 4. 同时间、同纵轴的波形：启动、普通片段、瞬变附近

每格蓝/橙曲线叠加在同一纵轴，单位 µV，横轴为 EDF 起点后的秒数；同一格内不改变显示比例。通道间可以有不同纵轴范围，因此比较滤波前后时应在同一格内观察。

- **startup**：文件最初 10 秒，检查初始化造成的过渡。
- **signal**：固定时间片段，观察慢漂移和快速成分变化。
- **transient**：原始 EEG 最大相邻样本跳变附近，观察瞬变后的过冲、振荡和恢复。这是回顾性画图选择，不是在线异常判定。

橙线仍可能有大幅瞬变；滤波不等于去除所有伪迹。图中相位错位是因果滤波预期行为，不应为让曲线对齐而移动事件。
''')
for site in ['Duke','ETSU']:
    for kind in ['startup','signal','transient']:
        code(f"show_saved('{site}', 'waveform_{kind}.png')")
md('''
**启动保护区的含义：**当前候选设置将前 10 秒视为需要保护的启动区，涉及 epoch 无效。这是保守开发选择，不表示已经证明滤波器恰好需要 10 秒稳定。后续可独立验证并调整。瞬变附近的振荡也可能来自真实信号及带通响应；这些图不提供 EOG/ECG/EMG 来源分类。
''')
md('''
## 5. 连续通道质量：先看原始数据，避免滤波掩盖平线

2 秒窗、1 秒步长，按当时已完成的历史窗口累计。固定候选阈值检查非有限、常值、全零、低变化及大幅/大波动；不依据整份测试 EDF 的分布重新设阈值。

下表的 fraction 是窗口占比，不是精确异常时间占比，因为窗口重叠。persistent_candidate 表示持续异常候选，需要后续决定通道策略，不能理解成自动删除通道。
''')
code('''
for site in FILES:
    display(Markdown(f'**{site}：原始连续通道质量**'))
    display(read_table(site, 'channel_quality.csv'))
''')
code('''
fig, axes=plt.subplots(2,1,figsize=(13,6),sharex=True,sharey=True,layout='constrained')
for ax,site in zip(axes,FILES):
    w=read_table(site,'continuous_windows.csv')
    for channel,g in w.groupby('channel',sort=False):
        ax.plot(g.available_sample / records[site]['sfreq'], g.history_anomaly_fraction,
                label=channel, alpha=.6, linewidth=1)
    ax.axhline(records[site]['config']['continuous_qc']['persistent_fraction'],color='red',ls='--',label='candidate fraction')
    ax.set(title=f'{site}: anomaly fraction in completed history',ylabel='window fraction',ylim=(-.02,1.02))
axes[-1].set_xlabel('EDF time (s)')
axes[0].legend(ncol=6,fontsize=7,loc='upper right')
plt.show()
''')
md('''
**本次结果：**两份文件未检出持续异常通道候选，不代表已证明所有片段无伪迹。当前原始平线检查时间尺度为 2 秒，较短平线可能漏检；epoch PTP 提供另一层检查。

如果后续文件出现持续异常：保留 16 通道，相关 epoch 的 `valid_epoch` 留空并标为 `pending_channel_policy`，文件暂不可分类。普通拒绝与“通道策略未确定”必须分开，不能把留空当作通过。
''')
md('''
## 6. Epoch 的样本窗口与事件、字符对应关系

每次 flash onset 为零点，提取 `[onset, onset + ceil(0.8×sfreq))`。256 Hz 下为 **205 个样本**，最后一个点是 796.875 ms；800 ms 右端点不包含。在线仍等待完整 800 ms 后判定。相邻 flash 的 epoch 重叠是允许的。

`StimulusType=0/1` 已在原有代码确认对应 non-target/target；`SelectedTarget` 上升沿作为字符结束边界。下面展示所有事件表的前几行，包括无效 epoch，没有只留下通过记录。Baseline=None。
''')
code('''
columns=['event_index','subject_id','session_id','site','edf_name','condition',
         'character_trial_id','flash_index','onset_sample','onset_seconds','stimulus_code',
         'label','available_sample','decision_seconds','complete_epoch','valid_epoch','rejection_reason']
for site,t in tables.items():
    display(Markdown(f'**{site}**'))
    display(t[columns].head(8))
    print('事件位置未移动：', bool((t.onset_sample==t.original_event_sample).all()),
          '；完整窗口：', int(t.complete_epoch.sum()), '/', len(t),
          '；未知字符边界：', int((~t.character_boundary_known).sum()))
''')
md('''
**如何理解：**onset_sample 是原始文件内样本位置；available_sample 是提取所需最后一个样本，decision_seconds 是最早等待满 800 ms 的时间。重复目标字符不会合并 trial，因为边界不靠 CurrentTarget 的 unique 数量推断。若末尾窗口不足，事件仍保留，complete_epoch=False 并记录 incomplete；本次两份文件没有此类事件。
''')
md('''
## 7. PTP 分布与少量候选阈值

PTP = 每个 epoch、每个 QC 通道的最大值减最小值，**在统一因果滤波之后计算**。图中单位 µV，实现内部转换为 V 比较。高 PTP 提示残留大幅异常，低 PTP 提示近乎平直，但两者都不是伪迹来源诊断。

直方图每格是一个通道的逐 epoch 分布。阈值候选为 (0.5,150)、(0.5,250)、(0.2,350) µV；当前选第二组仅为了输出一套可检查的质量表。这些不是已验证的最终标准。标签不参与是否通过的判定。
''')
for site in ['Duke','ETSU']:
    code(f"show_saved('{site}', 'ptp_distributions.png')")
code('''
comparison=pd.concat([read_table(site,'threshold_comparison.csv').assign(site=site) for site in FILES],ignore_index=True)
display(comparison)
fig,ax=plt.subplots(figsize=(9,4),layout='constrained')
for site,g in comparison.groupby('site',sort=False):
    ax.plot(g.candidate,g.retention*100,marker='o',label=site)
ax.set(xticks=[0,1,2],xticklabels=['0.5–150 µV','0.5–250 µV','0.2–350 µV'],
       ylabel='retention (%)',xlabel='candidate bounds',title='Retention includes all QC reasons, including startup')
ax.legend(); plt.show()
for site in FILES:
    display(Markdown(f'**{site} · 严格候选（0.5–150 µV）的通道触发数**'))
    display(pd.read_csv(OUT / FILES[site] / 'candidate_0_channel_triggers.csv', index_col=0))
''')
md('''
**本次结果：**默认候选两站各保留 734/756。严格 150 µV 上限使 Duke 保留数降为 729，由 EEG_T7 触发 5 个；ETSU 仍为 734。候选比较中的保留率包括启动等全部无效原因，不只是 PTP。

这说明结果会受上限影响，**不能据此直接选择保留率最高的一组**。最终选择还需遵守验证划分及 calibration budget，并用已完整检查的人工区间验证；未检查区间不能当作正常。本次没有用人工 annotation 校准阈值，也没有使用测试分布。
''')
md('''
## 8. 沿时间查看每个 epoch 的质量与拒绝原因

上图显示每个 epoch 的跨 QC 通道最大 PTP；红叉表示默认规则无效，灰色区域为启动保护区。下图为逐事件状态，便于分清异常集中在何时。这里只画最大 PTP，完整逐通道数值见 `epoch_ptp_uv.csv`。
''')
code('''
fig,axes=plt.subplots(2,2,figsize=(14,7),sharex='col',layout='constrained')
for col,(site,t) in enumerate(tables.items()):
    valid=t.valid_epoch.fillna(False).to_numpy(dtype=bool)
    pending=t.valid_epoch.isna().to_numpy()
    invalid=(~valid)&(~pending)
    hi=records[site]['config']['epoch_qc']['candidate_bounds_uv'][records[site]['config']['epoch_qc']['selected_candidate']][1]
    axes[0,col].scatter(t.onset_seconds,t.max_ptp_uv,s=8,label='all events',alpha=.5)
    axes[0,col].scatter(t.loc[invalid,'onset_seconds'],t.loc[invalid,'max_ptp_uv'],marker='x',s=20,c='red',label='invalid')
    axes[0,col].axhline(hi,color='orange',ls='--',label='PTP upper bound')
    axes[0,col].set(title=site,ylabel='max PTP across QC channels (µV)')
    axes[0,col].legend(fontsize=8)
    state=np.where(pending,-1,np.where(valid,1,0))
    axes[1,col].scatter(t.onset_seconds,state,c=np.where(pending,'orange',np.where(valid,'tab:green','tab:red')),s=9)
    axes[1,col].set(yticks=[-1,0,1],yticklabels=['pending','invalid','valid'],xlabel='onset time in EDF (s)',ylabel='epoch status')
    for ax in axes[:,col]: ax.axvspan(0,records[site]['config']['filter']['startup_seconds'],alpha=.12,color='gray')
plt.show()
for site,t in tables.items():
    display(Markdown(f'**{site}：默认候选的拒绝原因及触发通道**'))
    reasons=t.rejection_reason.fillna('').str.get_dummies(sep=';').sum().rename('events')
    channels=t.trigger_channels.fillna('').str.get_dummies(sep=';').sum().rename('events')
    display(reasons.to_frame()); display(channels.to_frame())
''')
md('''
**本次结果：**两站默认候选均只有 `filter_startup=22`，没有 PTP 触发通道。空的通道触发表是“没有通道触发拒绝”，不是结果丢失。启动保护由时间触发，所以不归属于某个通道。

同一 epoch 可以有多个原因、多个触发通道，计数相加可能大于拒绝总数。不合格 flash 后续不提供分类分数，也不填 0；同一字符的其他合格 flash 继续保留。
''')
md('''
## 9. EDF / subject / session 与类别保留率

总保留率使用全部 flash 为分母；另列 pending，避免把未确定有效性当作普通通过或拒绝。target/non-target 分开统计只用于观察 QC 对类别保留的影响，标签本身没有参与 QC。
''')
code('''
all_events=pd.concat(tables.values(),ignore_index=True)
def quality_summary(frame,keys):
    rows=[]
    for key,g in frame.groupby(keys,dropna=False,sort=False):
        key=key if isinstance(key,tuple) else (key,)
        n_valid=int(g.valid_epoch.fillna(False).sum())
        rows.append(dict(zip(keys,key)) | {'total':len(g),'valid':n_valid,
                    'pending':int(g.valid_epoch.isna().sum()),'retention_pct':100*n_valid/len(g)})
    return pd.DataFrame(rows)
for keys in [['site','edf_name'],['subject_id'],['subject_id','session_id'],['site','label']]:
    display(quality_summary(all_events,keys).round(3))
fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
label_summary=quality_summary(all_events,['site','label'])
label_summary.pivot(index='site',columns='label',values='retention_pct').plot.bar(ax=ax,ylim=(0,100),rot=0)
ax.set(ylabel='retention (%)',title='Retention by label (diagnostic only)'); plt.show()
''')
md('''
**本次结果：**每站 target 为 81/84（96.43%），non-target 为 653/672（97.17%）。小幅差异来自前 10 秒包含的标签数量差异，不表示算法使用了标签拒绝。每站当前只有一个 subject/session/EDF，因此三个汇总层级数值相同，不能看成三组独立验证。
''')
md('''
## 10. 每个 character trial 还有多少可用 flash？

下面同时画总数和有效数。一次异常只影响对应 flash，不自动删除整个字符。当前 `evidence_insufficient` 仅标记有效 flash 为零；其他最小证据量、字符评分及停止规则尚未决定，因此非零不等于“已经足够准确解码”。
''')
code('''
fig,axes=plt.subplots(1,2,figsize=(13,4),sharey=True,layout='constrained')
for ax,site in zip(axes,FILES):
    c=read_table(site,'character_trials.csv')
    display(Markdown(f'**{site}**')); display(c)
    ax.bar(c.character_trial_id,c.total_flashes,label='total',color='lightgray')
    ax.bar(c.character_trial_id,c.valid_flashes,label='valid',width=.55,color='tab:green')
    ax.set(title=site,xlabel='character trial',ylabel='flash count',xticks=c.character_trial_id)
    ax.legend()
plt.show()
''')
md('''
**本次结果：**每站 6 个字符、每字符 126 次 flash。首字符保留 104 次，其余各 126 次；首字符没有因启动期损失而被整段删除。
''')
md('''
## 11. 拒绝前后的 target / non-target 平均 ERP

红色为 target，蓝色为 non-target；虚线为拒绝前，实线为拒绝后。所有曲线来自**同一份因果滤波数据**，不是原始与滤波后的比较。每格使用共享坐标，图例的 n 是参与平均的 epoch 数。

拒绝前只平均完整窗口（包含启动期），拒绝后只平均有效窗口；Baseline=None，事件没有延迟校正。信号受到因果相位响应及相邻 flash 重叠影响，不能把图中每个峰直接归因于单次刺激或伪迹来源。
''')
for site in ['Duke','ETSU']:
    code(f"show_saved('{site}', 'erp_before_after.png')")
md('''
**如何理解：**两站拒绝后均为 target n=81、non-target n=653，拒绝前分别 n=84/672。前后接近说明本次删去的少量启动 epoch 对文件平均的影响有限，**不证明所有伪迹已清除，也不证明阈值最优或模型有效**。不同通道形态不同，不能因为某个通道波形不典型就直接删除它。

本 notebook 的人工浏览只服务开发和验证；正式测试应使用冻结自动规则，不根据这些图逐段手动删数据。
''')
md('''
## 12. 产物一致性检查与后续边界

最后检查图表背后的事件数量、样本数和有效标记是否一致。原始 EDF 的读前/读后 SHA256 校验已由预处理脚本执行并写入 processing_record.json；本 notebook 只读取产物，不重新声称对源 EDF 执行了校验。

当前仍待研究确定：最终 epoch 阈值、启动保护长度、持续坏通道处理策略、calibration budget 下的阈值估计、字符证据量与评分/停止规则。未加入 ICA、EOG/ECG/EMG 来源分类、插值、训练或 baseline correction。
''')
code('''
checks=[]
for site,t in tables.items():
    with np.load(OUT / FILES[site] / 'epochs_all.npz',allow_pickle=False) as e:
        assert len(t)==len(e['event_index'])==e['eeg_v'].shape[0]
        assert e['eeg_v'].shape[1]==16
        assert e['eeg_v'].shape[2]==records[site]['epoch_samples']
        np.testing.assert_array_equal(e['valid_epoch'],t.valid_epoch.fillna(False).to_numpy(dtype=bool))
        np.testing.assert_array_equal(e['event_index'],t.event_index.to_numpy())
    assert t.onset_sample.is_monotonic_increasing
    assert (t.onset_sample==t.original_event_sample).all()
    checks.append({'site':site,'artifact_checks':'PASS','source_sha256_recorded':records[site]['source_sha256']})
display(pd.DataFrame(checks))
''')
if (ROOT/'outputs/expanded_pilot/run_summary.json').exists():
    from build_expanded_pilot_notebook import expanded_overview_cells
    cells.extend(expanded_overview_cells())
nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python (bigp3bci)','language':'python','name':'python3'},'language_info':{'name':'python'}})
path=ROOT/'notebooks/05_bigp3bci_causal_preprocessing_pilot.ipynb'
nbf.write(nb,path)
print(path)
