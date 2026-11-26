"""Build the expanded preprocessing chapter in the consolidated report."""
from pathlib import Path
import textwrap
import nbformat as nb
from notebook_reports import write_report
ROOT=Path(__file__).resolve().parents[1]

def main():
    cells=[]
    def md(s):cells.append(nb.v4.new_markdown_cell(textwrap.dedent(s).strip()))
    def code(s):cells.append(nb.v4.new_code_cell(textwrap.dedent(s).strip()))
    md('''
# 07 · 扩展训练 pilot：事件审计、独立眨眼候选与 PTP 模拟

本轮实际运行生成产物的代码，不只是改图注。正式分析信号仍为 **60 Hz notch → 0.5–30 Hz 因果 SOS**；只有EEG被处理，原始EDF与状态未改，事件不移动，连续文件保留状态，不跨文件传状态。0–800 ms半开样本窗口与baseline=None沿用05。

新增分支输出候选和统计，**不新增自动删除、插值或重参考，不训练模型**。本notebook读取保存产物；前一版05已追加本轮概览。
''')
    code('''
from pathlib import Path
import json
import numpy as np
import pandas as pd
from IPython.display import display, Image, Markdown
ROOT=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'configs/expanded_pilot.json').exists())
OUT=ROOT/'outputs/expanded_pilot'
manifest=pd.read_csv(OUT/'selection_manifest.csv')
audit=pd.read_csv(OUT/'event_audit.csv')
events=pd.read_csv(OUT/'all_events.csv')
blinks=pd.read_csv(OUT/'all_blinks.csv')
quality=pd.read_csv(OUT/'all_quality.csv')
parameters=json.loads((OUT/'config_snapshot.json').read_text())
EXAMPLE_EDF='F_05_SE001_CBCol_Train02.edf' # 可改为manifest中任意文件
EDF_DIR=OUT/Path(EXAMPLE_EDF).stem
def picture(name,width=1200):display(Image(filename=str(OUT/'figures'/name),width=width))
print(json.dumps(json.loads((OUT/'run_summary.json').read_text()),indent=2))
''')
    md('''
## 1. 固定的文件选择，避免按检测结果挑选

四名被试F_03/F_05/F_06/F_20；每人取最早两个可用Session；每Session取字典序第一份Train/CBCol EDF。manifest在读取信号、检测候选前写入；后续重复运行若选择变化会报错。两站映射仍为inferred。没有使用任何Test EDF。
''')
    code('''
display(manifest.reset_index(names='zero_based_index'))
print('下面图中file index为上表zero_based_index + 1。')
''')
    md('''
## 2. 直接核对原始状态值、flash onset与标签读取

独立读取EDF int16数字样本和header缩放，与MNE读取的状态逐项核查。`StimulusType`原始0/1和浏览器事件码1/2是不同概念。保留原有0→non-target、1→target语义，并输出exact-value一致性；不靠四舍五入掩盖非预期值。

flash取`StimulusBegin`由≤0.5跃迁到>0.5的样本，在**同一个onset样本**读取StimulusType；另存前一个/后一个样本值用于核对是否存在错位。独立边沿提取与现有helper一致性逐文件检查，若不同则停止，不按预期改计数。
''')
    code('''
display(audit[['edf_name','stimulus_type_unique','stimulus_type_unique_at_flash','flash_count','helper_onsets_match',
               'helper_exact_label_mismatch','mne_matches_edf_physical_stimulus_type','sample_zero_active']])
counts=[]
for _,r in manifest.iterrows():
    c=pd.read_csv(OUT/Path(r.edf_name).stem/'state_value_counts.csv')
    counts.append(c[c.channel.eq('StimulusType') & c.representation.eq('raw_edf_digital')].assign(edf_name=r.edf_name))
display(pd.concat(counts,ignore_index=True))
state_examples=pd.read_csv(EDF_DIR/'events_audited.csv')
display(state_examples[['onset_sample','stimulus_begin_previous','stimulus_begin_at_onset','stimulus_type_previous',
                        'stimulus_type_raw','stimulus_type_next','stimulus_type_digital','stimulus_type_physical','event_code','label']].head(12))
''')
    md('''
**读法：**all_samples计数含刺激间隔，不能当作flash数；flash_onsets计数才对应标签事件数量。若数字值/物理值/MNE一致，说明不是读取缩放造成的0/1差异。标签语义仍来自已有项目确认结果；仅凭出现0/1本身并不能证明哪一个是target。
''')
    md('''
## 3. 字符边界与实际flash数量：不强制126

边界是SelectedTarget从零进入非零的上升沿。按 `previous_boundary < onset < next_boundary` 归组；边界重合及未分配事件单独计数。下面显示全部字符核查，不补齐数量。
''')
    code('''
characters=[]
for _,r in manifest.iterrows():
    c=pd.read_csv(OUT/Path(r.edf_name).stem/'character_event_audit.csv')
    characters.append(c.assign(edf_name=r.edf_name))
characters=pd.concat(characters,ignore_index=True)
display(characters[['edf_name','character_trial_id','end_boundary_exclusive','n_flashes','n_target','n_nontarget','n_unknown',
                    'current_target_values','code_first','code_last','code_adjacent_increment_one']])
display(audit[['edf_name','n_character_boundaries','unassigned_flashes','flashes_on_boundary']])
''')
    md('''
**实际差异：**F_20 SE002首字符108次（12 target、96 non-target），其他字符126次，该EDF共738次；其余7份各756次。108是原始状态与边界解析得到的数量，不把该字符改成126，也不自动剔除。其原因是否与采集流程有关，目前没有额外证据。

**刺激组：**StimulusCode在本轮文件中表现为递增flash计数，不能直接假定是固定行/列组。这里给每次flash实际亮起的字符通道集合生成稳定签名，保留完整成员表，不编造固定组数。同样的集合签名才合并统计；很多集合只出现一次，这会限制每组统计的解释。原始StimulusCode仍在metadata中。
''')
    code('''
groups=pd.read_csv(EDF_DIR/'stimulus_group_dictionary.csv')
display(groups.head(12))
print('示例EDF观察到的字符集合数：',len(groups))
''')
    md('''
## 4. 幅值与PTP的被试、Session、通道差异

第一张图是连续信号q95−q05幅值范围，分别展示原始和正式滤波信号；不是峰峰值拒绝阈值。第二张图是每通道的逐epoch PTP箱线图，文件索引对应manifest1–8；隐藏离群点只为可读性，数值全部保留。

分位数是描述性报告，不参与阈值估计。跨被试/Session差异不能用“每人固定拒绝同一比例”消除，也不自动转化为个体化规则。
''')
    code("picture('amplitude_distribution.png')\npicture('ptp_by_file_channel.png',1350)")
    code('''
display(pd.read_csv(OUT/'ptp_subject_quantiles.csv').head(20))
display(pd.read_csv(OUT/'ptp_session_quantiles.csv').query("channel in ['EEG_Fz','EEG_Pz','EEG_T7']"))
print('原始/滤波连续幅值与epoch PTP分位数：',OUT/'all_distributions.csv')
print('全部epoch×16通道原值：',OUT/'all_ptp.csv')
''')
    md('''
## 5. 独立眨眼候选：共同额区慢变化，而非Fz单峰

正式信号保持不变。独立分支在正式因果信号之后接二阶因果5 Hz低通，取F3/Fz/F4的带符号中位数作为共同额区波形，对其绝对幅值做滞回扫描，允许向上或向下。

达到entry后启动片段；降到exit以下并持续recovery时确认恢复。近邻片段按merge_gap合并。随后检查持续时间、至少两个额区通道的同向幅值与相关性、慢分量PTP占比、峰间距。幅值为峰时额区绝对幅值的中位数；后部为同一时点Pz/PO7/PO8/Oz绝对幅值中位数。额/后比只是支持指标（当前不设硬拒绝门槛）。它不证明来源是眼动，也不能把真实脑电或任务ERP排除在外。

检测函数只接受EEG、采样率、通道名及固定参数，不接受target/non-target，也不根据整份文件估计阈值。
''')
    code('''
display(pd.Series(parameters['blink_detector'],name='candidate parameter').to_frame())
counts=manifest[['edf_name','site','site_status']].merge(blinks.groupby('edf_name').size().rename('blink_candidates'),on='edf_name',how='left').fillna({'blink_candidates':0})
display(counts)
display(blinks[['edf_name','blink_candidate_id','label','onset_seconds','peak_seconds','offset_seconds','decision_seconds',
                'frontal_amplitude_uv','posterior_amplitude_uv','frontal_posterior_metric','frontal_dominance_support',
                'supporting_channels','slow_ptp_fraction','filter_startup']].head(15))
''')
    md('''
**延迟解读：**onset/offset是辅助分支上的回溯波形边界；decision_sample为恢复确认与offset后合并等待两者中较晚的样本，检测不会在decision之前输出。辅助滤波还增加频率相关的信号延迟，不能把候选起点当作原始眨眼真实起点，也不能把它当成实时决策时间。

EOF未确认恢复的片段只进入excursion log，不包装为已确认候选。启动期候选可保留绘图，但filter_startup单独标记，不算人工验证中的成功检出。
''')
    code('''
display(blinks.groupby('edf_name').detection_delay_from_peak_seconds.agg(['count','min','median','max']))
picture('causal_branch_responses.png',1000)
''')
    md('''
## 6. 少量候选与未触发对照片段

每个站点固定取manifest的第一份文件，展示最早的非启动期候选和最早固定网格上无候选的4秒片段，共最多4图；不挑最好看的例子。上行为额区，中行为后部，下行为辅助慢分支。灰竖线为flash，橙区为候选，红虚线为峰，紫点线为最早判定时刻。

“未触发”仅是算法对照，**不等于人工确认正常**。同样，触发也不意味着epoch必须拒绝。
''')
    code('''
examples=pd.read_csv(OUT/'plotted_examples.csv')
display(examples)
for name in examples.figure:picture(name,1100)
''')
    md('''
## 7. 原始信号基础故障与逐epoch指标分别保存

原始数据检查NaN/Inf、≥0.25秒常值/全零、0.5秒窗内低变化，以及相邻样本≥100 µV跳变候选。低变化窗步长0.25秒，PTP≤0.5 µV且STD≤0.1 µV；这些是显式工程候选，不是最终标准。不开展心搏、肌电来源分类。

原始候选区间与epoch的关联保留涉及通道和decision时间；完整窗口的原始标记是回顾性关联，模拟时只使用在epoch决定前已经可用的基础标记。持续异常仅报告，绝不删通道、插值或排文件。
''')
    code('''
if len(quality):
    display(quality.groupby(['edf_name','label']).size().rename('candidates').to_frame())
    display(quality.head(12))
else:print('本轮未检出基础质量候选；不表示已证明无故障。')
persistent=pd.read_csv(OUT/'all_persistent.csv')
display(persistent[persistent.persistent_candidate])
display(events[['edf_name','event_index','flat_candidate','flat_channels','jump_candidate','jump_channels','nonfinite_flag','filter_startup','max_ptp_uv','min_ptp_uv']].groupby('edf_name').agg({'event_index':'size','flat_candidate':'sum','jump_candidate':'sum','nonfinite_flag':'sum','filter_startup':'sum'}))
''')
    md('''
## 8. 候选与epoch的多对多关联，不自动拒绝

每条epoch保存has_blink_candidate、ID列表、重叠并集时长、逐通道PTP、flat/jump/nonfinite与原始涉及通道。独立关联表保留一条“候选—epoch”配对，候选可能对应多个重叠epoch。

若候选decision晚于epoch的800 ms等待终点，known_by_epoch_decision=False；未来若要使用该标记控制分类，必须等到earliest_joint_decision_seconds或采用另定的在线政策。本轮只输出事实，不实施该政策。
''')
    code('''
e=events[events.edf_name.eq(EXAMPLE_EDF)]
display(e[['event_index','character_trial_id','onset_sample','label','has_blink_candidate','blink_candidate_ids','blink_overlap_duration',
           'max_ptp_uv','min_ptp_uv','flat_candidate','jump_candidate','nonfinite_flag','ptp_by_channel']].head(10))
links=pd.read_csv(EDF_DIR/'blink_epoch_links.csv')
display(links.head(15))
ptp=pd.read_csv(EDF_DIR/'epoch_ptp_long.csv')
display(links.merge(ptp,on='event_index').head(20))
assert 'valid_epoch' not in events.columns
''')
    md('''
## 9. 人工annotation与完整检查区间

读取对应CSV和FIF并核对一致性。当前F_05 SE001已有51段疑似眨眼；沿用用户确认将此文件MANUAL_bad_span解释为疑似眨眼，不把其他文件所有bad_span都猜成眨眼。原始标注不重写。

只有在显式reviewed_intervals中完整检查过、且不跨启动期/不确定标记的事件才参与验证。未检查区间不当正常，不把未匹配候选算误报。**当前没有reviewed_intervals，所以漏检和未匹配统计不可评价，相关指标留空。**下面重叠/边界差只作描述，不是检出率。
''')
    code('''
display(pd.read_csv(OUT/'manual_validation_status.csv'))
display(pd.read_csv(EDF_DIR/'manual_event_comparison.csv').head(15))
display(pd.read_csv(EDF_DIR/'manual_candidate_overlaps.csv').head(15))
display(pd.read_csv(EDF_DIR/'manual_event_epoch_ptp.csv').head(20))
''')
    md('''
未来人工完整检查文件格式为CSV：edf_name,start_sec,end_sec,review_status；仅显式complete区间进入评价，路径见config。程序不会自行创建ground truth。人工眨眼也仅代表疑似眨眼，不代表相关epoch必须拒绝。

边界差定义为自动onset−人工start、自动offset−人工end。原始人工边界与经过额外因果滤波的候选边界不在相同信号域，差异混合了标注与滤波时序影响，不可直接解释为定位误差。
''')
    md('''
## 10. PTP只做模拟对照：150/250/350 µV与无上限

all16为全通道对照；example8只展示未来8通道QC的可计算性，**不是已确定的分类输入集合**。每个集合的4个方案基础条件相同：不完整、启动期、非有限、滤波恢复、已可用的原始平线、PTP低于0.5 µV。jump和持续异常只报告，眨眼候选完全不参与would_reject。

no_upper只关闭PTP上限，仍保留其他条件，所以预计保留率可能小于100%。方案只产生would_keep/would_reject，没有实际删数据或选最优规则。
''')
    code('''
simulation=pd.read_csv(OUT/'simulation_edf_summary.csv')
display(simulation[simulation.scheme.str.startswith('all16')].pivot(index='edf_name',columns='scheme',values='expected_retention').round(4))
picture('threshold_retention.png')
display(pd.read_csv(OUT/'simulation_label_summary.csv').query("scheme in ['all16_no_upper','all16_upper_150uv','all16_upper_250uv']"))
''')
    md('''
## 11. 每字符、每刺激集合的有效flash数与主要触发通道

“有效”在本节仅表示假设采用该方案时would_keep，不是最终入模判定。保留所有原始flash，可按任意scheme查看。不因一次异常自动删整个字符。刺激集合为实际亮起的字符状态集合签名，不能冒称固定行/列编号。
''')
    code('''
character_summary=pd.read_csv(OUT/'simulation_character_summary.csv')
display(character_summary[character_summary.edf_name.eq(EXAMPLE_EDF)&character_summary.scheme.str.startswith('all16')])
group_summary=pd.read_csv(OUT/'simulation_stimulus_group_summary.csv')
display(group_summary[group_summary.edf_name.eq(EXAMPLE_EDF)&group_summary.scheme.eq('all16_upper_150uv')].head(20))
display(pd.read_csv(OUT/'simulation_trigger_channels.csv').sort_values('count',ascending=False).head(20))
print('全部刺激集合计数：',OUT/'simulation_stimulus_group_summary.csv')
''')
    md('''
## 12. 产物一致性与结论边界

检查：事件位置和数量不变、16通道逐epoch指标完整、候选decision晚于peak、眨眼候选没有被变为valid_epoch=False、每个方案保存全部事件。源EDF读前/读后hash已记录。测试另外覆盖正负波形对称、拒绝Fz单峰解释、未来数据独立、基础故障与候选不控制模拟。

仍缺少：站点映射的独立证据、108次首字符的采集原因、完整人工复核阴性区间、候选的独立训练集验证、真正刺激组语义说明、最终分类通道与阈值政策。候选数差异不能当作真实眨眼率差异。本轮不确定个体化阈值，不声称候选有效。
''')
    code('''
assert audit.helper_onsets_match.all() and audit.helper_exact_label_mismatch.eq(0).all()
assert (events.onset_sample==events.original_event_sample).all()
assert 'valid_epoch' not in events.columns
assert (blinks.decision_seconds>=blinks.offset_seconds).all()
assert (blinks.decision_seconds>blinks.peak_seconds).all()
checks=[]
for _,r in manifest.iterrows():
    folder=OUT/Path(r.edf_name).stem
    rec=json.loads((folder/'processing_record.json').read_text())
    p=pd.read_csv(folder/'epoch_ptp_long.csv')
    s=pd.read_csv(folder/'threshold_simulation_epochs.csv')
    assert rec['source_sha256_before']==rec['source_sha256_after']
    assert p.groupby('event_index').size().eq(16).all()
    assert s.groupby('scheme').size().eq(rec['n_epochs']).all()
    with np.load(folder/'epochs_all.npz') as a:
        assert a['eeg_v'].shape==(rec['n_epochs'],16,rec['epoch_samples'])
    checks.append({'edf_name':r.edf_name,'epochs':rec['n_epochs'],'integrity':'PASS'})
display(pd.DataFrame(checks))
''')
    n=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python (bigp3bci)','language':'python'}})
    path=ROOT/'notebooks/07_bigp3bci_expanded_candidate_pilot.ipynb';write_report(n,path)
    print(path)

if __name__=='__main__':main()
