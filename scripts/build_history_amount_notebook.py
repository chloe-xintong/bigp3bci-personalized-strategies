"""Report notebook 12 for the frozen exploratory historical-file enumeration."""
from pathlib import Path
import nbformat as nb

ROOT=Path(__file__).resolve().parents[1]


def main():
    md=nb.v4.new_markdown_cell;code=nb.v4.new_code_cell
    cells=[md('''# 12 · 历史数据量 H × 校准字符 k：探索性 SWLDA 分析

**这是在看到过测试结果后进行的探索性分析，只使用 Train 验证，不读取 Test。** 不改变主实验或测试结果。规则在运行前冻结于 `docs/history_amount_exploratory_protocol.md` 和 `configs/history_amount_exploratory.json`，不按结果挑选子集或调参。

历史池：每人 S1、S2 各 3 个 Train EDF，共 6 文件、36 个字符。以完整文件枚举 H=0…6，子集数依次为 1、6、15、20、15、6、1；每文件 6 个字符。当前池：S3 Train 前 12 个字符，只使用前 k∈{0,3,6,12} 个。最后 6 个字符固定验证，与训练无重叠。H=0,k=0 无模型，不报告仅当前 k=0 的指标或差异。

固定主实验的 16 通道、256 维特征、A 质量规则与 SWLDA 设置。共 2,550 个实际模型；所有子集均保留。验证共 7,560 个 flash，保护期排除 220 个，保留 7,340 个。主指标是逐被试 flash ROC-AUC，**先平均同一被试同一 H,k 的全部子集，再平均被试**。不把子集或相关 flash 当成独立样本。

区间为 10,000 次按被试 bootstrap 的百分位 95% 区间，固定种子 20261126。比较先求被试内差异，使用配对重采样。只有 10 人，区间只作探索性不确定性描述，不作经过多重比较校正的确认性显著性结论；字符准确率仅描述。
'''),code('''from pathlib import Path
import json
import pandas as pd
from IPython.display import display, Image, Markdown
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents]
            if (p / 'configs/history_amount_exploratory.json').exists())
OUT = ROOT / 'outputs/history_amount_exploratory_v1'
if not (OUT / 'completion.json').exists():
    raise FileNotFoundError('请先完成 scripts/run_history_amount_exploratory.py；本 notebook 不启动拟合。')
macro = pd.read_csv(OUT / 'macro_summary.csv')
subject = pd.read_csv(OUT / 'subject_summary.csv')
subsets = pd.read_csv(OUT / 'subset_results.csv', dtype={'subset_id': str})
display(pd.Series(json.loads((OUT / 'completion.json').read_text())).to_frame('audit'))
if (OUT / 'independent_audit.json').exists():
    display(pd.Series(json.loads((OUT / 'independent_audit.json').read_text())).to_frame('independent checks'))
'''),md('''## 历史数量与校准量：10 个被试面板＋宏平均

主结果先给放大的宏平均图；10 人的 2×5 网格作为附录。所有面板共享固定纵轴 0.4–1.0，便于看出被试间的天花板差异。配色：k=0（仅历史，属于另一种方案）为黑色方块；k=3、6、12 为由浅到深的紫色 `#BCBDDC`、`#807DBA`、`#54278F`（圆、三角、菱形），与 notebook 11 的方案色（蓝/橙/蓝绿）分开，灰度打印也可区分。散点是每个枚举文件子集，轻微抖动仅用于避免重叠；实线是子集均值。浅灰细虚线为仅当前参考，右侧标注对应 k；k=0 无该参考。宏平均只画 k=3 与 k=12 的按被试 bootstrap 区间，避免多层叠色。k=12 仅当前 AUC ≥ 0.95 的被试在标题注明“baseline near ceiling”，其历史增益空间本来就小。

每位被试只验证 6 个字符；flash 也存在字符内相关。单人面板不作显著性判断，不能用散点数量代替被试数。
'''),code('''display(Image(filename=str(OUT / 'history_amount_macro.png')))
display(Image(filename=str(OUT / 'history_amount_small_multiples.png')))
display(macro[['H', 'k', 'history_characters', 'subsets_per_subject', 'mean_auc', 'ci_low', 'ci_high']])
'''),md('''## 相同 k 下相对仅当前的差异

先在每名被试内计算 `mean_subset_auc(H,k) − current_only_auc(k)`，再对被试进行配对 bootstrap。题述“历史数量效应”和“历史＋当前 vs 仅当前”在此是同一组对照，不算成两份独立证据。k=0 无当前模型，相关差异保留缺失。
'''),code('''display(Image(filename=str(OUT / 'history_vs_current_small_multiples.png')))
display(pd.read_csv(OUT / 'paired_history_vs_current.csv'))
'''),md('''## Session 来源：同样文件数，仅 S2 减仅 S1

每个 Session 只有 3 文件，因此纯 Session 对照只在 H=1、2、3 定义。每人先分别平均所有纯 S1、纯 S2 子集，再求差；宏平均区间按被试配对 bootstrap。正值表示同等历史文件数下 S2 较高。历史顺序与电极状态等可能混杂，不能据此作 Session 的因果解释。
'''),code('''display(Image(filename=str(OUT / 'session_contrast_small_multiples.png')))
display(pd.read_csv(OUT / 'session_macro_contrasts.csv'))
'''),md('''## 附录：每名被试的 AUC 均值与字符准确率描述

主图散点已保留全部子集；以下均值表不删除低分子集。字符准确率不计算显著性检验或推断性区间。
'''),code('''for sid in sorted(subject.subject_id.unique()):
    display(Markdown('### ' + sid))
    display(Image(filename=str(OUT / 'subjects' / (sid + '.png'))))
    g = subject[subject.subject_id.eq(sid)]
    display(g.pivot(index='history_characters', columns='k', values='mean_auc'))
display(macro[['H', 'k', 'mean_character_accuracy']])
display(Image(filename=str(OUT / 'character_accuracy_small_multiples.png')))
'''),md('''## EEGNet：仅预留计划，本次无训练结果

`eegnet_subset_plan.csv` 固定每个 H 的种子 0–4，在不同 k 间复用子集，记录 S1/S2 文件数。不同种子允许重复；H=0、6 的唯一文件组合不能被当成五个独立子集。

本仓库尚无 EEGNet 训练循环，固定 epoch 数也尚未指定。因此这次不生成 EEGNet 指标或图。若后续实施，必须先固定 epoch 数，**本探索不能根据报告用验证集选择 checkpoint**；其他实验中的早停接口不适用于此方案。
'''),md('''## 复现与输出

运行：`python scripts/run_history_amount_exploratory.py`。同一配置可恢复已完成模型，配置和来源变化时拒绝续跑。重绘：`python scripts/plot_history_amount.py`；重建报告：`python scripts/build_history_amount_notebook.py`。

- `history_file_pool.csv`、`enumeration_manifest.csv`：历史文件、组合及 S1/S2 构成。
- `character_roles.csv`、`validation_flash_manifest.csv`：固定角色与验证 flash。
- `subset_results.csv`、`jobs/`：全部子集指标、模型、逐字符预测和训练特征行，未挑选子集。
- `subject_summary.csv`、`macro_summary.csv`：先被试内平均再宏平均的结果。
- `paired_history_vs_current.csv`、`session_subject_contrasts.csv`、`session_macro_contrasts.csv`：对照与区间。
- `main_endpoint_checks.csv`：与主实验可对应端点的重算一致性检查。

PNG 与 PDF 图采用同一数据和坐标范围。所有结果是看到过测试结果后的验证集探索，不影响主结果。
''')]
    n=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'bigp3bci','display_name':'Python (bigp3bci)','language':'python'},'language_info':{'name':'python'}})
    path=ROOT/'notebooks/12_history_amount_exploratory.ipynb';nb.validate(n);nb.write(n,path);print(path)


if __name__=='__main__':main()
