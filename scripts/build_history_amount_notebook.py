"""Report notebook 08 for the frozen exploratory historical-file enumeration."""
from pathlib import Path
import nbformat as nb

ROOT=Path(__file__).resolve().parents[1]


def main():
    md=nb.v4.new_markdown_cell;code=nb.v4.new_code_cell
    cells=[md('''# 08 · 历史数据量 H × 校准字符 k：探索性 SWLDA 分析

探索性分析，在看过测试结果之后进行，只使用 Train 验证。规则在运行前写定于 `docs/history_amount_exploratory_protocol.md` 和 `configs/history_amount_exploratory.json`。

历史池：每人 S1、S2 各 3 个 Train EDF，共 6 个文件、36 个字符，每个文件 6 个字符。按完整文件枚举 H=0–6，子集数依次为 1、6、15、20、15、6、1。当前池：S3 Train 前 12 个字符，取前 k∈{0,3,6,12} 个；最后 6 个字符为固定验证集。H=0、k=0 没有模型。

特征、质量规则和 SWLDA 设置与主实验相同（16 通道、256 维、A 规则）。共拟合 2,550 个模型，全部子集都保留。验证集共 7,560 个 flash，保护期排除 220 个，保留 7,340 个。主指标是 flash ROC-AUC：先对同一被试、同一 H,k 的全部子集取平均，再对被试取宏平均。

区间为按被试 bootstrap（10,000 次，种子 20261126）的百分位 95% 区间；差值先在被试内计算，再配对重采样。
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
'''),md('''## 历史数据量与校准量

先给放大的宏平均图，再给 10 名被试的 2×5 网格。所有面板纵轴为 0.4–1.0。配色：k=0（仅历史）为黑色方块；k=3、6、12 为由浅到深的紫色（圆、三角、菱形）。散点是各文件子集（加了轻微抖动），实线是子集均值。浅灰细虚线为仅当前参考，右侧标注对应的 k。宏平均画 k=3 与 k=12 的区间。k=12 仅当前 AUC ≥ 0.95 的被试，标题注明 “baseline near ceiling”。
'''),code('''display(Image(filename=str(OUT / 'history_amount_macro.png')))
display(Image(filename=str(OUT / 'history_amount_small_multiples.png')))
display(macro[['H', 'k', 'history_characters', 'subsets_per_subject', 'mean_auc', 'ci_low', 'ci_high']])
'''),md('''## 相同 k 下相对仅当前的差异

每名被试计算 `mean_subset_auc(H,k) − current_only_auc(k)`，再对被试配对 bootstrap。k=0 没有仅当前模型，差值为空。
'''),code('''display(Image(filename=str(OUT / 'history_vs_current_small_multiples.png')))
display(pd.read_csv(OUT / 'paired_history_vs_current.csv'))
'''),md('''## Session 来源：相同文件数下，仅 S2 减仅 S1

每个 Session 有 3 个文件，所以纯 Session 对照定义在 H=1、2、3。每名被试先分别平均全部纯 S1、纯 S2 子集，再求差；正值表示 S2 更高。S2 在时间上更接近 S3，二者的差异同时包含时间顺序和记录状态的影响。
'''),code('''display(Image(filename=str(OUT / 'session_contrast_small_multiples.png')))
display(pd.read_csv(OUT / 'session_macro_contrasts.csv'))
'''),md('''## 附录：逐被试 AUC 与字符准确率
'''),code('''for sid in sorted(subject.subject_id.unique()):
    display(Markdown('### ' + sid))
    display(Image(filename=str(OUT / 'subjects' / (sid + '.png'))))
    g = subject[subject.subject_id.eq(sid)]
    display(g.pivot(index='history_characters', columns='k', values='mean_auc'))
display(macro[['H', 'k', 'mean_character_accuracy']])
display(Image(filename=str(OUT / 'character_accuracy_small_multiples.png')))
'''),md('''## EEGNet 子集计划

`eegnet_subset_plan.csv` 为每个 H 固定种子 0–4 的子集，在不同 k 之间复用，并记录 S1/S2 文件数。H=0 和 H=6 只有一种文件组合。EEGNet 训练按 `docs/eegnet_protocol.md` 进行，epoch 数在训练前固定。
'''),md('''## 复现与输出

运行：`python scripts/run_history_amount_exploratory.py`，可从已完成的模型续跑。独立审计：`python scripts/audit_history_amount_exploratory.py`。重画图：`python scripts/plot_history_amount.py`；重建报告：`python scripts/build_history_amount_notebook.py`。

- `history_file_pool.csv`、`enumeration_manifest.csv`：历史文件、组合及 S1/S2 构成。
- `character_roles.csv`、`validation_flash_manifest.csv`：字符角色与验证 flash。
- `subset_results.csv`、`jobs/`：全部子集的指标、模型、逐字符预测和训练特征行。
- `subject_summary.csv`、`macro_summary.csv`：逐被试与宏平均结果。
- `paired_history_vs_current.csv`、`session_subject_contrasts.csv`、`session_macro_contrasts.csv`：对照与区间。
- `main_endpoint_checks.csv`：与主实验对应端点的一致性检查。
''')]
    n=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'bigp3bci','display_name':'Python (bigp3bci)','language':'python'},'language_info':{'name':'python'}})
    path=ROOT/'notebooks/08_history_amount_exploratory.ipynb';nb.validate(n);nb.write(n,path);print(path)


if __name__=='__main__':main()
