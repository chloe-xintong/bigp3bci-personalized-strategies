"""Report notebook 13 for the exploratory session-wise EA sensitivity analysis."""
from pathlib import Path
import nbformat as nb

ROOT=Path(__file__).resolve().parents[1]


def main():
    md=nb.v4.new_markdown_cell;code=nb.v4.new_code_cell
    cells=[md('''# 13 · 欧氏对齐（EA）敏感性分析

**探索性敏感性分析，只使用 Train 验证，不读取 Test/Static，不改变主设计和已报告结果。** 规则在运行前冻结于 `docs/ea_sensitivity_protocol.md` 与 `configs/ea_sensitivity.json`。

## 方法（位于预处理之后、特征提取之前）

对每个 session，在其允许数据的全部保留 flash 上计算参考协方差 R_s（0–800 ms，逐通道去均值，ddof=1，µV²，算术平均），固定正则 `(1−a)R + a·tr(R)/16·I`，a=0.001；每个试次变换为 R_s^(−1/2)·X。

| Session | 计算 R_s 可用数据 |
|---|---|
| S1 | S1 Train 全部 18 字符（历史） |
| S2 | S2 Train 全部 18 字符（历史） |
| S3 | 校准池前 k 个字符 |

S3 最后 6 个验证字符、Test/Static 与未标注当前数据一律不进入 R。k=0 时 S3 无可用数据，因此只报告 k=3、6、12。窗口均值与通道线性变换可交换，所以 EA 在缓存特征上实现，并逐被试用原始 epoch 验证了严格等价（最大误差约 1e−14 µV）。

对照：**A** 历史＋当前不对齐（主设计）；**B** 历史＋当前 session-wise EA；**仅当前** 不对齐；**C** 仅当前 EA。全部历史（H=6，36 字符），SWLDA 固定主实验设置。A 与仅当前逐被试精确复现了 notebook 12 的结果。
'''),code('''from pathlib import Path
import json
import pandas as pd
from IPython.display import display, Image, Markdown
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'configs/ea_sensitivity.json').exists())
OUT = ROOT / 'outputs/ea_sensitivity_v1'
if not (OUT / 'completion.json').exists():
    raise FileNotFoundError('请先运行 scripts/run_ea_sensitivity.py；本 notebook 不启动拟合。')
completion = json.loads((OUT / 'completion.json').read_text())
display(pd.Series(completion).to_frame('audit'))
'''),md('''## 结果：宏平均差值（按被试配对 bootstrap，10,000 次）

`ea_with_history` = B − A；`ea_current_only` = C − 仅当前。方向规则（运行前写定）：至少 8/10 名被试与宏平均同号才称“方向一致”，否则按被试报告、不用平均值下结论。EEGNet 跟进条件：某 k 下 B−A 区间不含 0、方向一致且 |B−A| ≥ 0.01。
'''),code('''macro = pd.read_csv(OUT / 'macro_deltas.csv')
display(macro.round(4))
'''),md('''## 各条件 AUC（10 名被试＋宏平均）

A（不对齐，主设计）为蓝色 `#0072B2`，B（session-wise EA）为朱红 `#D55E00`，C（仅当前＋EA）为灰色虚线；不对齐的仅当前为浅灰点线参考，直接标注、不进图例。纵轴固定 0.4–1.0。宏平均只给 A 画填充区间（15%），B 的区间用细虚线边界表示，避免两层填充叠色。
'''),code('''display(Image(filename=str(OUT / 'ea_condition_auc.png')))
'''),md('''## 按被试差值（F_07、F_21 为漂移不对称最明显的被试）

k 为有序变量，沿用 notebook 12 的紫色深浅（k=3 浅、k=12 深）。
'''),code('''display(Image(filename=str(OUT / 'ea_subject_deltas.png')))
deltas = pd.read_csv(OUT / 'subject_deltas.csv')
display(deltas.pivot(index='subject_id', columns='k', values='ea_with_history').round(3).add_prefix('B-A k='))
display(deltas.pivot(index='subject_id', columns='k', values='ea_current_only').round(3).add_prefix('C-cur k='))
'''),md('''## 结论（一句话）

在 SWLDA 上，session-wise EA 对“历史＋当前”的宏平均影响很小（B−A 在 k=3、6、12 为 +0.001、+0.005、+0.007 AUC，区间均含 0），且各 k 下被试方向不一致（5–6/10 为正）；唯一明显且稳定的改善是漂移最大的 F_07（+0.04 至 +0.07），因此只按被试报告，不改变主设计，也不触发 EEGNet 跟进。

补充：仅当前的 EA（C−仅当前）同样方向不一致，说明 EA 本身并非普遍有益；F_07 在 B−A 上的改善随 k 增大而增大，而在 C−仅当前上 k=12 时消失，提示其收益主要来自把 S1 的历史数据对齐到 S3，而不是来自 EA 本身。
'''),md('''## 诊断：参考协方差稳定性

`s3k_to_s3_12_normalized`：R_S3(k) 与 R_S3(12) 的归一化 log-Euclidean 距离；k=3 时 R 只基于约 356 个保留 flash（378 个减去保护期排除），距离普遍大于 k=6，估计较不稳定。`s1_to_s3k`/`s2_to_s3k` 为对齐前的 session 漂移。
'''),code('''display(pd.read_csv(OUT / 'reference_stability.csv').round(3))
display(pd.read_csv(OUT / 'reference_covariances.csv').round(1))
'''),md('''## 限制

- k=3 时 R_S3 估计可能不稳定（见上表）。
- EA 是线性变换；SWLDA 随后在合并训练集上逐特征 z-score。z-score 是全局的，不消除 session 间差异；EA 在合并前逐 session 统一尺度并混合通道，因此 SWLDA 实际看到的特征尺度和逐步选入的特征都会改变，B−A 反映的是两者的组合效应。
- 只检验 session-wise EA；黎曼对齐等变体不在范围内。
- 10 名被试、每人 6 个验证字符；区间为探索性描述，不作确认性显著性结论。字符准确率见 `condition_results.csv`，只作描述。

复现：`python scripts/run_ea_sensitivity.py`；重绘：`python scripts/plot_ea_sensitivity.py`；重建报告：`python scripts/build_ea_sensitivity_notebook.py`。
''')]
    n=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'bigp3bci','display_name':'Python (bigp3bci)','language':'python'},'language_info':{'name':'python'}})
    path=ROOT/'notebooks/13_ea_sensitivity.ipynb';nb.validate(n);nb.write(n,path);print(path)


if __name__=='__main__':main()
