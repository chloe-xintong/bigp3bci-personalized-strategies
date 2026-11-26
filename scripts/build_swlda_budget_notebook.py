"""Notebook 07: Train holdout calibration curves, flash AUC as primary metric."""
from pathlib import Path
import nbformat as nb

ROOT=Path(__file__).resolve().parents[1]


def main():
    md=nb.v4.new_markdown_cell;code=nb.v4.new_code_cell
    cells=[md('''# 07 · SWLDA 校准预算：Train 留出验证与 flash ROC-AUC

本实验只使用 Train 文件。

每名被试的 SE003 Train 共 18 个字符，按采集顺序前 12 个作为校准池，最后 6 个固定为验证集，因此 k 取 0–12。SE001、SE002 的全部 Train 作为历史数据。特征为预先固定的 16 通道、256 维窗口均值，质量规则为 A（基础故障）。标准化和 SWLDA 特征筛选只使用对应预算的训练 epoch。

四条曲线：仅当前、历史 S1+S2＋当前、历史 S1＋当前、历史 S2＋当前。后三者分别使用 36、18、18 个历史字符，再加当前前 k 个字符。仅当前在 k=0 时没有模型。全部数据参考使用历史 36＋当前 12＝48 个字符，作为数据最充分时的对照。

主指标为 **flash 级 ROC-AUC**：先逐被试计算，再对被试取宏平均。所有方案和 k 使用相同的验证 flash 与质量掩码。另报告训练内概率校准后的类别加权 log-loss，字符准确率作为辅助结果。相邻 epoch 有重叠、同一字符内的 flash 相关，因此区间按被试计算。
'''),code('''from pathlib import Path
import json
import pandas as pd
from IPython.display import display, Image
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents]
            if (p / 'configs/swlda_calibration_budget.json').exists())
OUT = ROOT / 'outputs/swlda_calibration_budget_train_validation_v1'
if not (OUT / 'completion.json').exists():
    raise FileNotFoundError('请先运行 scripts/run_swlda_calibration_budget.py。')
overall = pd.read_csv(OUT / 'overall_results.csv')
subjects = pd.read_csv(OUT / 'subject_results.csv')
LOSS_OUT = OUT / 'weighted_logloss'
has_logloss = (LOSS_OUT / 'completion.json').exists()
if has_logloss:
    overall = overall.merge(pd.read_csv(LOSS_OUT / 'overall_metrics.csv')[
        ['scheme', 'k', 'macro_weighted_log_loss', 'sd_subject_weighted_log_loss']],
        on=['scheme', 'k'], validate='one_to_one')
    subjects = subjects.merge(pd.read_csv(LOSS_OUT / 'subject_metrics.csv')[
        ['subject_id', 'scheme', 'k', 'weighted_log_loss', 'probability_calibration']],
        on=['subject_id', 'scheme', 'k'], validate='one_to_one')
display(pd.Series(json.loads((OUT / 'completion.json').read_text())).to_frame('run'))
'''),md('''## 主结果：flash 级 ROC-AUC

先给 10 名被试的 2×5 小图和宏平均面板，纵轴统一为 0.4–1.0；宏平均阴影为按被试 bootstrap（10,000 次）的 95% 区间，画主设计和仅当前两层。配色：仅当前为灰色细线，历史 S1+S2＋当前（主设计）为最粗的蓝色，历史 S1＋当前为橙色，历史 S2＋当前为蓝绿色；仅历史与全部数据参考为深灰虚线和点线，标注在线的右侧。之后是宏平均总览和放大图。
'''),code('''display(overall[['scheme', 'k', 'available_models', 'macro_flash_roc_auc',
                 'sd_subject_flash_roc_auc', 'validation_flashes', 'scored_validation_flashes']])
display(Image(filename=str(OUT / 'auc_small_multiples.png')))
display(Image(filename=str(OUT / 'calibration_curves.png')))
'''),md('''## 类别加权 flash log-loss（越低越好）

验证集保留 815 个靶标、6,525 个非靶标，约 1:8。每名被试分别计算：

`weighted log-loss = -0.5 × mean_target(log p) - 0.5 × mean_nontarget(log(1-p))`

两类各占一半权重，再对被试宏平均。概率截断为 [1e-7, 1-1e-7]；常数预测 p=0.5 的损失为 ln 2 ≈ 0.693。

概率层为 Platt sigmoid：在模型允许的训练字符内，按完整字符分三折交叉拟合，用未参与该折拟合的训练 flash 分数拟合 sigmoid。sigmoid 拟合保留自然训练先验，类别加权只用于评价。SWLDA 分数、AUC 和字符预测保持不变。仅当前 k=1 只有一个训练字符，改用训练内分数拟合，结果表中标记为 `training_in_sample_single_character`。
'''),code('''if not has_logloss:
    print('请先运行 scripts/add_weighted_logloss.py；本格不会自动拟合概率层。')
else:
    display(overall[['scheme', 'k', 'macro_weighted_log_loss', 'sd_subject_weighted_log_loss']])
    display(Image(filename=str(LOSS_OUT / 'logloss_small_multiples_zoom.png')))
    display(Image(filename=str(LOSS_OUT / 'logloss_small_multiples.png')))
    display(Image(filename=str(LOSS_OUT / 'weighted_logloss_curves.png')))
    display(subjects[subjects.probability_calibration.eq('training_in_sample_single_character')][
        ['subject_id', 'scheme', 'k', 'weighted_log_loss', 'probability_calibration']])
'''),md('''## 端点检查

仅历史端点与历史＋当前曲线的 k=0、仅当前最大端点与 k=12、全部数据参考与历史 S1+S2＋当前的 k=12，均独立重训并核对权重和预测。
'''),code('''endpoint_names = ['history_only', 'history_se001_only', 'history_se002_only',
                  'current_max_endpoint', 'all_available_reference']
display(overall[overall.scheme.isin(endpoint_names)][
    ['scheme', 'k', 'macro_flash_roc_auc', 'correct_characters', 'validation_characters', 'pooled_accuracy']])
display(pd.read_csv(OUT / 'endpoint_checks.csv'))
display(subjects[subjects.scheme.isin(endpoint_names)].pivot(
    index='subject_id', columns='scheme', values='flash_roc_auc'))
'''),md('''## 辅助结果：字符准确率

每名被试有 6 个验证字符。EEGNet 的停止点用 flash 级指标确定。
'''),code('''display(Image(filename=str(OUT / 'character_accuracy_small_multiples.png')))
display(Image(filename=str(OUT / 'character_accuracy_curves.png')))
display(overall[['scheme', 'k', 'correct_characters', 'validation_characters', 'pooled_accuracy']])
'''),md('''## EEGNet 的停止点

以上结果均为 SWLDA。`src/bigp3bci_flash_early_stopping.py` 提供 `FlashAUCEarlyStopping`（越大越好）和 `FlashLogLossEarlyStopping`（类别加权 log-loss，越小越好）。训练前固定一种规则：用加权 log-loss 选停止点，同时记录 AUC。评估时使用 `model.eval()` 并关闭梯度。

```python
stopper = FlashLogLossEarlyStopping(patience=15)
# 在 EEGNet 训练循环中：
# decision = stopper.update(validation_flash_labels, validation_positive_probabilities, epoch)
# if decision['improved']:
#     best_state = copy.deepcopy(model.state_dict())
# if decision['should_stop']:
#     break
# 训练结束后：model.load_state_dict(best_state)
```

训练加权可用 `bigp3bci_probability.binary_class_weights(training_labels)`，得到 `N/(2N_class)` 的权重，只从该模型预算内的训练标签计算。patience=15 为接口默认值。
'''),md('''## 复现

运行 `python scripts/run_swlda_calibration_budget.py`（已有输出时用 `--output` 指定新目录），配置为 `configs/swlda_calibration_budget.json`。补算概率层和加权 log-loss：`python scripts/add_weighted_logloss.py`，结果在 `weighted_logloss/`。重画图：`python scripts/plot_swlda_calibration_budget.py`；重建报告：`python scripts/build_swlda_budget_notebook.py`。

- `flash_predictions.csv`：每个模型的逐 flash 标签、分数和保留标记。
- `subject_results.csv`、`overall_results.csv`：逐被试 AUC 和宏平均曲线。
- `character_roles.csv`、`training_character_membership.csv`、`fixed_validation_characters.csv`、`budget_leakage_audit.csv`：划分与预算核查。
''')]
    n=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'bigp3bci','display_name':'Python (bigp3bci)','language':'python'},'language_info':{'name':'python'}})
    path=ROOT/'notebooks/07_swlda_calibration_budget.ipynb';nb.validate(n);nb.write(n,path);print(path)


if __name__=='__main__':main()
