# EEGNet 协议

本协议在任何 EEGNet 训练之前写定。下列“已确定”的条目与 SWLDA 实验保持一致；“训练前确定”的条目必须在第一次训练前填写并提交，提交后不再根据验证结果修改。

## 已确定

| 项目 | 规定 |
|---|---|
| 数据 | 只用 Train 文件（`bigp3bci_data_policy.require_train_edf`）；Test、Static、Dyn、DynBigram 不读取 |
| 划分 | 与 SWLDA 相同：SE003 Train 前 12 个字符为校准池，最后 6 个字符为固定验证集；SE001、SE002 全部 Train 为历史数据 |
| 预处理 | 与 SWLDA 相同的因果滤波、0–800 ms epoch、A 质量规则；被排除的 flash 不给分数 |
| 输入通道 | 预先固定的 16 个通道 |
| 训练方案 | 仅当前、历史 S1+S2＋当前，k∈{0,3,6,12}；历史子集按 `outputs/history_amount_exploratory_v1/eegnet_subset_plan.csv`（每个 H 种子 0–4） |
| 标准化 | 只用该模型的训练数据计算 |
| 类别加权 | 训练权重由 `bigp3bci_probability.binary_class_weights(training_labels)` 计算，只用该模型的训练标签 |
| 主指标 | 验证集 flash ROC-AUC，逐被试计算后宏平均；同时报告类别加权 flash log-loss |
| 停止点 | 用 `FlashLogLossEarlyStopping` 按验证集加权 log-loss 选 checkpoint，同时记录 AUC；字符准确率不参与选择 |
| 评估 | `model.eval()`，关闭梯度 |
| 统计 | 按被试 bootstrap 10,000 次，种子 20261126，百分位 95% 区间；比较为被试内配对差值 |

## 训练前确定

| 项目 | 取值 |
|---|---|
| 网络结构与超参数（F1、D、F2、卷积核长度、dropout） | |
| 优化器、学习率、batch size | |
| 最大 epoch 数与 patience | |
| 随机种子（网络初始化） | |
| 历史数据量探索（notebook 08 计划）是否采用固定 epoch 数而不早停；若是，epoch 数 | |

## 输出

每次运行在 `outputs/` 下新建目录，保存配置快照、来源哈希、逐 flash 分数、逐被试指标和选中的 checkpoint，并在 `docs/experiment_log.md` 追加一行。
