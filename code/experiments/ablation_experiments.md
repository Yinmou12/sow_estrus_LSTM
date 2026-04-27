# 消融实验说明文档

## 一、实验目的

通过消融实验验证数据增强策略中各组件（ADASYN、SMOTE、TomekLinks）对模型性能的贡献，为论文提供实验支撑。

## 二、实验设计

### 实验矩阵

| 实验编号 | 实验名称 | ADASYN | SMOTE | TomekLinks | 目的 |
|---------|---------|--------|-------|------------|------|
| Exp0 | Baseline | ❌ | ❌ | ❌ | 原始数据性能基准 |
| Exp1 | +ADASYN | ✅ | ❌ | ❌ | 验证ADASYN单独贡献 |
| Exp2 | +SMOTE | ❌ | ✅ | ❌ | 验证SMOTE单独贡献 |
| Exp3 | +TomekLinks | ❌ | ❌ | ✅ | 验证TomekLinks单独贡献 |
| Exp4 | ADASYN+SMOTE | ✅ | ✅ | ❌ | 验证组合过采样 |
| Exp5 | ADASYN+TomekLinks | ✅ | ❌ | ✅ | 验证过采样+欠采样 |
| Exp6 | SMOTE+TomekLinks | ❌ | ✅ | ✅ | 验证另一种组合 |
| Exp7 | Full Pipeline | ✅ | ✅ | ✅ | 完整数据增强方案 |

### 实验配置

- **模型**: EstrusLSTM (双向LSTM, 4层)
- **隐藏层大小**: 64
- **Dropout率**: 0.5
- **批大小**: 32
- **学习率**: 0.0005
- **早停耐心**: 7
- **运行次数**: 每个实验运行5次取平均

## 三、文件结构

```
code/experiments/
├── configs/
│   ├── __init__.py
│   └── ablation_configs.py    # 实验配置定义
├── ablation_study.py          # 核心实验代码
├── run_ablation.py            # 批量运行脚本
└── results_analyzer.py        # 结果分析脚本
```

## 四、运行方法

### 1. 查看可用实验

```bash
cd code/experiments
python run_ablation.py --list_experiments
```

### 2. 运行所有消融实验

```bash
python run_ablation.py --data_path <数据路径> --num_runs 5
```

**参数说明**:
- `--data_path`: 训练数据目录（包含train.xlsx, val.xlsx, test.xlsx）
- `--result_dir`: 结果保存目录（默认: result/ablation）
- `--num_runs`: 每个实验运行次数（默认: 5）
- `--num_features`: 特征数量（默认: 2）

### 3. 运行指定实验

```bash
python run_ablation.py --experiments Exp0_Baseline Exp7_FullPipeline
```

### 4. 分析结果

```bash
python results_analyzer.py --result_dir <结果目录>
```

### 5. 快速测试（单次运行）

```bash
python run_ablation.py --num_runs 1
```

## 五、输出结果

### 目录结构

```
result/ablation/ablation_YYYY_MMDD_HHMM/
├── run_config.json                    # 运行配置
├── summary_table.xlsx                 # 汇总表格
├── summary_table.csv                  # CSV格式汇总
├── Exp0_Baseline/
│   ├── experiment_result.json         # 实验详细结果
│   └── best_model_run*.pth           # 最佳模型文件
├── Exp1_ADASYN/
│   └── ...
└── analysis/
    ├── metrics_comparison.png         # 指标对比图
    ├── radar_comparison.png           # 雷达图
    ├── metrics_heatmap.png            # 热力图
    ├── significance_test.xlsx         # 显著性检验
    └── summary_table.tex              # LaTeX表格
```

### 结果指标

每个实验记录以下指标：
- **Accuracy**: 准确率
- **Precision**: 精确率
- **Recall**: 召回率
- **F1**: F1分数
- **AUC**: ROC曲线下面积

所有指标以 **均值±标准差** 形式报告。

## 六、结果解读指南

### 1. 评估数据增强效果

对比Exp0(Baseline)和Exp7(Full Pipeline)：
- 如果Exp7显著优于Exp0，说明数据增强策略有效
- 关注F1和AUC的改善程度

### 2. 分析各组件贡献

对比单项实验（Exp1-Exp3）与Baseline：
- 确定哪个组件贡献最大
- 分析不同组件对不同指标的影响

### 3. 验证组合效果

对比组合实验（Exp4-Exp6）：
- 验证组件是否存在协同效应
- 确定最优组合策略

### 4. 统计显著性

查看`significance_test.xlsx`：
- p值 < 0.05 表示改进具有统计显著性
- 同时关注t检验和Wilcoxon检验结果

## 七、常见问题

### Q1: 运行时间过长怎么办？

减少运行次数：
```bash
python run_ablation.py --num_runs 3
```

### Q2: 内存不足怎么办？

减小批大小或隐藏层大小，修改`ablation_configs.py`中的配置。

### Q3: 如何添加新的实验配置？

在`ablation_configs.py`的`ABLATION_EXPERIMENTS`字典中添加新配置。

### Q4: 如何修改模型参数？

修改`ablation_configs.py`中对应实验的`ModelConfig`。

## 八、后续工作

完成消融实验后，建议：

1. **对比实验**: 与传统ML方法（SVM、Random Forest、XGBoost）对比
2. **模型改进**: 尝试Attention机制、Focal Loss等
3. **可视化分析**: 绘制混淆矩阵、ROC曲线等

## 九、注意事项

1. **数据路径**: 确保数据目录包含正确的文件（train.xlsx, val.xlsx, test.xlsx）
2. **GPU使用**: 代码自动检测GPU，建议使用GPU加速训练
3. **结果复现**: 设置了固定随机种子，但GPU运算可能存在微小差异
4. **存储空间**: 完整实验会产生较多模型文件，注意磁盘空间

---

**文档版本**: 1.0
**最后更新**: 2026-04-26
