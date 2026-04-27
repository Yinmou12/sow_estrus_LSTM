# 消融实验代码执行流程详解

本文档详细解释消融实验代码的执行流程、数据流转和关键函数功能。

---

## 一、整体架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                          run_ablation.py                            │
│                      (批量运行入口脚本)                               │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       ablation_configs.py                           │
│                    (实验配置定义模块)                                 │
│                                                                     │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐                   │
│  │DataAugConfig│ │ ModelConfig │ │ExperimentCfg│                   │
│  └─────────────┘ └─────────────┘ └─────────────┘                   │
│                                                                     │
│  ABLATION_EXPERIMENTS = {Exp0, Exp1, ..., Exp7}                    │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        ablation_study.py                            │
│                       (核心实验模块)                                 │
│                                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │数据加载与增强 │→ │  模型训练    │→ │  结果记录    │              │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      results_analyzer.py                            │
│                       (结果分析模块)                                 │
│                                                                     │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐       │
│  │ 汇总表格   │ │ 对比图表   │ │ 统计检验   │ │ LaTeX输出  │       │
│  └────────────┘ └────────────┘ └────────────┘ └────────────┘       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 二、执行流程详解

### 阶段1: 启动与配置加载

```
run_ablation.py main()
       │
       ├── 解析命令行参数
       │   --data_path: 数据目录
       │   --result_dir: 结果目录
       │   --num_runs: 运行次数
       │   --experiments: 指定实验
       │
       ▼
run_all_experiments()
       │
       ├── 创建结果目录 (带时间戳)
       │   result/ablation/ablation_YYYY_MMDD_HHMM/
       │
       ├── 加载实验配置列表
       │   experiment_keys = get_all_experiment_keys()
       │   或使用 --experiments 指定的实验
       │
       └── 保存运行配置 (run_config.json)
```

### 阶段2: 单个实验执行

```
run_experiment(config, data_path, result_dir)
       │
       ├── [Step 1] 创建实验目录
       │   result/ablation/.../Exp0_Baseline/
       │
       ├── [Step 2] 加载原始数据
       │   load_raw_train_data()
       │   ├── train_df: 训练集DataFrame (用于数据增强)
       │   ├── X_val, y_val: 验证集
       │   └── X_test, y_test: 测试集
       │
       └── [Step 3] 多次运行循环 (n=5次)
              │
              └── train_single_run()
```

### 阶段3: 单次训练流程

```
train_single_run(X_train, y_train, X_val, y_val, config, seed, ...)
       │
       ├── [Step 3.1] 设置随机种子
       │   random.seed(seed)
       │   np.random.seed(seed)
       │   torch.manual_seed(seed)
       │
       ├── [Step 3.2] 数据增强 (每次运行重新应用)
       │   apply_data_augmentation(train_df, config.data_augmentation)
       │   │
       │   ├── 判断是否使用 ADASYN → myFunction.ADASYN()
       │   ├── 判断是否使用 SMOTE → myFunction.SMOTE()
       │   └── 判断是否使用 TomekLinks → myFunction.TomekLinked()
       │
       ├── [Step 3.3] 数据准备
       │   prepare_augmented_data(augmented_df)
       │   ├── 提取特征列 (排除ID和标签)
       │   ├── 转换为numpy数组
       │   └── reshape为3D (samples, seq_len, features)
       │
       ├── [Step 3.4] 构建训练组件
       │   ├── build_model() → EstrusLSTM
       │   ├── build_criterion() → BCELoss
       │   ├── build_optimizer() → Adam
       │   └── build_scheduler() → ReduceLROnPlateau
       │
       ├── [Step 3.5] 训练循环
       │   for epoch in range(num_epochs):
       │       ├── 训练阶段: model.train()
       │       │   └── 前向传播 → 计算损失 → 反向传播 → 更新参数
       │       │
       │       ├── 验证阶段: evaluate_model()
       │       │   └── 计算各项指标 (Acc, Prec, Rec, F1, AUC)
       │       │
       │       ├── 学习率调度: scheduler.step(val_loss)
       │       │
       │       └── 早停检查: early_stopping(val_loss, model)
       │           └── 若触发早停, 退出循环
       │
       ├── [Step 3.6] 加载最佳模型
       │   model.load_state_dict(best_model_path)
       │
       └── [Step 3.7] 返回结果
           ├── history: 训练历史
           ├── best_val_metrics: 最佳验证指标
           └── early_stop_epoch: 早停轮次
```

### 阶段4: 结果汇总与保存

```
run_experiment() 续
       │
       ├── [Step 4] 计算平均指标
       │   avg_metrics = {accuracy, precision, recall, f1, auc}
       │   std_metrics = {...}
       │
       ├── [Step 5] 保存实验结果
       │   ├── experiment_result.json (完整结果)
       │   └── best_model_run*.pth (模型文件)
       │
       └── [Step 6] 返回结果字典
```

### 阶段5: 全部实验完成后

```
run_all_experiments() 续
       │
       ├── [Step 5] 生成汇总表
       │   ├── summary_table.xlsx
       │   └── summary_table.csv
       │
       └── [Step 6] 更新运行配置
           run_config.json (添加结束时间和状态)
```

---

## 三、数据流转详解

### 3.1 原始数据结构

```
train.xlsx:
┌────────────┬──────────────┬──────────────┬─────┬───────────────┐
│  sSowsNo   │ temperature_1│ temperature_2│ ... │ label_isEstrus│
├────────────┼──────────────┼──────────────┼─────┼───────────────┤
│  母猪编号   │   温度值1    │   温度值2    │ ... │   0 或 1      │
└────────────┴──────────────┴──────────────┴─────┴───────────────┘

列结构:
- 第1列: sSowsNo (ID列)
- 第2-49列: temperature_1 ~ temperature_48 (48个温度特征)
- 第50-97列: temp_rate_1 ~ temp_rate_48 (48个温度变化率特征) [如果有]
- 第98列: label_isEstrus (标签列)

特征总数: 96列 → reshape为 (samples, 48, 2)
```

### 3.2 数据转换流程

```
原始 train.xlsx (DataFrame)
        │
        │  load_raw_train_data()
        ▼
train_df (DataFrame, 保持原始结构)
        │
        │  apply_data_augmentation()
        ▼
augmented_df (DataFrame, 增加合成样本)
        │
        │  prepare_augmented_data()
        ▼
X_3d: np.ndarray (samples, 48, 2)
y: np.ndarray (samples,)
        │
        │  EstrusDataset()
        ▼
DataLoader (batch_size=32)
```

### 3.3 数据增强执行顺序

```
原始数据
    │
    ├── use_adasyn=True ──→ ADASYN() ──→ 增加少数类合成样本
    │
    ├── use_smote=True ───→ SMOTE() ───→ 进一步过采样
    │
    └── use_tomek_links=True → TomekLinked() → 删除边界多数类样本
    │
    ▼
增强后数据
```

**注意**: ADASYN 和 SMOTE 都是过采样方法，组合使用会导致样本数量显著增加。

---

## 四、关键函数说明

### 4.1 配置相关 (ablation_configs.py)

| 函数/类 | 功能 | 输入 | 输出 |
|--------|------|-----|------|
| `DataAugmentationConfig` | 数据增强配置 | 参数值 | 配置对象 |
| `ModelConfig` | 模型超参数配置 | 参数值 | 配置对象 |
| `ExperimentConfig` | 完整实验配置 | 上述两个配置 | 实验配置对象 |
| `get_experiment_config(key)` | 获取指定配置 | 实验键名 | ExperimentConfig |
| `get_all_experiment_keys()` | 获取所有键名 | 无 | List[str] |

### 4.2 数据处理 (ablation_study.py)

| 函数 | 功能 | 输入 | 输出 |
|------|------|-----|------|
| `load_raw_train_data()` | 加载原始数据 | 数据目录 | train_df, X_val, y_val, X_test, y_test |
| `apply_data_augmentation()` | 应用数据增强 | DataFrame, 配置 | 增强后DataFrame |
| `prepare_augmented_data()` | 准备训练数据 | DataFrame | X_3d, y |
| `load_data_from_excel()` | 从Excel加载 | 文件路径 | X, y |

### 4.3 模型构建 (ablation_study.py)

| 函数 | 功能 | 输入 | 输出 |
|------|------|-----|------|
| `build_model()` | 构建模型 | ModelConfig | nn.Module |
| `build_criterion()` | 构建损失函数 | ModelConfig | Loss函数 |
| `build_optimizer()` | 构建优化器 | model, ModelConfig | Optimizer |
| `build_scheduler()` | 构建学习率调度器 | optimizer, ModelConfig | Scheduler |

### 4.4 训练评估 (ablation_study.py)

| 函数 | 功能 | 输入 | 输出 |
|------|------|-----|------|
| `train_single_run()` | 单次训练 | 数据, 配置, 种子 | 训练结果字典 |
| `evaluate_model()` | 评估模型 | model, loader | 损失和指标 |
| `run_experiment()` | 运行完整实验 | ExperimentConfig | 实验结果字典 |

---

## 五、实验配置详解

### 5.1 8组消融实验配置

```python
ABLATION_EXPERIMENTS = {
    # Exp0: 无数据增强 (Baseline)
    "Exp0_Baseline": ExperimentConfig(
        data_augmentation=DataAugmentationConfig(
            use_adasyn=False,
            use_smote=False,
            use_tomek_links=False,
        ),
    ),

    # Exp1: 仅ADASYN
    "Exp1_ADASYN": ExperimentConfig(
        data_augmentation=DataAugmentationConfig(
            use_adasyn=True,    # ← 开启
            use_smote=False,
            use_tomek_links=False,
        ),
    ),

    # Exp2: 仅SMOTE
    "Exp2_SMOTE": ExperimentConfig(
        data_augmentation=DataAugmentationConfig(
            use_adasyn=False,
            use_smote=True,     # ← 开启
            use_tomek_links=False,
        ),
    ),

    # ... 依此类推 ...
}
```

### 5.2 模型默认配置

```python
ModelConfig(
    model_type="EstrusLSTM",    # 模型类型
    input_size=2,               # 输入特征维度 (温度 + 温度变化率)
    hidden_size=64,             # LSTM隐藏层大小
    num_layers=4,               # LSTM层数
    dropout_rate=0.5,           # Dropout比例

    batch_size=32,              # 批大小
    learning_rate=0.0005,       # 学习率
    weight_decay=1e-4,          # L2正则化
    num_epochs=100,             # 最大训练轮次
    early_patience=7,           # 早停耐心值
)
```

---

## 六、命令行参数详解

### run_ablation.py

```bash
python run_ablation.py [参数]

必需参数:
  --data_path PATH         数据目录路径 (包含train.xlsx, val.xlsx, test.xlsx)

可选参数:
  --result_dir PATH        结果保存目录 (默认: result/ablation)
  --num_runs N             每个实验运行次数 (默认: 5)
  --num_features N         特征维度 (默认: 2)
  --experiments KEY ...    指定运行的实验 (默认: 全部)
  --list_experiments       列出所有实验配置
  --quiet                  减少输出信息
```

### results_analyzer.py

```bash
python results_analyzer.py [参数]

必需参数:
  --result_dir PATH        实验结果目录

可选参数:
  --save_dir PATH          分析结果保存目录 (默认: result_dir/analysis)
```

---

## 七、输出文件结构

```
result/ablation/ablation_YYYY_MMDD_HHMM/
│
├── run_config.json              # 运行配置记录
├── summary_table.xlsx           # 汇总表格 (Excel)
├── summary_table.csv            # 汇总表格 (CSV)
│
├── Exp0_Baseline/               # 实验结果目录
│   ├── experiment_result.json   # 详细结果
│   ├── best_model_run1.pth      # 各次运行的模型
│   ├── best_model_run2.pth
│   └── ...
│
├── Exp1_ADASYN/
│   └── ...
│
├── ...
│
└── Exp7_FullPipeline/
    └── ...

运行 results_analyzer.py 后:

└── analysis/
    ├── metrics_comparison.png   # 指标对比柱状图
    ├── radar_comparison.png     # 雷达图
    ├── metrics_heatmap.png      # 热力图
    ├── significance_test.xlsx   # 统计显著性检验
    └── summary_table.tex        # LaTeX表格
```

---

## 八、常见问题排查

### Q1: 数据维度错误

**错误信息**: `cannot reshape array of size X into shape (Y,Z)`

**原因**:
- 标签列名不匹配 (代码用 `isEstrus`，数据用 `label_isEstrus`)
- 特征列数不是 `seq_len * num_features`

**解决**: 检查 `apply_data_augmentation()` 中的列重命名

### Q2: 模型加载失败

**错误信息**: `RuntimeError: Error loading model`

**原因**:
- PyTorch版本不兼容
- 模型结构已修改

**解决**: 确保使用相同的模型定义

### Q3: 内存不足

**原因**: 数据增强后样本数量过大

**解决**:
- 减少过采样比例 (`smote_amount`)
- 减小 `batch_size`
- 只使用一种过采样方法

---

## 九、代码执行示例

### 完整执行流程

```bash
# 1. 进入实验目录
cd D:\_Software_Projects\VSCode\scientific_research\sow_estrus\my_code\code\experiments

# 2. 查看可用实验
python run_ablation.py --list_experiments

# 3. 快速测试 (每个实验运行1次)
python run_ablation.py --num_runs 1

# 4. 正式运行 (每个实验运行5次)
python run_ablation.py --num_runs 5

# 5. 分析结果
python results_analyzer.py --result_dir ..\..\result\ablation\ablation_YYYY_MMDD_HHMM
```

### 运行指定实验

```bash
# 只运行 baseline 和完整pipeline对比
python run_ablation.py --experiments Exp0_Baseline Exp7_FullPipeline --num_runs 5
```

---

**文档版本**: 1.0
**最后更新**: 2026-04-26





