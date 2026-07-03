马修斯相关系数(MCC)


$$
MCC = \frac{TP \times TN - FP \times FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}
$$




有，而且建议你不要只堆表格。你现在的实验体系其实很适合做成“表格 + 趋势图 + 稳定性分析 + 可解释性分析”的组合。论文里这样会更完整。

我建议优先考虑下面这些。

**1. 实验整体流程图**

这张图非常值得做。可以放在方法部分，展示完整流程：

```
原始耳温数据
-> 缺失填充 / 48h窗口构建
-> 固定 train_val / independent_test 划分
-> 训练折内数据增强 AST
-> 构造耳温 + 变化率特征
-> Bi-LSTM / RNN / GRU
-> 验证集动态阈值
-> 独立测试集评估
```

这张图不是结果图，但能让审稿人快速理解你的实验设计，尤其是你固定独立测试集、只在训练折做数据增强，这些都是方法严谨性的体现。

**2. Bi-LSTM 模型结构图**

你前面已经理解了数据流，可以画成：

```
48 × 2 输入序列
-> BiLSTM layer 1
-> BiLSTM layer 2
-> BiLSTM layer 3/4
-> hidden state concat
-> FC
-> sigmoid
-> estrus probability
```

重点标注：

```
input_size = 2
hidden_sizes = [64,64,32] 或 [64,64,64,64]
bidirectional = True
dynamic threshold
```

这比只写文字清楚很多。

**3. 贝叶斯优化收敛曲线**

如果你用了 `experiment_7`，可以画：

```
x轴: Bayesian iteration
y轴: validation F1-Score
```

再加一条：

```
best-so-far F1
```

这张图可以说明：

```
贝叶斯优化是否逐渐找到更优结构
第几轮后性能趋于稳定
继续搜索是否还有明显收益
```

比单纯给“最优超参数表”更有说服力。



```
采取画：1. 每一轮 BO 实际得到的 Validation F12. 截止当前轮的 best-so-far Validation F1；但是不用更粗的折现，一样粗细就好，换个颜色。
```

**4. 超参数影响图**

从 `final_summary.xlsx` 里可以画：

```
hidden_sizes vs F1
learning_rate vs F1
dropout_rate vs F1
batch_size vs F1
weight_decay vs F1
```

形式可以是：

```
箱线图
散点图
热力图
parallel coordinate plot
```

尤其是：

```
hidden_sizes + learning_rate -> F1 热力图
```

很适合解释为什么 `[64,64,32]` 比 `[64,64,64,64]` 好。

**5. 动态阈值对比图**

你已经在比较 `dynamic_threshold=True/False`，不要只放表格。建议画：

```
固定阈值 0.5 vs 动态阈值
```

指标可以是：

```
F1
Recall
Precision
Specificity
MCC
```

用 grouped bar plot 或者 boxplot。

另外还可以画每折搜索到的阈值分布：

```
Fold 1 threshold
Fold 2 threshold
...
```

这能说明模型最优阈值是否明显偏离 0.5。如果大多数阈值集中在 0.2-0.4，说明类别不平衡下固定 0.5 确实不合适。

**6. ROC 曲线和 PR 曲线**

类别不平衡任务里，PR 曲线比 ROC 更有信息量。

建议至少放：

```
ROC curve
Precision-Recall curve
```

对比对象可以是：

```
Bi-LSTM
LSTM
GRU
RNN
```

或者：

```
固定阈值 BiLSTM
动态阈值 BiLSTM
```

如果篇幅有限，优先放 PR 曲线。

不过这个需要保存每个样本的预测概率。你当前代码主要保存 metric，如果没有保存 `y_true` 和 `y_prob`，后面需要加一个 `predictions.xlsx`。

**7. 混淆矩阵图**

对独立测试集画 normalized confusion matrix：

```
True Negative / False Positive
False Negative / True Positive
```

这张图非常适合解释模型实际错误类型。

对发情检测来说，尤其要关注：

```
False Negative：发情但没识别出来
False Positive：非发情但误报
```

这比 Accuracy 更直观。

**8. 多随机种子稳定性图**

你之前已经设计了 `REPEATED_RUNS` 和随机种子记录。这个非常适合做稳定性分析。

可以画：

```
不同实验设置下 F1 的箱线图
不同实验设置下 Recall 的箱线图
不同实验设置下 MCC 的箱线图
```

这样可以说明：

```
某个方法不是偶然一次跑得好，而是在多次随机划分/初始化下稳定更好
```

论文里这个比单次最优结果更有价值。

**9. 数据增强消融图**

你的实验 2 和实验 4 都适合画消融图：

```
Baseline
A
S
T
AS
AT
ST
AST
```

可以画 grouped bar：

```
x轴: 数据处理方法
y轴: F1 / Recall / MCC
```

建议不要只画 Accuracy，因为类别不平衡下 Accuracy 容易误导。

更推荐：

```
F1
Recall
MCC
AUC
```

**10. SMOTE 比例趋势图**

实验 5 很适合画折线图：

```
x轴: SMOTE ratio, 3x 到 10x
y轴: F1 / Recall / Precision / MCC
```

这张图可以回答一个很实际的问题：

```
过采样越多是否越好？
```

如果曲线在某个比例后下降，可以讨论：

```
过度过采样可能引入重复/边界样本噪声，导致泛化下降
```

**11. 时间步敏感性分析**

这个比较有亮点。可以做一个简单的 occlusion analysis：

```
每次遮挡某一个小时的输入
观察预测概率下降多少
```

最后画成：

```
x轴: 1-48 小时
y轴: 预测概率变化
```

如果某些时间段对发情预测特别重要，这张图可以作为模型可解释性分析。

也可以分别遮挡：

```
耳温通道
变化率通道
```

看哪个通道更关键。

这比单纯说“使用耳温变化率提升了效果”更有说服力。

**12. 预测概率分布图**

画正类和负类在测试集上的预测概率分布：

```
发情样本概率分布
非发情样本概率分布
```

如果两个分布分离明显，说明模型判别能力好。

如果重叠很大，可以解释为什么 Precision/Recall 难以同时提高。

**推荐论文组合**

如果你不想做太多，我建议至少加这 6 类图：

```
1. 整体实验流程图
2. Bi-LSTM 结构图
3. 数据增强消融柱状图
4. 贝叶斯优化收敛曲线
5. 固定阈值 vs 动态阈值对比图
6. 独立测试集 PR 曲线或混淆矩阵
```

如果想进一步提高论文质量，再加：

```
7. 多随机种子箱线图
8. 时间步敏感性分析
```

这两个会明显增强结果可信度和可解释性。
