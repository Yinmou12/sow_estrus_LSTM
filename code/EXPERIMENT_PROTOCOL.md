# 第1套数据实验协议与运行说明

入口：`run_all_experiments.py`。旧实验函数保留用于阅读和历史复现，新 `main()` 委托 `experiment_runner.cli()`，不执行 `record_old_experiment()`。

## 数据与训练

- 默认读取 `data/cross_validation_dataset/1/train_val_df.xlsx` 和 `test.xlsx`，不自动重划外层测试集。
- 当前真实训练/验证：272头、679窗口（246正/433负）；测试：69头、164窗口（60正/104负）。
- 分组按 `sSowsNo`；一个模型样本对应一个 `sSowsNo_split`，严格48个连续整点小时。未完整窗口报错，不能静默丢失。
- 标签沿用已确认的原始母猪类别，温差在各窗口内重算，首小时为0。原数据不覆盖。
- 五折seed54；标准化器只拟合当前折真实训练窗口；增强只作用于训练折。
- 自定义ADASYN：threshold=0.9、gamma=1、k=7。自定义SMOTE在两类内分别合成，800%为额外8倍，含原样本约9倍。A/S/T顺序固定。
- 自定义T保留异类互近邻清理；标准T使用全体样本最近邻、只删除负类端。
- 参考模型：4×64 BiLSTM、学习率5e-4、Dropout0.2、batch32、weight_decay1e-4、不使用cell state。
- CV最多100轮；早停7，ReduceLROnPlateau耐心5、衰减0.5；梯度裁剪1。最终训练采用各验证最佳轮数中位数向上取整、固定学习率、无测试调度。
- 每个配置/种子/折独立初始化。增强seed=seed×1000+fold，网络seed在此基础加10000000。完整批次保留，`drop_last=False`。

## E0—E8

|阶段|设置|种子×折|
|---|---|---|
|E0|数据/组划分审计，短训练验收另用smoke配置|不参与论文统计|
|E1|8种采样组合×单/双通道，倍率800，自定义T|3×5|
|E2|标准T/AT/ST/AST双通道，对照E1|3×5|
|E3|[64]、[64,64]、[64,64,64]、[64,64,64,64]、[64,64,32]|1×5|
|E4|E3最佳结构，额外采样100%至1000%，步长100%|1×5|
|E5|原BO候选加单层/三层等宽、倍率候选；共60次，前12次初始化，首项E4最优|1×5|
|E6|E5前三候选补齐3种子，冻结最优|3×5|
|E7|RNN/LSTM/GRU/BiLSTM均4×64、同参考训练参数、AST及倍率800|3×5|
|E8|E7四配置+调优BiLSTM全量重训并测试|每配置10次|

重复种子：766、929、208。初筛/搜索seed766。最终种子：766、929、208、676、211、443、616、558、59、131。

E2依据AST的跨种子OOF均值选择清理方法，F1/AUC/Recall依次排序，完全平局保留自定义。其他选模也使用同样指标顺序；BO仅反馈F1。E7是统一配置的结构比较，不代表每个基线都单独调优。最优恰好等于参考配置时复用，不能重复计为两个独立训练结果。

主结果阈值0.5。调优BiLSTM用3种子OOF记录联合搜索0.05—0.95、步长0.01的F1最优阈值；平局先最接近0.5，再取较小值。补充阈值不会用于模型排序。所有模型冻结后才进入E8。

## 命令

在代码项目根目录使用已安装的 `machine_learning` 环境，例如PowerShell：

```powershell
& 'E:\_Softwore\Anaconda3\envs\machine_learning\python.exe' -B code/run_all_experiments.py --profile audit
& 'E:\_Softwore\Anaconda3\envs\machine_learning\python.exe' -B code/run_all_experiments.py --profile smoke
& 'E:\_Softwore\Anaconda3\envs\machine_learning\python.exe' -B code/run_all_experiments.py --profile full
```

`audit`仅审计；`smoke`使用无增强和AST配置、5折、最多2轮、seed766，另测试最终导出。其结果有独立目录且不能作为正式性能结论。

正式任务中断后，使用 `--resume` 指定此前 `protocol_v1_full_...` 目录；如之前指定了 `--epochs`，恢复必须相同。`--through E3`等可执行到某阶段；之后在同一运行目录增加 `--through E8` 继续。阶段未完成的模型重新训练，完成的折和配置自动复用。源数据、训练代码、设备或协议变化时拒绝恢复。

`--data-dir`可指定另一套输入目录；`--path-index`默认1；`--result-root`可指定模型/日志目录。预测位置始终由输入目录确定，不跟随结果目录更改。

## 预测导出

逐窗口预测直接写到原输入文件同目录，不含特征列：

```text
train_val_df__oof__CV_{config_key}__seed{seed}__{run_id}.xlsx
test__predictions__FINAL_{config_key}__seed{seed}__{run_id}.xlsx
```

保留 `sEarTagCode`、`sSowsNo`、`sSowsNo_split`（文本型、保留前导零）、存在的 `sBrand/dBreedDate/dWeanDate`、窗口起止及终点时间。附加 `isEstrus/y_prob/y_pred/threshold`、来源文件、数据划分编号、实验编号、运行编号、模型、种子、折和数据集名称。调优BiLSTM另有校准阈值及对应分类列。

一个种子一个OOF表679行，一个最终模型/种子一个测试表164行。概率按窗口键一对一关联；缺失、重复、多余窗口或非法概率均拒绝导出。重名文件增加后缀，不覆盖。

有效配置指纹支持跨阶段复用，因此文件使用 `CV_<key>`；通过阶段结果JSON、`cv/<key>/seed_*/complete.json`、`final/<key>/seed_*/complete.json`可定位配置、模型和预测文件。

## 输出与测试

结果目录包含数据/折清单、完整配置与代码/数据指纹、逐折模型与scaler、损失曲线、采样计数、OOF指标、阶段汇总、冻结配置、最终10种子指标及报告图表。指标包含F1、Accuracy、Precision、Recall、Specificity、ROC-AUC、AP（PR排序指标）、MCC；样本标准差ddof=1。

最终ROC/PR展示全部种子，混淆矩阵汇总所有种子的预测计数，不解释为独立动物数量；不选择最佳测试种子。

```powershell
& 'E:\_Softwore\Anaconda3\envs\machine_learning\python.exe' -B -m unittest discover -s code -p 'test_experiment_*.py' -v
```

已有数据处理测试另执行：`test_fill_data`、`test_processing_feature`、`test_split_estrus_data`、`test_stratified_group_split`。用户既有修改保留，本任务不自动提交或覆盖原始Excel。
