## `.py`文件

| 文件名                                                   | 说明                                                         |
| -------------------------------------------------------- | ------------------------------------------------------------ |
| `sow_estrus_LSTM_Info.py`                                | 记录实验中的一些信息                                         |
| `sow_estrus_LSTM_Function.py`                            | 函数声明和定义                                               |
| `correct_abnormal_temperatures.py`                       | 处理异常耳温值的函数，前期测试用的，由于效果不好所以后续实验过程中并未用到，不必关注 |
| `data_preparation.py`                                    | 部分数据预处理操作                                           |
| `lstm_model.py`                                          | LSTM及相关模型构建                                           |
| `sow_estrus_LSTM_train.py`和`sow_estrus_LSTM_predict.py` | 前期构建的模型训练和预测代码                                 |
| `sow_estrus_ablation.py`                                 | ADASYN,SMOTE,Tomek Linked三种方法不同组合的消融实验，后面被`cross_validation.py`替换掉了 |
| `cross_validation.py`                                    | 交叉验证及消融实验，目前想法是按照该文件得到的结果去撰写论文中的"结果"相关部分 |
| `analysis.py`                                            | PCA 和 t-SNE 图的绘制                                        |
| `ablation_smote_tomek_attention.py`                      | 将SMOTE和Tomek Linked组合得到Smote-Tomek，并引入注意力机制观察不同模型下有无Smote-Tomek的效果 |



