from lstm_model import EstrusLSTM, EstrusGRU
from sow_estrus_LSTM_Function import plot_matrix
from sow_estrus_LSTM_Info import result_save_path
from sow_estrus_LSTM_train import EstrusDataset, evaluate_model, load_combined_dataset
from sow_estrus_LSTM_train import layer_hidden_size, saved_file_path

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

if __name__ == "__main__":
    # 更改模型存放的对应文件名称
    file_name = "LSTM\\BiLSTM_2026_0406_1222"
    saved_path = os.path.join(result_save_path, file_name)
    model_saved_path = os.path.join(saved_path, "best_model.pth")

    # 读取测试集
    X_test, y_test = load_combined_dataset(
        os.path.join(saved_file_path, "test.xlsx"), num_features=2
    )
    batch_size = 32
    test_loader = DataLoader(EstrusDataset(X_test, y_test), batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EstrusLSTM(input_size=2, hidden_size=layer_hidden_size).to(device)
    model.load_state_dict(torch.load(model_saved_path))

    # 评估
    criterion = nn.BCELoss()
    _, t_acc, t_precision, t_recall, t_f1, t_auc, test_labels, test_preds = (
        evaluate_model(
            model,
            DataLoader(EstrusDataset(X_test, y_test), batch_size),
            criterion,
            device,
        )
    )
    print("-" * 30)
    print(f"测试集准确率 (Accuracy): {t_acc:.4f}")
    print(f"测试集精确率 (Precision): {t_precision:.4f}")
    print(f"测试集召回率 (Recall): {t_recall:.4f}")
    print(f"测试集 F1 分数: {t_f1:.4f}")
    print(f"测试集 AUC 指标: {t_auc:.4f}")

    # 绘图
    saved_pictures_path = os.path.join(saved_path, "pictures")
    os.makedirs(saved_pictures_path, exist_ok=True)

    # 混淆矩阵
    plot_matrix(test_labels, test_preds, save_dir=saved_pictures_path)
