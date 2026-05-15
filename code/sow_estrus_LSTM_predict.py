from lstm_model import EstrusLSTM, EstrusGRU, EstrusLSTM_MultiHeadAttn
from sow_estrus_LSTM_Function import plot_matrix
from sow_estrus_LSTM_Info import result_save_path
from sow_estrus_LSTM_train import EstrusDataset, evaluate_model, load_combined_dataset
from sow_estrus_LSTM_train import __train_info

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

if __name__ == "__main__":
    train_info = __train_info()
    saved_file_path = train_info.saved_file_path
    input_size = train_info.input_size
    layer_hidden_size = train_info.layer_hidden_size
    num_feature = train_info.num_feature
    use_cell_state = train_info.use_cell_state

    # 更改模型存放的对应文件名称
    file_name = "LSTM\\BiLSTM_AddTempRate_2026_0515_1316"
    saved_path = os.path.join(result_save_path, file_name)
    model_saved_path = os.path.join(saved_path, "best_model.pth")

    # 读取测试集
    X_test, y_test = load_combined_dataset(
        os.path.join(saved_file_path, "test.xlsx"), num_features=num_feature
    )
    batch_size = 32

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EstrusLSTM(
        input_size=input_size,
        hidden_size=layer_hidden_size,
        use_cell_state=use_cell_state,
    ).to(device)
    model.load_state_dict(torch.load(model_saved_path))

    # 评估
    criterion = nn.BCELoss()
    record_dict, test_labels, test_preds = evaluate_model(
        model,
        DataLoader(EstrusDataset(X_test, y_test), batch_size),
        criterion,
        device,
    )

    # 绘图
    saved_pictures_path = os.path.join(saved_path, "pictures")
    # saved_pictures_path = os.path.join(saved_path, "again", "pictures")
    os.makedirs(saved_pictures_path, exist_ok=True)

    # 混淆矩阵
    plot_matrix(test_labels, test_preds, save_dir=saved_pictures_path)
