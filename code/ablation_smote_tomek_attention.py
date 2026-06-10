"""
Run three ablation experiments on unbalanced filled data:

1. No data augmentation + layer-wise BiLSTM
2. SMOTE-Tomek data augmentation + layer-wise BiLSTM
3. SMOTE-Tomek data augmentation + layer-wise BiLSTM with temporal attention
"""

import argparse
import itertools
import json
import os
import random
import sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import sow_estrus_LSTM_Function as myFunction
from lstm_model import (
    EarlyStopping,
    EstrusLSTM,
    EstrusLSTM_Attn,
    EstrusLSTM_DotProductAttn,
    EstrusLSTM_ScaledDotProductAttn,
    EstrusLSTM_MultiHeadAttn,
)
from sow_estrus_LSTM_Info import experimentRecord_data_path, result_save_path
from sow_estrus_LSTM_train import EstrusDataset, __train_info as TrainInfo
from sow_estrus_LSTM_train import evaluate_model

pd.set_option("future.no_silent_downcasting", True)

# 定义实验配置：特征组合、数据增强组合与注意力机制组合
EXPERIMENTS = []
for use_rate in [False, True]:
    feat_prefix = "MultiFeat" if use_rate else "SingleFeat"
    for bidirectional in [True, False]:
        bi_prefix = "Bi" if bidirectional else "Uni"
        for aug_flag, aug_name in [(False, "No_Aug"), (True, "SMOTE_Tomek")]:
            # 包含无注意力(None)和四种注意力机制
            for attn_type in [None, "Temporal", "Dot", "ScaledDot", "MultiHead"]:
                name_suffix = f"_{attn_type}Attn" if attn_type else ""
                EXPERIMENTS.append(
                    {
                        "name": f"{feat_prefix}_{bi_prefix}_{aug_name}{name_suffix}",
                        "description": f"{feat_prefix} ({'Temp+Rate' if use_rate else 'Temp Only'}), {bi_prefix}directional, {aug_name} with {attn_type or 'No'} Attention",
                        "use_smote_tomek": aug_flag,
                        "use_attention": attn_type is not None,
                        "attn_type": attn_type,
                        "use_rate": use_rate,
                        "bidirectional": bidirectional,
                    }
                )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def parse_hidden_sizes(value):
    hidden_sizes = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not hidden_sizes:
        raise ValueError("hidden_sizes cannot be empty")
    return hidden_sizes


def add_temperature_rate_features(flat_df):
    temp_features = flat_df.iloc[:, 1:-1].copy().astype(float)
    rate_features = temp_features.diff(axis=1).fillna(0)
    rate_features.columns = [f"rate_{i}" for i in range(1, rate_features.shape[1] + 1)]
    return pd.concat([flat_df.iloc[:, :-1], rate_features, flat_df.iloc[:, -1]], axis=1)


def make_feature_frame(filled_df):
    flat_df = myFunction.convert_features(filled_df)
    return add_temperature_rate_features(flat_df)


def apply_smote_tomek(train_flat_df, smote_amount, smote_k, tomek_k):
    augmented_df = myFunction.SMOTE(
        train_flat_df,
        amount_oversampling=smote_amount,
        k=smote_k,
    )
    return myFunction.TomekLinked(augmented_df, k=tomek_k)


def build_model(exp_config, train_info, hidden_sizes, device):
    # 根据配置决定输入维度 (1: 仅耳温, 2: 耳温+变化率)
    input_size = 2 if exp_config.get("use_rate", True) else 1
    bidirectional = exp_config.get("bidirectional", True)
    current_hidden_sizes = exp_config.get("hidden_sizes", hidden_sizes)

    common_kwargs = {
        "input_size": input_size,
        "hidden_sizes": current_hidden_sizes,
        "num_layers": len(current_hidden_sizes),
        "dropout_rate": train_info.dropout_rate,
        "bidirectional": bidirectional,
    }

    # 注意力模型映射表
    ATTN_MODELS = {
        "Temporal": EstrusLSTM_Attn,
        "Dot": EstrusLSTM_DotProductAttn,
        "ScaledDot": EstrusLSTM_ScaledDotProductAttn,
        "MultiHead": EstrusLSTM_MultiHeadAttn,
    }

    attn_type = exp_config.get("attn_type")
    if exp_config["use_attention"] and attn_type in ATTN_MODELS:
        model_class = ATTN_MODELS[attn_type]
        if attn_type == "MultiHead" and "num_heads" in exp_config:
            common_kwargs["num_heads"] = exp_config["num_heads"]
        return model_class(**common_kwargs).to(device)

    # 默认模型
    return EstrusLSTM(
        **common_kwargs,
        use_cell_state=train_info.use_cell_state,
    ).to(device)


def load_state_dict(model, model_path, device):
    try:
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)


def train_one_fold(
    X_train,
    y_train,
    X_val,
    y_val,
    exp_config,
    train_info,
    hidden_sizes,
    save_dir,
    fold_idx,
    seed,
):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(exp_config, train_info, hidden_sizes, device)

    criterion = nn.BCELoss()

    # 提取当前超参数，支持网格搜索动态调整
    current_lr = exp_config.get("learning_rate", train_info.learning_rate)
    current_bs = exp_config.get("batch_size", train_info.batch_size)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=current_lr,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=train_info.lr_patience,
    )

    batch_size = min(current_bs, max(1, len(y_train)))
    train_loader = DataLoader(
        EstrusDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        drop_last=len(y_train) >= batch_size,
    )
    val_loader = DataLoader(
        EstrusDataset(X_val, y_val),
        batch_size=current_bs,
        shuffle=False,
    )

    best_model_path = os.path.join(save_dir, f"best_model_fold{fold_idx}.pth")
    early_stopping = EarlyStopping(
        patience=train_info.early_patience,
        verbose=False,
        path=best_model_path,
    )

    history = []
    for epoch in range(train_info.num_epochs):
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        val_metrics, _, _ = evaluate_model(model, val_loader, criterion, device)
        avg_train_loss = train_loss / max(1, len(train_loader))
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                **val_metrics,
            }
        )

        scheduler.step(val_metrics["avg_loss"])
        early_stopping(val_metrics["avg_loss"], model)
        if early_stopping.early_stop:
            break

    load_state_dict(model, best_model_path, device)
    final_val_metrics, _, _ = evaluate_model(model, val_loader, criterion, device)
    final_val_metrics.pop("avg_loss", None)

    history_path = os.path.join(save_dir, f"history_fold{fold_idx}.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    return model, criterion, device, final_val_metrics


def evaluate_with_model(model, criterion, device, X, y, batch_size):
    loader = DataLoader(EstrusDataset(X, y), batch_size=batch_size, shuffle=False)
    metrics, _, _ = evaluate_model(model, loader, criterion, device)
    metrics.pop("avg_loss", None)
    return metrics


def average_metrics(records):
    return {
        key: float(np.mean([record[key] for record in records]))
        for key in records[0].keys()
    }


def run_ablation(
    df,
    experiment_list,
    n_splits=5,
    test_ratio=0.2,
    stride=6,
    hidden_sizes=None,
    smote_amount=800,
    smote_k=7,
    tomek_k=1,
    random_state=123,
    evaluate_independent_test=True,
):
    train_info = TrainInfo()
    if hidden_sizes is None:
        hidden_sizes = [train_info.layer_hidden_size] * 4

    # 保存路径
    timestamp = datetime.now().strftime("%Y_%m%d_%H%M")
    save_root = os.path.join(
        result_save_path,
        "cv_ablation_smote_tomek_attention",
        timestamp,
    )
    os.makedirs(save_root, exist_ok=True)

    # 划分独立测试集和交叉验证折
    independent_test_df, folds = myFunction.stratified_group_kfold(
        df,
        n_splits=n_splits,
        test_ratio=test_ratio,
        random_state=random_state,
    )

    # 保存独立测试集划分结果
    independent_test_df.to_excel(
        os.path.join(save_root, "independent_test_df.xlsx"),
        index=False,
    )

    # 是否进行测试集评估
    if evaluate_independent_test:
        test_filled = myFunction.fill_data(
            independent_test_df,
            balanced_data=True,
            stride=stride,
        )
    # 不在此处预处理 test_flat，因为特征维度随实验变化

    final_val_summary = {}
    final_test_summary = {}
    details = []

    # 交叉验证训练和评估所选的实验配置
    for exp_index, exp_config in enumerate(experiment_list):
        exp_name = exp_config["name"]
        use_rate = exp_config.get("use_rate", True)
        exp_dir = os.path.join(save_root, exp_name)
        os.makedirs(exp_dir, exist_ok=True)
        print(f"\n{'=' * 60} {exp_name} {'=' * 60}")

        fold_val_metrics = []
        fold_test_metrics = []

        for fold_idx, (train_df_raw, val_df_raw) in enumerate(folds, start=1):
            print(f"\n--- {exp_name} | Fold {fold_idx} ---")
            fold_seed = random_state + exp_index * 10 + fold_idx

            train_filled = myFunction.fill_data(
                train_df_raw,
                balanced_data=True,
                stride=stride,
            )
            val_filled = myFunction.fill_data(
                val_df_raw,
                balanced_data=True,
                stride=stride,
            )

            train_flat = myFunction.convert_features(train_filled)
            if exp_config["use_smote_tomek"]:
                train_flat = apply_smote_tomek(
                    train_flat,
                    smote_amount=smote_amount,
                    smote_k=smote_k,
                    tomek_k=tomek_k,
                )

            if use_rate:
                train_final = add_temperature_rate_features(train_flat)
                val_final = make_feature_frame(val_filled)
            else:
                train_final = train_flat
                val_final = myFunction.convert_features(val_filled)

            X_train, y_train, scaler = myFunction.prepare_lstm_data(train_final)
            X_val, y_val, _ = myFunction.prepare_lstm_data(val_final, scaler=scaler)

            scaler_path = os.path.join(exp_dir, f"scaler_fold{fold_idx}.joblib")
            joblib.dump(scaler, scaler_path)

            print(
                f"Train X={X_train.shape}, positives={int((y_train == 1).sum())}, "
                f"negatives={int((y_train == 0).sum())}"
            )
            print(
                f"Val   X={X_val.shape}, positives={int((y_val == 1).sum())}, "
                f"negatives={int((y_val == 0).sum())}"
            )

            model, criterion, device, val_metrics = train_one_fold(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                exp_config=exp_config,
                train_info=train_info,
                hidden_sizes=hidden_sizes,
                save_dir=exp_dir,
                fold_idx=fold_idx,
                seed=fold_seed,
            )
            fold_val_metrics.append(val_metrics)
            details.append(
                {
                    "Experiment": exp_name,
                    "Dataset": "Validation",
                    "Fold": f"Fold_{fold_idx}",
                    **val_metrics,
                }
            )
            print(f"Validation metrics: {val_metrics}")

            # 评估独立测试集
            if evaluate_independent_test:
                if use_rate:
                    test_final = make_feature_frame(test_filled)
                else:
                    test_final = myFunction.convert_features(test_filled)

                X_test, y_test, _ = myFunction.prepare_lstm_data(
                    test_final,
                    scaler=scaler,
                )
                print(
                    f"Test X={X_test.shape}, positives={int((y_test == 1).sum())}, "
                    f"negatives={int((y_test == 0).sum())}"
                )
                test_metrics = evaluate_with_model(
                    model,
                    criterion,
                    device,
                    X_test,
                    y_test,
                    exp_config.get("batch_size", train_info.batch_size),
                )
                fold_test_metrics.append(test_metrics)
                details.append(
                    {
                        "Experiment": exp_name,
                        "Dataset": "Independent_Test",
                        "Fold": f"Fold_{fold_idx}",
                        **test_metrics,
                    }
                )
                print(f"Independent test metrics: {test_metrics}")

        val_avg = average_metrics(fold_val_metrics)
        final_val_summary[exp_name] = val_avg
        details.append(
            {
                "Experiment": exp_name,
                "Dataset": "Validation",
                "Fold": "Average",
                **val_avg,
            }
        )

        exp_summary_df = pd.DataFrame(fold_val_metrics)
        exp_summary_df.index = [
            f"Fold_{i}" for i in range(1, len(fold_val_metrics) + 1)
        ]
        exp_summary_df.loc["Average"] = val_avg
        exp_summary_df.to_excel(os.path.join(exp_dir, "validation_summary.xlsx"))

        if fold_test_metrics:
            test_avg = average_metrics(fold_test_metrics)
            final_test_summary[exp_name] = test_avg
            details.append(
                {
                    "Experiment": exp_name,
                    "Dataset": "Independent_Test",
                    "Fold": "Average",
                    **test_avg,
                }
            )

            test_summary_df = pd.DataFrame(fold_test_metrics)
            test_summary_df.index = [
                f"Fold_{i}" for i in range(1, len(fold_test_metrics) + 1)
            ]
            test_summary_df.loc["Average"] = test_avg
            test_summary_df.to_excel(
                os.path.join(exp_dir, "independent_test_summary.xlsx")
            )

    pd.DataFrame(final_val_summary).T.to_excel(
        os.path.join(save_root, "final_validation_summary.xlsx")
    )
    if final_test_summary:
        pd.DataFrame(final_test_summary).T.to_excel(
            os.path.join(save_root, "final_independent_test_summary.xlsx")
        )
    pd.DataFrame(details).to_excel(
        os.path.join(save_root, "all_experiment_details.xlsx"),
        index=False,
    )

    run_config = {
        "hidden_sizes": hidden_sizes,
        "n_splits": n_splits,
        "test_ratio": test_ratio,
        "stride": stride,
        "smote_amount": smote_amount,
        "smote_k": smote_k,
        "tomek_k": tomek_k,
        "random_state": random_state,
        "experiments": experiment_list,
    }
    with open(os.path.join(save_root, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {save_root}")
    return save_root


def build_arg_parser():
    default_data_file = os.path.join(
        experimentRecord_data_path,
        "splited_dataset",
        "splited_dataset_2026_0406_1140.xlsx",
    )

    parser = argparse.ArgumentParser(
        description="Run no-augmentation, SMOTE-Tomek, and SMOTE-Tomek+Attention ablations."
    )
    parser.add_argument("--data-file", default=default_data_file)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--stride", type=int, default=6)
    parser.add_argument("--hidden-sizes", default="64,64,64,64")
    parser.add_argument("--smote-amount", type=int, default=800)
    parser.add_argument("--smote-k", type=int, default=7)
    parser.add_argument("--tomek-k", type=int, default=1)
    parser.add_argument("--random-state", type=int, default=123)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--skip-independent-test", action="store_true")
    parser.add_argument(
        "--exp-name",
        type=str,
        nargs="+",
        default=None,
        help="Specific experiment name(s) or keywords to run (supports multiple inputs)",
    )
    parser.add_argument(
        "--grid-search", action="store_true", help="启用多头注意力超参数网格搜索"
    )
    return parser


def main(target_exp_names=None, grid_search=False):
    args = build_arg_parser().parse_args()

    train_info = TrainInfo()
    if args.num_epochs is not None:
        train_info.num_epochs = args.num_epochs
    if args.batch_size is not None:
        train_info.batch_size = args.batch_size

    # TrainInfo is a lightweight class used by existing scripts; update its class
    # attributes so new instances inside the runner see CLI overrides.
    TrainInfo.num_epochs = train_info.num_epochs
    TrainInfo.batch_size = train_info.batch_size

    # 1. 首先根据目标名称过滤基础实验
    base_experiments = EXPERIMENTS
    exp_names_to_use = target_exp_names if target_exp_names else args.exp_name
    if exp_names_to_use:
        base_experiments = [e for e in EXPERIMENTS if e["name"] in exp_names_to_use]
        if not base_experiments:
            print(f"Error: No experiments matching {exp_names_to_use} were found.")
            print(f"Available experiments: {[e['name'] for e in EXPERIMENTS]}")
            return

    run_grid_search = grid_search or args.grid_search

    if run_grid_search:
        print("\n" + "=" * 50)
        print("初始化超参数网格搜索 (Grid Search) ...")
        print("=" * 50)

        # 定义搜索空间 (可根据需要调整)
        gs_hidden_sizes = [[64, 64, 64, 64], [128, 64, 64, 32]]
        gs_lrs = [0.001, 0.0005]
        gs_batch_sizes = [16, 32]
        gs_num_heads = [2, 4, 8]

        target_experiments = []
        for base_exp in base_experiments:
            for hs, lr, bs, nh in itertools.product(
                gs_hidden_sizes, gs_lrs, gs_batch_sizes, gs_num_heads
            ):
                # 如果当前实验不是 MultiHead 注意力，不需要重复跑多次不同 num_heads 的实验
                if base_exp.get("attn_type") != "MultiHead" and nh != gs_num_heads[0]:
                    continue

                new_exp = base_exp.copy()
                nh_str = f"_NH{nh}" if base_exp.get("attn_type") == "MultiHead" else ""
                new_exp["name"] = (
                    f"{base_exp['name']}_GS_HS{hs[0]}_LR{lr}_BS{bs}{nh_str}"
                )
                new_exp["description"] = (
                    f"{base_exp['description']} | GS: HS={hs}, LR={lr}, BS={bs}, Heads={nh if base_exp.get('attn_type') == 'MultiHead' else 'N/A'}"
                )
                new_exp["hidden_sizes"] = hs
                new_exp["learning_rate"] = lr
                new_exp["batch_size"] = bs
                if base_exp.get("attn_type") == "MultiHead":
                    new_exp["num_heads"] = nh
                target_experiments.append(new_exp)
        print(f"即将运行 {len(target_experiments)} 组网格搜索实验。")
    else:
        target_experiments = base_experiments
        print(
            f"Selected {len(target_experiments)} experiments to run: {[e['name'] for e in target_experiments]}"
        )

    df = pd.read_excel(args.data_file, index_col=False)
    run_ablation(
        df=df,
        experiment_list=target_experiments,
        n_splits=args.folds,
        test_ratio=args.test_ratio,
        stride=args.stride,
        hidden_sizes=parse_hidden_sizes(args.hidden_sizes),
        smote_amount=args.smote_amount,
        smote_k=args.smote_k,
        tomek_k=args.tomek_k,
        random_state=args.random_state,
        evaluate_independent_test=not args.skip_independent_test,
    )


if __name__ == "__main__":
    # 单独运行 MultiFeat_Bi_SMOTE_Tomek_TemporalAttn 并启用网格搜索
    main(target_exp_names=["MultiFeat_Bi_SMOTE_Tomek_TemporalAttn"], grid_search=True)
