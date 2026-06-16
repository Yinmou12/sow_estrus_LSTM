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

pd.set_option("future.no_silent_downcasting", True)

import sow_estrus_LSTM_Function as myFunction
from lstm_model import EarlyStopping
from sow_estrus_LSTM_Info import (
    experimentRecord_data_path,
    info_FINAL_SAVE_PATH,
    result_save_path,
)
from sow_estrus_LSTM_train import (
    __train_info as TrainInfo,
    EstrusDataset,
    evaluate_model,
    get_model,
)

FIXED_SPLIT_DIR = os.path.join(info_FINAL_SAVE_PATH, "cross_validation")
FIXED_TRAIN_VAL_PATH = os.path.join(FIXED_SPLIT_DIR, "train_val_df.xlsx")
FIXED_TEST_PATH = os.path.join(FIXED_SPLIT_DIR, "independent_test_df.xlsx")
ALL_EXPERIMENTS_ROOT = os.path.join(result_save_path, "all_experiments")
DEFAULT_RANDOM_STATE = 123

# Experiments 2-5 are repeated to observe stability across CV fold splits.
# Increase this value if you want more repeated results in one execution.
REPEATED_RUNS = 10

TRAIN_INFO_KEYS = {
    "model_name",
    "hidden_sizes",
    "learning_rate",
    "weight_decay",
    "dropout_rate",
    "batch_size",
    "use_cell_state",
    "bidirectional",
    "num_heads",
    "num_epochs",
    "early_patience",
    "lr_patience",
}


def set_seed(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def generate_run_seeds(run_repeats, base_random_state=DEFAULT_RANDOM_STATE):
    # Generate unique seeds in (0, 1000). The base state makes the seed list reproducible.
    if run_repeats > 999:
        raise ValueError(
            "run_repeats cannot exceed 999 when seeds are sampled from 1..999."
        )
    return random.Random(base_random_state).sample(range(1, 1000), run_repeats)


def load_source_df():
    return pd.read_excel(
        os.path.join(
            experimentRecord_data_path,
            "splited_dataset",
            "splited_dataset_2026_0406_1140.xlsx",
        ),
        index_col=False,
    )


def load_or_create_fixed_split(df=None, random_state=123):
    # Keep the independent test set fixed across all experiment groups.
    if os.path.exists(FIXED_TRAIN_VAL_PATH) and os.path.exists(FIXED_TEST_PATH):
        train_val_df = pd.read_excel(FIXED_TRAIN_VAL_PATH, index_col=False)
        test_df = pd.read_excel(FIXED_TEST_PATH, index_col=False)
        return train_val_df, test_df

    if df is None:
        raise FileNotFoundError(
            "Fixed split files were not found. Provide df once to create them."
        )

    os.makedirs(FIXED_SPLIT_DIR, exist_ok=True)
    train_val_df, test_df = myFunction.split_dataset_train_val_test(
        df, random_state=random_state
    )
    train_val_df.to_excel(FIXED_TRAIN_VAL_PATH, index=False)
    test_df.to_excel(FIXED_TEST_PATH, index=False)
    return train_val_df, test_df


def apply_train_info_config(train_info, config):
    # Map experiment config values onto the existing TrainInfo object.
    for key in TRAIN_INFO_KEYS:
        if key in config:
            setattr(train_info, key, config[key])

    if getattr(train_info, "hidden_sizes", None):
        train_info.num_layers = len(train_info.hidden_sizes)

    return train_info


def config_to_result_fields(config):
    # Store hyperparameters as flat Excel columns for easy sorting/filtering.
    fields = {"Experiment": config["name"]}
    for key in [
        "experiment_id",
        "A",
        "S",
        "T",
        "smote_amount",
        *sorted(TRAIN_INFO_KEYS),
    ]:
        if key not in config:
            continue
        value = config[key]
        if isinstance(value, (list, tuple)):
            value = "_".join(map(str, value))
        fields[f"param_{key}"] = value

    if "hidden_sizes" in config:
        fields["param_num_layers"] = len(config["hidden_sizes"])

    return fields


def make_experiment_dir(group_name):
    timestamp = datetime.now().strftime("%Y_%m%d_%H%M")
    group_dir = os.path.join(ALL_EXPERIMENTS_ROOT, f"{group_name}_{timestamp}")
    os.makedirs(group_dir, exist_ok=True)
    return group_dir


def save_split_snapshot(group_dir, train_val_df, test_df):
    train_val_df.to_excel(os.path.join(group_dir, "train_val_df.xlsx"), index=False)
    test_df.to_excel(os.path.join(group_dir, "independent_test_df.xlsx"), index=False)


def augmentation_configs(include_ast=True):
    # Data augmentation switches: A=ADASYN, S=SMOTE, T=Tomek Links.
    configs = [
        ("Baseline", False, False, False),
        ("A", True, False, False),
        ("S", False, True, False),
        ("T", False, False, True),
        ("AS", True, True, False),
        ("AT", True, False, True),
        ("ST", False, True, True),
    ]
    if include_ast:
        configs.append(("AST", True, True, True))
    return [{"name": name, "A": a, "S": s, "T": t} for name, a, s, t in configs]


def apply_augmentation(train_df, config):
    # Apply augmentation only to the training fold to avoid validation leakage.
    train_df_flat = myFunction.convert_features(train_df)

    if config.get("A", False):
        df_min = train_df_flat[train_df_flat["isEstrus"] == 1]
        df_maj = train_df_flat[train_df_flat["isEstrus"] == 0]
        train_df_flat = myFunction.ADASYN(
            threshold=0.9,
            gamma=1,
            df_min=df_min,
            df_maj=df_maj,
        )

    if config.get("S", False):
        train_df_flat = myFunction.SMOTE(
            train_df_flat,
            amount_oversampling=config.get("smote_amount", 800),
            k=7,
        )

    if config.get("T", False):
        train_df_flat = myFunction.TomekLinked(train_df_flat, k=1)

    return train_df_flat


def add_rate_features(flat_df):
    # Build the second feature channel from adjacent temperature differences.
    temp_feats = flat_df.iloc[:, 1:-1].copy()
    rate_feats = temp_feats.diff(axis=1).fillna(0)
    rate_feats.columns = [f"rate_{i}" for i in range(1, rate_feats.shape[1] + 1)]
    return pd.concat([flat_df.iloc[:, :-1], rate_feats, flat_df.iloc[:, -1]], axis=1)


def prepare_fold_data(train_df_raw, val_df_raw, config, feature_mode):
    # Return tensors and scaler for either temp-only or temp+rate experiments.
    if feature_mode == "temp_only":
        train_df = myFunction.fill_data(train_df_raw, balanced_data=True, stride=6)
        val_df = myFunction.fill_data(val_df_raw)
        train_df_flat = apply_augmentation(train_df, config)
        X_train, y_train, scaler = myFunction.prepare_univariate_lstm_data(
            train_df_flat
        )
        X_val, y_val, _ = myFunction.prepare_univariate_lstm_data(val_df, scaler=scaler)
        return X_train, y_train, X_val, y_val, scaler, 1

    train_df = myFunction.fill_data(train_df_raw)
    val_df = myFunction.fill_data(val_df_raw)
    train_df_flat = apply_augmentation(train_df, config)
    train_final_df = add_rate_features(train_df_flat)
    X_train, y_train, scaler = myFunction.prepare_lstm_data(train_final_df)
    X_val, y_val, _ = myFunction.prepare_lstm_data(val_df, scaler=scaler)
    return X_train, y_train, X_val, y_val, scaler, 2


def prepare_test_data(test_df_raw, scaler, feature_mode):
    test_df = myFunction.fill_data(test_df_raw)
    if feature_mode == "temp_only":
        X_test, y_test, _ = myFunction.prepare_univariate_lstm_data(
            test_df, scaler=scaler
        )
    else:
        X_test, y_test, _ = myFunction.prepare_lstm_data(test_df, scaler=scaler)
    return X_test, y_test


def save_threshold(exp_dir, fold_idx, threshold):
    with open(
        os.path.join(exp_dir, f"threshold_fold{fold_idx}.txt"),
        "w",
        encoding="utf-8",
    ) as f:
        f.write(str(float(threshold)))


def train_and_evaluate_experiment(
    train_val_df,
    test_df,
    group_dir,
    config,
    feature_mode,
    dynamic_threshold=False,
    threshold_metric="f1",
    n_splits=5,
    random_state=123,
    run_label="run_01",
):
    exp_dir = os.path.join(group_dir, config["name"])
    os.makedirs(exp_dir, exist_ok=True)

    folds = myFunction.stratified_group_kfold_only(
        train_val_df, n_splits=n_splits, random_state=random_state
    )

    fold_rows = []
    test_rows = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.BCELoss()

    for fold_idx, (train_df_raw, val_df_raw) in enumerate(folds, start=1):
        X_train, y_train, X_val, y_val, scaler, input_size = prepare_fold_data(
            train_df_raw, val_df_raw, config, feature_mode
        )

        train_info = apply_train_info_config(TrainInfo(), config)
        model = get_model(train_info, device, input_size=input_size)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=train_info.learning_rate,
            weight_decay=getattr(train_info, "weight_decay", 1e-4),
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=train_info.lr_patience
        )

        train_loader = DataLoader(
            EstrusDataset(X_train, y_train),
            train_info.batch_size,
            shuffle=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            EstrusDataset(X_val, y_val), train_info.batch_size, shuffle=False
        )

        model_path = os.path.join(exp_dir, f"best_model_fold{fold_idx}.pth")
        scaler_path = os.path.join(exp_dir, f"scaler_fold{fold_idx}.joblib")
        early_stopping = EarlyStopping(
            patience=train_info.early_patience,
            verbose=False,
            path=model_path,
        )

        for _ in range(train_info.num_epochs):
            model.train()
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                loss = criterion(model(batch_X), batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            val_metrics, _, _ = evaluate_model(model, val_loader, criterion, device)
            scheduler.step(val_metrics["avg_loss"])
            early_stopping(val_metrics["avg_loss"], model)
            if early_stopping.early_stop:
                break

        model.load_state_dict(torch.load(model_path, weights_only=True))
        joblib.dump(scaler, scaler_path)

        val_metrics, _, _ = evaluate_model(
            model,
            val_loader,
            criterion,
            device,
            optimize_threshold=dynamic_threshold,
            threshold_metric=threshold_metric,
        )
        threshold = val_metrics.get("Threshold", 0.5)
        if dynamic_threshold:
            save_threshold(exp_dir, fold_idx, threshold)

        X_test, y_test = prepare_test_data(test_df, scaler, feature_mode)
        test_loader = DataLoader(
            EstrusDataset(X_test, y_test), train_info.batch_size, shuffle=False
        )
        test_metrics, _, _ = evaluate_model(
            model, test_loader, criterion, device, threshold=threshold
        )

        val_metrics.pop("avg_loss", None)
        test_metrics.pop("avg_loss", None)

        fold_rows.append(
            {
                "Run": run_label,
                "Seed": random_state,
                **config_to_result_fields(config),
                "Fold": f"Fold_{fold_idx}",
                "Dataset": "Validation",
                **val_metrics,
            }
        )
        test_rows.append(
            {
                "Run": run_label,
                "Seed": random_state,
                **config_to_result_fields(config),
                "Fold": f"Fold_{fold_idx}",
                "Dataset": "Independent_Test",
                **test_metrics,
            }
        )

    detail_df = pd.DataFrame(fold_rows + test_rows)
    detail_df.to_excel(os.path.join(exp_dir, "fold_details.xlsx"), index=False)

    avg_rows = []
    for dataset_name, rows in [
        ("Validation", fold_rows),
        ("Independent_Test", test_rows),
    ]:
        df = pd.DataFrame(rows)
        metric_cols = [
            col
            for col in df.columns
            if col not in {"Experiment", "Fold", "Dataset"}
            and not col.startswith("param_")
        ]
        avg = {col: df[col].mean() for col in metric_cols}
        avg_rows.append(
            {
                "Run": run_label,
                "Seed": random_state,
                **config_to_result_fields(config),
                "Dataset": dataset_name,
                **avg,
            }
        )

    avg_df = pd.DataFrame(avg_rows)
    avg_df.to_excel(os.path.join(exp_dir, "summary.xlsx"), index=False)
    return fold_rows, test_rows, avg_rows


def run_config_set(
    train_val_df,
    test_df,
    group_name,
    configs,
    feature_mode,
    dynamic_threshold_rule,
    threshold_metric="f1",
    run_repeats=1,
    base_random_state=DEFAULT_RANDOM_STATE,
):
    group_dir = make_experiment_dir(group_name)
    save_split_snapshot(group_dir, train_val_df, test_df)

    all_details = []
    all_summaries = []

    prepared_configs = []
    for idx, config in enumerate(configs, start=1):
        config = config.copy()
        config.setdefault("experiment_id", f"E{idx:03d}")
        config["name"] = f"{config['experiment_id']}_{config['name']}"
        prepared_configs.append(config)

    run_seeds = generate_run_seeds(run_repeats, base_random_state)
    pd.DataFrame(
        [
            {"Run": f"run_{idx:02d}", "Seed": seed}
            for idx, seed in enumerate(run_seeds, start=1)
        ]
    ).to_excel(os.path.join(group_dir, "run_seeds.xlsx"), index=False)

    # Repeat experiments with unique sampled seeds for CV folds and model training.
    for run_idx, run_seed in enumerate(run_seeds, start=1):
        run_label = f"run_{run_idx:02d}"
        set_seed(run_seed)
        run_dir = os.path.join(group_dir, run_label) if run_repeats > 1 else group_dir
        os.makedirs(run_dir, exist_ok=True)

        run_details = []
        run_summaries = []

        for config in prepared_configs:
            dynamic_threshold = dynamic_threshold_rule(config)

            fold_rows, test_rows, avg_rows = train_and_evaluate_experiment(
                train_val_df=train_val_df,
                test_df=test_df,
                group_dir=run_dir,
                config=config,
                feature_mode=feature_mode,
                dynamic_threshold=dynamic_threshold,
                threshold_metric=threshold_metric,
                random_state=run_seed,
                run_label=run_label,
            )
            run_details.extend(fold_rows)
            run_details.extend(test_rows)
            run_summaries.extend(avg_rows)

        pd.DataFrame(run_details).to_excel(
            os.path.join(run_dir, "all_experiments_cv_details.xlsx"), index=False
        )
        pd.DataFrame(run_summaries).to_excel(
            os.path.join(run_dir, "final_summary.xlsx"), index=False
        )

        all_details.extend(run_details)
        all_summaries.extend(run_summaries)

    pd.DataFrame(all_details).to_excel(
        os.path.join(group_dir, "all_experiments_cv_details.xlsx"), index=False
    )
    pd.DataFrame(all_summaries).to_excel(
        os.path.join(group_dir, "final_summary.xlsx"), index=False
    )
    return group_dir


def grid_configs(base_config, param_grid):
    keys = list(param_grid.keys())
    configs = []
    for idx, values in enumerate(itertools.product(*param_grid.values()), start=1):
        config = base_config.copy()
        for key, value in zip(keys, values):
            config[key] = value
        config["name"] = f"BiLSTM_Grid_{idx:03d}"
        configs.append(config)
    return configs


def parse_hidden_sizes(value):
    if isinstance(value, list):
        return [int(item) for item in value]
    if pd.isna(value):
        return None
    return [int(item) for item in str(value).split("_") if item]


def parse_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).lower() == "true"


def select_best_bilstm_config(group_dir):
    # Pick the best architecture from validation results only, avoiding test leakage.
    # Ranking priority: F1-Score, AUC, then Recall.
    summary_path = os.path.join(group_dir, "final_summary.xlsx")
    summary_df = pd.read_excel(summary_path)
    val_df = summary_df[summary_df["Dataset"] == "Validation"].copy()

    sort_cols = [col for col in ["F1-Score", "AUC", "Recall"] if col in val_df.columns]
    val_df = val_df.sort_values(sort_cols, ascending=[False] * len(sort_cols))
    best_row = val_df.iloc[0]

    best_config = {
        "model_name": "EstrusLSTM",
        "hidden_sizes": parse_hidden_sizes(best_row["param_hidden_sizes"]),
        "learning_rate": float(best_row["param_learning_rate"]),
        "dropout_rate": float(best_row.get("param_dropout_rate", 0.2)),
        "batch_size": int(best_row.get("param_batch_size", 32)),
        "bidirectional": parse_bool(best_row.get("param_bidirectional", True)),
        "use_cell_state": parse_bool(best_row.get("param_use_cell_state", False)),
    }
    if "param_weight_decay" in best_row and not pd.isna(best_row["param_weight_decay"]):
        best_config["weight_decay"] = float(best_row["param_weight_decay"])

    best_record = {
        **best_config,
        "source_experiment": best_row["Experiment"],
        "source_run": best_row["Run"],
        "selection_metric": "Validation F1-Score",
        "selection_score": float(best_row["F1-Score"]),
    }

    with open(
        os.path.join(group_dir, "best_bilstm_config.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(best_record, f, indent=2)
    pd.DataFrame([best_record]).to_excel(
        os.path.join(group_dir, "best_bilstm_config.xlsx"), index=False
    )
    return best_config


def experiment_1_bilstm_ast_structure_tuning(train_val_df, test_df):
    # Experiment 1: AST + temp/rate features + Bi-LSTM structure grid search.
    # Dynamic threshold is enabled and selected by validation F1.
    base_config = {
        "A": True,
        "S": True,
        "T": True,
        "model_name": "EstrusLSTM",
        "bidirectional": True,
        "use_cell_state": False,
    }
    param_grid = {
        "hidden_sizes": [
            [32, 32],
            [64, 32],
            [64, 64, 32],
            [128, 64, 32],
            [64, 64, 64, 64],
            [128, 128, 64, 32],
        ],
        "learning_rate": [5e-4, 1e-3],
        "dropout_rate": [0.2],
        "batch_size": [32],
    }
    configs = grid_configs(base_config, param_grid)
    group_dir = run_config_set(
        train_val_df,
        test_df,
        "01_bilstm_ast_structure_tuning",
        configs,
        feature_mode="temp_rate",
        dynamic_threshold_rule=lambda config: True,
        threshold_metric="f1",
        run_repeats=1,
    )
    return select_best_bilstm_config(group_dir)


def experiment_2_aug_ablation_bilstm_no_tuning(
    train_val_df, test_df, best_bilstm_config=None
):
    # Experiment 2: seven augmentation settings with a fixed Bi-LSTM.
    # No hyperparameter tuning and no dynamic threshold.
    bilstm_config = {
        "model_name": "EstrusLSTM",
        "hidden_sizes": [64, 64, 64, 64],
        "learning_rate": 5e-4,
        "dropout_rate": 0.2,
        "batch_size": 32,
        "bidirectional": True,
        "use_cell_state": False,
    }
    if best_bilstm_config is not None:
        bilstm_config.update(best_bilstm_config)

    configs = []
    for config in augmentation_configs(include_ast=False):
        config.update(bilstm_config)
        configs.append(config)
    return run_config_set(
        train_val_df,
        test_df,
        "02_aug_ablation_bilstm",
        configs,
        feature_mode="temp_rate",
        dynamic_threshold_rule=lambda config: False,
        run_repeats=REPEATED_RUNS,
    )


def experiment_3_model_comparison_ast(train_val_df, test_df):
    # Experiment 3: compare simple RNN, LSTM, and GRU under AST augmentation.
    # All models use the fixed 0.5 decision threshold.
    base = {
        "A": True,
        "S": True,
        "T": True,
        "hidden_sizes": [64, 64, 64, 64],
        "learning_rate": 5e-4,
        "dropout_rate": 0.2,
        "batch_size": 32,
        "use_cell_state": False,
    }
    configs = [
        {
            **base,
            "name": "RNN_sample",
            "model_name": "EstrusRNN_sample",
            "bidirectional": False,
        },
        {
            **base,
            "name": "LSTM",
            "model_name": "EstrusLSTM",
            "bidirectional": False,
        },
        {
            **base,
            "name": "GRU",
            "model_name": "EstrusGRU",
            "bidirectional": False,
        },
    ]
    return run_config_set(
        train_val_df,
        test_df,
        "03_model_comparison_ast",
        configs,
        feature_mode="temp_rate",
        dynamic_threshold_rule=lambda config: False,
        run_repeats=REPEATED_RUNS,
    )


def experiment_4_temp_only_aug_ablation_bilstm(train_val_df, test_df):
    # Experiment 4: temp-only feature ablation across all eight augmentation settings.
    # Dynamic threshold is enabled only for the AST setting.
    configs = []
    for config in augmentation_configs(include_ast=True):
        config.update(
            {
                "model_name": "EstrusLSTM",
                "hidden_sizes": [64, 64, 64, 64],
                "learning_rate": 5e-4,
                "dropout_rate": 0.2,
                "batch_size": 32,
                "bidirectional": True,
                "use_cell_state": False,
            }
        )
        configs.append(config)
    return run_config_set(
        train_val_df,
        test_df,
        "04_temp_only_aug_ablation_bilstm",
        configs,
        feature_mode="temp_only",
        dynamic_threshold_rule=lambda config: bool(
            config.get("A") and config.get("S") and config.get("T")
        ),
        threshold_metric="f1",
        run_repeats=REPEATED_RUNS,
    )


def experiment_5_smote_ratio_ast(train_val_df, test_df, best_bilstm_config=None):
    # Experiment 5: keep AST enabled and sweep SMOTE oversampling from 3x to 10x.
    bilstm_config = {
        "model_name": "EstrusLSTM",
        "hidden_sizes": [64, 64, 64, 64],
        "learning_rate": 5e-4,
        "dropout_rate": 0.2,
        "batch_size": 32,
        "bidirectional": True,
        "use_cell_state": False,
    }
    if best_bilstm_config is not None:
        bilstm_config.update(best_bilstm_config)

    configs = []
    for ratio in range(3, 11):
        config = {
            "name": f"SMOTE_{ratio}x",
            "A": True,
            "S": True,
            "T": True,
            "smote_amount": ratio * 100,
            **bilstm_config,
        }
        configs.append(config)
    return run_config_set(
        train_val_df,
        test_df,
        "05_smote_ratio_ast",
        configs,
        feature_mode="temp_rate",
        dynamic_threshold_rule=lambda config: False,
        run_repeats=REPEATED_RUNS,
    )


def main():
    # The fixed split makes the independent test set identical for every group.
    # Repeated runs only change the train/validation fold seed.
    set_seed(123)
    os.makedirs(ALL_EXPERIMENTS_ROOT, exist_ok=True)

    df = load_source_df()
    train_val_df, test_df = load_or_create_fixed_split(df)

    # Run the five experiment groups in the order described in the manuscript plan.
    best_bilstm_config = None
    best_bilstm_config = experiment_1_bilstm_ast_structure_tuning(train_val_df, test_df)
    experiment_2_aug_ablation_bilstm_no_tuning(
        train_val_df, test_df, best_bilstm_config=best_bilstm_config
    )
    experiment_3_model_comparison_ast(train_val_df, test_df)
    experiment_4_temp_only_aug_ablation_bilstm(train_val_df, test_df)
    experiment_5_smote_ratio_ast(
        train_val_df, test_df, best_bilstm_config=best_bilstm_config
    )


if __name__ == "__main__":
    main()
