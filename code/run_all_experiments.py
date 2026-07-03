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
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
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
DEFAULT_RANDOM_STATE = 26

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
        "bayes_iteration",
        "dynamic_threshold",
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


def prepare_final_train_data(train_val_df_raw, config, feature_mode):
    # Build the final training set from the full train_val split.
    if feature_mode == "temp_only":
        train_df = myFunction.fill_data(train_val_df_raw, balanced_data=True, stride=6)
        train_df_flat = apply_augmentation(train_df, config)
        X_train, y_train, scaler = myFunction.prepare_univariate_lstm_data(
            train_df_flat
        )
        return X_train, y_train, scaler, 1

    train_df = myFunction.fill_data(train_val_df_raw)
    train_df_flat = apply_augmentation(train_df, config)
    train_final_df = add_rate_features(train_df_flat)
    X_train, y_train, scaler = myFunction.prepare_lstm_data(train_final_df)
    return X_train, y_train, scaler, 2


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
            if col not in {"Run", "Seed", "Experiment", "Fold", "Dataset"}
            and not col.startswith("param_")
            and pd.api.types.is_numeric_dtype(df[col])
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


def train_and_evaluate_validation_only(
    train_val_df,
    group_dir,
    config,
    feature_mode,
    dynamic_threshold=True,
    threshold_metric="f1",
    n_splits=5,
    random_state=123,
    run_label="run_01",
):
    # Train five CV models and evaluate only on validation folds.
    exp_dir = os.path.join(group_dir, config["name"])
    os.makedirs(exp_dir, exist_ok=True)

    folds = myFunction.stratified_group_kfold_only(
        train_val_df, n_splits=n_splits, random_state=random_state
    )

    fold_rows = []
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

        val_metrics.pop("avg_loss", None)
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

    detail_df = pd.DataFrame(fold_rows)
    detail_df.to_excel(os.path.join(exp_dir, "fold_details.xlsx"), index=False)

    df = pd.DataFrame(fold_rows)
    metric_cols = [
        col
        for col in df.columns
        if col not in {"Run", "Seed", "Experiment", "Fold", "Dataset"}
        and not col.startswith("param_")
        and pd.api.types.is_numeric_dtype(df[col])
    ]
    avg = {col: df[col].mean() for col in metric_cols}
    avg_rows = [
        {
            "Run": run_label,
            "Seed": random_state,
            **config_to_result_fields(config),
            "Dataset": "Validation",
            **avg,
        }
    ]
    pd.DataFrame(avg_rows).to_excel(os.path.join(exp_dir, "summary.xlsx"), index=False)
    return fold_rows, avg_rows


def collect_model_predictions(model, loader, device):
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            all_probs.extend(outputs.cpu().numpy().flatten())
            all_labels.extend(batch_y.cpu().numpy().flatten())
    return np.asarray(all_labels).astype(int), np.asarray(all_probs)


def compute_binary_metrics(labels, probs, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    return {
        "Threshold": float(threshold),
        "Positive_Predictions": int(np.sum(preds == 1)),
        "Prob_Min": float(np.min(probs)),
        "Prob_Max": float(np.max(probs)),
        "Prob_Mean": float(np.mean(probs)),
        "Accuracy": accuracy_score(labels, preds),
        "Precision": precision_score(labels, preds, zero_division=0),
        "Recall": recall_score(labels, preds, zero_division=0),
        "F1-Score": f1_score(labels, preds, zero_division=0),
        "Specificity": specificity,
        "AUC": roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0,
        "MCC": matthews_corrcoef(labels, preds),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def train_final_model_and_evaluate_test(
    train_val_df,
    test_df,
    group_dir,
    config,
    feature_mode="temp_rate",
    decision_threshold=0.5,
    random_state=54,
    run_label="final_train",
):
    # Train one final model on all train_val data and evaluate test once.
    exp_dir = os.path.join(group_dir, config["name"])
    os.makedirs(exp_dir, exist_ok=True)
    set_seed(random_state)

    X_train, y_train, scaler, input_size = prepare_final_train_data(
        train_val_df, config, feature_mode
    )
    X_test, y_test = prepare_test_data(test_df, scaler, feature_mode)

    train_info = apply_train_info_config(TrainInfo(), config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(train_info, device, input_size=input_size)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_info.learning_rate,
        weight_decay=getattr(train_info, "weight_decay", 1e-4),
    )

    train_loader = DataLoader(
        EstrusDataset(X_train, y_train),
        train_info.batch_size,
        shuffle=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        EstrusDataset(X_test, y_test), train_info.batch_size, shuffle=False
    )

    history_rows = []
    for epoch in range(1, train_info.num_epochs + 1):
        model.train()
        epoch_losses = []
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_losses.append(loss.item())

        history_rows.append(
            {"Epoch": epoch, "Train_Loss": float(np.mean(epoch_losses))}
        )

    model_path = os.path.join(exp_dir, "final_model.pth")
    scaler_path = os.path.join(exp_dir, "final_scaler.joblib")
    torch.save(model.state_dict(), model_path)
    joblib.dump(scaler, scaler_path)

    labels, probs = collect_model_predictions(model, test_loader, device)
    preds = (probs >= decision_threshold).astype(int)
    metrics = compute_binary_metrics(labels, probs, threshold=decision_threshold)

    pd.DataFrame(history_rows).to_excel(
        os.path.join(exp_dir, "final_training_history.xlsx"), index=False
    )
    pd.DataFrame(
        [
            {
                "Run": run_label,
                "Seed": random_state,
                **config_to_result_fields(config),
                "Dataset": "Independent_Test",
                **metrics,
            }
        ]
    ).to_excel(os.path.join(exp_dir, "final_test_metrics.xlsx"), index=False)
    pd.DataFrame(
        {
            "Sample_Index": np.arange(len(labels)),
            "y_true": labels,
            "y_prob": probs,
            "y_pred": preds,
            "Threshold": decision_threshold,
        }
    ).to_excel(os.path.join(exp_dir, "final_test_predictions.xlsx"), index=False)
    pd.DataFrame([config]).to_excel(
        os.path.join(exp_dir, "final_model_config.xlsx"), index=False
    )

    return exp_dir, metrics


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
    specified_run_seeds=None,
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

    if specified_run_seeds is not None:
        run_seeds = specified_run_seeds
        run_repeats = len(run_seeds)
    else:
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


def add_dynamic_threshold_variants(configs):
    """
    为每组超参数生成动态阈值关闭/开启两个版本，便于直接对比
    """
    expanded_configs = []
    for config in configs:
        for enabled in [False, True]:
            new_config = config.copy()
            new_config["dynamic_threshold"] = enabled
            new_config["name"] = (
                f"{config.get('name', 'config')}_DT_{'on' if enabled else 'off'}"
            )
            expanded_configs.append(new_config)
    return expanded_configs


def _normalize_search_space(search_space):
    normalized = {}
    for key, values in search_space.items():
        if not isinstance(values, (list, tuple)) or len(values) == 0:
            raise ValueError(f"search_space['{key}'] must be a non-empty list.")
        normalized[key] = list(values)
    return normalized


def _all_index_points(search_space, tunable_keys):
    index_ranges = [range(len(search_space[key])) for key in tunable_keys]
    return [tuple(point) for point in itertools.product(*index_ranges)]


def _point_to_config_values(point, search_space, tunable_keys, fixed_values):
    config_values = fixed_values.copy()
    for key, index in zip(tunable_keys, point):
        value = search_space[key][int(index)]
        if key == "hidden_sizes":
            value = list(value)
        config_values[key] = value
    return config_values


def run_bayesian_config_set(
    train_val_df,
    test_df,
    group_name,
    base_config,
    search_space,
    feature_mode,
    dynamic_threshold_rule,
    threshold_metric="f1",
    n_calls=12,
    n_initial_points=4,
    run_seed=54,
    save_every=5,
):
    # Bayesian optimization over user-defined discrete candidate values.
    try:
        from skopt import Optimizer
        from skopt.space import Categorical
    except ImportError as exc:
        raise ImportError(
            "experiment_7 requires scikit-optimize. Install it with: "
            "pip install scikit-optimize"
        ) from exc

    search_space = _normalize_search_space(search_space)
    fixed_values = {
        key: values[0] for key, values in search_space.items() if len(values) == 1
    }
    tunable_keys = [key for key, values in search_space.items() if len(values) > 1]
    if not tunable_keys:
        raise ValueError(
            "At least one search_space entry must contain multiple values."
        )

    all_points = _all_index_points(search_space, tunable_keys)
    max_calls = min(n_calls, len(all_points))
    dimensions = [
        Categorical(list(range(len(search_space[key]))), name=key)
        for key in tunable_keys
    ]
    optimizer = Optimizer(
        dimensions=dimensions,
        base_estimator="GP",
        n_initial_points=min(n_initial_points, max_calls),
        random_state=run_seed,
    )

    group_dir = make_experiment_dir(group_name)
    save_split_snapshot(group_dir, train_val_df, test_df)
    pd.DataFrame([{"Run": "run_01", "Seed": run_seed}]).to_excel(
        os.path.join(group_dir, "run_seeds.xlsx"), index=False
    )
    pd.DataFrame(
        [
            {
                "Parameter": key,
                "Candidate_Values": json.dumps(values, ensure_ascii=False),
            }
            for key, values in search_space.items()
        ]
    ).to_excel(os.path.join(group_dir, "bayesian_search_space.xlsx"), index=False)

    set_seed(run_seed)
    evaluated_points = set()
    all_details = []
    all_summaries = []

    for iteration in range(1, max_calls + 1):
        point = tuple(optimizer.ask())
        if point in evaluated_points:
            remaining_points = [p for p in all_points if p not in evaluated_points]
            if not remaining_points:
                break
            point = remaining_points[0]
        evaluated_points.add(point)

        config = base_config.copy()
        config.update(
            _point_to_config_values(point, search_space, tunable_keys, fixed_values)
        )
        config["experiment_id"] = f"BO{iteration:03d}"
        config["bayes_iteration"] = iteration
        config["name"] = f"BiLSTM_BO_{iteration:03d}"

        dynamic_threshold = dynamic_threshold_rule(config)
        fold_rows, test_rows, avg_rows = train_and_evaluate_experiment(
            train_val_df=train_val_df,
            test_df=test_df,
            group_dir=group_dir,
            config=config,
            feature_mode=feature_mode,
            dynamic_threshold=dynamic_threshold,
            threshold_metric=threshold_metric,
            random_state=run_seed,
            run_label="run_01",
        )

        all_details.extend(fold_rows)
        all_details.extend(test_rows)
        all_summaries.extend(avg_rows)

        val_row = next(row for row in avg_rows if row["Dataset"] == "Validation")
        score = (
            float(val_row.get("F1-Score", 0.0))
            + 1e-3 * float(val_row.get("AUC", 0.0))
            + 1e-6 * float(val_row.get("Recall", 0.0))
        )
        optimizer.tell(list(point), -score)

        should_save = (iteration % save_every == 0) or (iteration == max_calls)
        if should_save:
            pd.DataFrame(all_details).to_excel(
                os.path.join(group_dir, "all_experiments_cv_details.xlsx"), index=False
            )
            pd.DataFrame(all_summaries).to_excel(
                os.path.join(group_dir, "final_summary.xlsx"), index=False
            )

    return group_dir


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


def load_bayesian_configs_from_summary(summary_path):
    # Replay the exact hyperparameter combinations selected in a previous BO run.
    summary_df = pd.read_excel(summary_path)
    val_df = summary_df[summary_df["Dataset"] == "Validation"].copy()
    val_df["param_bayes_iteration"] = pd.to_numeric(
        val_df["param_bayes_iteration"], errors="coerce"
    )
    val_df = val_df.dropna(subset=["param_bayes_iteration"])
    val_df["param_bayes_iteration"] = val_df["param_bayes_iteration"].astype(int)
    val_df = val_df.sort_values("param_bayes_iteration")

    configs = []
    for _, row in val_df.iterrows():
        iteration = int(row["param_bayes_iteration"])
        config = {
            "name": f"BiLSTM_BO_{iteration:03d}_DT_on",
            "experiment_id": f"BO{iteration:03d}",
            "bayes_iteration": iteration,
            "A": True,
            "S": True,
            "T": True,
            "model_name": "EstrusLSTM",
            "bidirectional": True,
            "use_cell_state": False,
            "hidden_sizes": parse_hidden_sizes(row["param_hidden_sizes"]),
            "batch_size": int(row["param_batch_size"]),
            "dropout_rate": float(row["param_dropout_rate"]),
            "learning_rate": float(row["param_learning_rate"]),
            "weight_decay": float(row["param_weight_decay"]),
            "dynamic_threshold": True,
        }
        configs.append(config)
    return configs


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
        "learning_rate": [5e-4],
        "dropout_rate": [0.2],
        "batch_size": [32],
    }
    configs = add_dynamic_threshold_variants(grid_configs(base_config, param_grid))
    group_dir = run_config_set(
        train_val_df,
        test_df,
        "01_bilstm_ast_structure_tuning",
        configs,
        feature_mode="temp_rate",
        dynamic_threshold_rule=lambda config: bool(
            config.get("dynamic_threshold", False)
        ),
        threshold_metric="f1",
        run_repeats=1,
        specified_run_seeds=[54],
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
    for config in augmentation_configs(include_ast=True):
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
        # specified_run_seeds=[54],
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
            "name": "RNN",
            "model_name": "EstrusRNN",
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
        {
            **base,
            "name": "BiLSTM",
            "model_name": "EstrusLSTM",
            "bidirectional": True,
        },
    ]
    return run_config_set(
        train_val_df,
        test_df,
        "03_model_comparison_ast",
        configs,
        feature_mode="temp_rate",
        dynamic_threshold_rule=lambda config: False,
        # run_repeats=REPEATED_RUNS,
        specified_run_seeds=[54],
    )


def experiment_3_final_model_comparison_ast(
    train_val_df,
    test_df,
    model_configs=None,
    run_repeats=1,
    base_random_state=DEFAULT_RANDOM_STATE,
    specified_run_seeds=None,
):
    # Final test comparison against the selected BiLSTM result.
    # These models use the same selected training hyperparameters as BiLSTM.
    selected_base = {
        "A": True,
        "S": True,
        "T": True,
        "hidden_sizes": [64, 64, 32],
        "learning_rate": 5e-4,
        "dropout_rate": 0.2,
        "batch_size": 32,
        "weight_decay": 1e-4,
        "use_cell_state": False,
        "dynamic_threshold": False,
    }
    if model_configs is None:
        model_configs = [
            {
                **selected_base,
                "name": "Final_RNN_AST",
                "model_name": "EstrusRNN",
                "bidirectional": False,
            },
            {
                **selected_base,
                "name": "Final_LSTM_AST",
                "model_name": "EstrusLSTM",
                "bidirectional": False,
            },
            {
                **selected_base,
                "name": "Final_GRU_AST",
                "model_name": "EstrusGRU",
                "bidirectional": False,
            },
        ]

    group_dir = make_experiment_dir("03_final_model_comparison_ast")
    save_split_snapshot(group_dir, train_val_df, test_df)

    if specified_run_seeds is not None:
        run_seeds = specified_run_seeds
        run_repeats = len(run_seeds)
    else:
        run_seeds = generate_run_seeds(run_repeats, base_random_state)

    pd.DataFrame(
        [
            {"Run": f"run_{idx:02d}", "Seed": seed}
            for idx, seed in enumerate(run_seeds, start=1)
        ]
    ).to_excel(os.path.join(group_dir, "run_seeds.xlsx"), index=False)

    result_rows = []
    for run_idx, run_seed in enumerate(run_seeds, start=1):
        run_label = f"run_{run_idx:02d}"
        run_dir = os.path.join(group_dir, run_label) if run_repeats > 1 else group_dir
        os.makedirs(run_dir, exist_ok=True)

        for config in model_configs:
            exp_dir, metrics = train_final_model_and_evaluate_test(
                train_val_df=train_val_df,
                test_df=test_df,
                group_dir=run_dir,
                config=config,
                feature_mode="temp_rate",
                decision_threshold=0.5,
                random_state=run_seed,
                run_label=run_label,
            )
            result_rows.append(
                {
                    "Run": run_label,
                    "Seed": run_seed,
                    "Experiment_Dir": exp_dir,
                    **config_to_result_fields(config),
                    **metrics,
                }
            )

    result_df = pd.DataFrame(result_rows)
    result_df.to_excel(
        os.path.join(group_dir, "final_model_comparison_test_summary.xlsx"),
        index=False,
    )

    if len(run_seeds) > 1:
        metric_cols = [
            col
            for col in result_df.columns
            if col
            not in {
                "Run",
                "Seed",
                "Experiment_Dir",
                "Experiment",
            }
            and not col.startswith("param_")
            and pd.api.types.is_numeric_dtype(result_df[col])
        ]
        group_cols = [
            col
            for col in result_df.columns
            if col == "Experiment" or col.startswith("param_")
        ]
        summary_rows = []
        for group_values, group_df in result_df.groupby(group_cols, dropna=False):
            if not isinstance(group_values, tuple):
                group_values = (group_values,)
            base_row = dict(zip(group_cols, group_values))
            for metric in metric_cols:
                metric_mean = group_df[metric].mean()
                metric_std = group_df[metric].std(ddof=1)
                summary_rows.append(
                    {
                        **base_row,
                        "Metric": metric,
                        "Mean": metric_mean,
                        "Std": metric_std,
                        "Mean_Std": f"{metric_mean:.4f} +/- {metric_std:.4f}",
                    }
                )
        pd.DataFrame(summary_rows).to_excel(
            os.path.join(group_dir, "final_model_comparison_metric_mean_std.xlsx"),
            index=False,
        )

    return group_dir


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
        # dynamic_threshold_rule=lambda config: bool(config.get("A") and config.get("S") and config.get("T")),
        dynamic_threshold_rule=lambda config: False,
        threshold_metric="f1",
        run_repeats=REPEATED_RUNS,
    )


def experiment_5_smote_ratio_ast(train_val_df, test_df, best_bilstm_config=None):
    # Experiment 5: keep AST enabled and sweep SMOTE oversampling from 1x to 10x.
    bilstm_config = {
        "model_name": "EstrusLSTM",
        "hidden_sizes": [64, 64, 32],
        "learning_rate": 2e-4,
        "dropout_rate": 0.3,
        "batch_size": 64,
        "weight_decay": 1e-5,
        "bidirectional": True,
        "use_cell_state": False,
    }
    if best_bilstm_config is not None:
        bilstm_config.update(best_bilstm_config)

    configs = []
    for ratio in range(1, 11):
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
        run_repeats=1,
    )


def experiment_6_bilistm_ast_structure_tuning_(train_val_df, test_df):
    # Experiment 6: tune the structure of the BiLSTM model with AST features.
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
            [64, 64, 32],
            [64, 64, 64, 64],
            [128, 128, 64, 32],
        ],
        "learning_rate": [0.1, 0.01, 0.001, 0.0007, 0.0005, 0.0003, 0.0001],
        "dropout_rate": [0.1, 0.2, 0.3],
        "batch_size": [32],
    }
    configs = add_dynamic_threshold_variants(grid_configs(base_config, param_grid))
    group_dir = run_config_set(
        train_val_df,
        test_df,
        "06_bilstm_ast_structure_tuning_",
        configs,
        feature_mode="temp_rate",
        dynamic_threshold_rule=lambda config: bool(
            config.get("dynamic_threshold", False)
        ),
        threshold_metric="f1",
        run_repeats=1,
        specified_run_seeds=[54],
    )
    return select_best_bilstm_config(group_dir)


def experiment_7_bilstm_ast_bayesian_tuning(train_val_df, test_df):
    # Experiment 7: Bayesian optimization over specified BiLSTM candidates.
    # Edit search_space and n_calls to control the candidate set and budget.
    base_config = {
        "A": True,
        "S": True,
        "T": True,
        "model_name": "EstrusLSTM",
        "bidirectional": True,
        "use_cell_state": False,
    }
    search_space = {
        "hidden_sizes": [
            [32, 32],
            [64, 32],
            [64, 64],
            [64, 64, 32],
            [128, 64, 32],
            [64, 64, 64, 64],
            [128, 128, 64, 32],
        ],
        "learning_rate": [1e-4, 2e-4, 3e-4, 5e-4, 7e-4, 1e-3, 2e-3],
        "dropout_rate": [0.1, 0.2, 0.3, 0.4],
        "batch_size": [16, 32, 64],
        "weight_decay": [0.0, 1e-5, 1e-4, 5e-4, 1e-3],
        # Change this to [False, True] if you also want BO to compare threshold modes.
        "dynamic_threshold": [False],
    }
    group_dir = run_bayesian_config_set(
        train_val_df=train_val_df,
        test_df=test_df,
        group_name="07_bilstm_ast_bayesian_tuning",
        base_config=base_config,
        search_space=search_space,
        feature_mode="temp_rate",
        dynamic_threshold_rule=lambda config: bool(
            config.get("dynamic_threshold", False)
        ),
        threshold_metric="f1",
        n_calls=60,
        n_initial_points=12,
        run_seed=54,
    )
    return select_best_bilstm_config(group_dir)


def experiment_8_replay_bayesian_configs_with_dynamic_threshold(train_val_df):
    # Replay the BO-selected hyperparameters from experiment 7 with dynamic threshold.
    # Only validation folds are evaluated; the independent test set is not touched.
    source_summary_path = os.path.join(
        ALL_EXPERIMENTS_ROOT,
        "07_bilstm_ast_bayesian_tuning_2026_0701_2159",
        "final_summary.xlsx",
    )
    configs = load_bayesian_configs_from_summary(source_summary_path)

    group_dir = make_experiment_dir("08_bilstm_ast_bayes_replay_dynamic_threshold")
    train_val_df.to_excel(os.path.join(group_dir, "train_val_df.xlsx"), index=False)
    pd.DataFrame(
        [{"Source_Final_Summary": source_summary_path, "Config_Count": len(configs)}]
    ).to_excel(os.path.join(group_dir, "source_summary.xlsx"), index=False)
    pd.DataFrame([{"Run": "run_01", "Seed": 54}]).to_excel(
        os.path.join(group_dir, "run_seeds.xlsx"), index=False
    )

    run_seed = 54
    set_seed(run_seed)
    all_details = []
    all_summaries = []

    for config in configs:
        fold_rows, avg_rows = train_and_evaluate_validation_only(
            train_val_df=train_val_df,
            group_dir=group_dir,
            config=config,
            feature_mode="temp_rate",
            dynamic_threshold=True,
            threshold_metric="f1",
            random_state=run_seed,
            run_label="run_01",
        )
        all_details.extend(fold_rows)
        all_summaries.extend(avg_rows)

        pd.DataFrame(all_details).to_excel(
            os.path.join(group_dir, "all_experiments_cv_details.xlsx"), index=False
        )
        pd.DataFrame(all_summaries).to_excel(
            os.path.join(group_dir, "final_summary.xlsx"), index=False
        )

    return select_best_bilstm_config(group_dir)


def experiment_9_train_final_model_and_evaluate_test(
    train_val_df,
    test_df,
    run_repeats=1,
    base_random_state=DEFAULT_RANDOM_STATE,
    specified_run_seeds=None,
):
    # Final evaluation: train one model on all train_val data and test once.
    # Set best_config_path to the selected best_bilstm_config.json if available.
    best_config_path = None
    final_config = {
        "name": "Final_BiLSTM_AST",
        "A": True,
        "S": True,
        "T": True,
        "model_name": "EstrusLSTM",
        "hidden_sizes": [64, 64, 32],
        "learning_rate": 5e-4,
        "dropout_rate": 0.2,
        "batch_size": 32,
        "weight_decay": 1e-4,
        "bidirectional": True,
        "use_cell_state": False,
        "dynamic_threshold": False,
    }
    if best_config_path:
        with open(best_config_path, "r", encoding="utf-8") as f:
            final_config.update(json.load(f))
        final_config["name"] = "Final_BiLSTM_AST"

    group_dir = make_experiment_dir("09_final_model_test_evaluation")
    save_split_snapshot(group_dir, train_val_df, test_df)

    if specified_run_seeds is not None:
        run_seeds = specified_run_seeds
        run_repeats = len(run_seeds)
    else:
        run_seeds = generate_run_seeds(run_repeats, base_random_state)

    pd.DataFrame(
        [
            {"Run": f"run_{idx:02d}", "Seed": seed}
            for idx, seed in enumerate(run_seeds, start=1)
        ]
    ).to_excel(os.path.join(group_dir, "run_seeds.xlsx"), index=False)

    result_rows = []
    exp_dirs = []
    for run_idx, run_seed in enumerate(run_seeds, start=1):
        run_label = f"run_{run_idx:02d}"
        run_dir = os.path.join(group_dir, run_label) if run_repeats > 1 else group_dir
        os.makedirs(run_dir, exist_ok=True)

        exp_dir, metrics = train_final_model_and_evaluate_test(
            train_val_df=train_val_df,
            test_df=test_df,
            group_dir=run_dir,
            config=final_config,
            feature_mode="temp_rate",
            decision_threshold=0.5,
            random_state=run_seed,
            run_label=run_label,
        )
        exp_dirs.append(exp_dir)
        result_rows.append(
            {
                "Run": run_label,
                "Seed": run_seed,
                "Experiment_Dir": exp_dir,
                **config_to_result_fields(final_config),
                **metrics,
            }
        )

    result_df = pd.DataFrame(result_rows)
    result_df.to_excel(
        os.path.join(group_dir, "final_test_result_summary.xlsx"), index=False
    )

    metric_cols = [
        col
        for col in result_df.columns
        if col
        not in {
            "Run",
            "Seed",
            "Experiment_Dir",
            "Experiment",
        }
        and not col.startswith("param_")
        and pd.api.types.is_numeric_dtype(result_df[col])
    ]
    summary_rows = []
    for metric in metric_cols:
        metric_mean = result_df[metric].mean()
        metric_std = result_df[metric].std(ddof=1)
        summary_rows.append(
            {
                "Metric": metric,
                "Mean": metric_mean,
                "Std": metric_std,
                "Mean_Std": f"{metric_mean:.4f} ± {metric_std:.4f}",
            }
        )
    pd.DataFrame(summary_rows).to_excel(
        os.path.join(group_dir, "final_test_metric_mean_std.xlsx"), index=False
    )
    return exp_dirs


def main():
    # The fixed split makes the independent test set identical for every group.
    # Repeated runs only change the train/validation fold seed.
    set_seed(DEFAULT_RANDOM_STATE)
    os.makedirs(ALL_EXPERIMENTS_ROOT, exist_ok=True)

    df = load_source_df()
    train_val_df, test_df = load_or_create_fixed_split(df)

    # Run the five experiment groups in the order described in the manuscript plan.
    best_bilstm_config = None
    # best_bilstm_config = experiment_1_bilstm_ast_structure_tuning(train_val_df, test_df)

    """ experiment_2_aug_ablation_bilstm_no_tuning(
        train_val_df, test_df, best_bilstm_config=best_bilstm_config
    ) """

    # experiment_3_model_comparison_ast(train_val_df, test_df)

    experiment_3_final_model_comparison_ast(
        train_val_df, test_df, specified_run_seeds=[676, 211, 443, 616, 558, 59, 131]
    )

    # experiment_4_temp_only_aug_ablation_bilstm(train_val_df, test_df)

    """ experiment_5_smote_ratio_ast(
        train_val_df, test_df, best_bilstm_config=best_bilstm_config
    ) """

    # experiment_6_bilistm_ast_structure_tuning_(train_val_df, test_df)

    # experiment_7_bilstm_ast_bayesian_tuning(train_val_df, test_df)

    # experiment_8_replay_bayesian_configs_with_dynamic_threshold(train_val_df)

    """ experiment_9_train_final_model_and_evaluate_test(
        train_val_df, test_df, specified_run_seeds=[676, 211, 443, 616, 558, 59, 131]
    ) """


if __name__ == "__main__":
    main()
