# -*- coding: utf-8 -*-
"""
消融实验结果分析脚本

生成对比表格、绘制对比图表、统计显著性检验。
"""

import argparse
import json
import os
import sys
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def load_experiment_results(result_dir: str) -> dict:
    """
    加载消融实验结果

    Args:
        result_dir: 结果目录

    Returns:
        实验结果字典
    """
    results = {}

    # 查找所有实验结果文件
    pattern = os.path.join(result_dir, "*", "experiment_result.json")
    result_files = glob(pattern)

    for file_path in result_files:
        with open(file_path, "r", encoding="utf-8") as f:
            result = json.load(f)

        exp_id = result["config"]["experiment_id"]
        exp_name = result["config"]["experiment_name"]
        key = f"{exp_id}_{exp_name}"

        results[key] = result

    # 按实验ID排序
    results = dict(sorted(results.items()))

    return results


def generate_summary_table(results: dict) -> pd.DataFrame:
    """
    生成汇总表格

    Args:
        results: 实验结果字典

    Returns:
        汇总DataFrame
    """
    rows = []

    for key, result in results.items():
        if "error" in result:
            continue

        config = result["config"]
        avg = result["avg_metrics"]
        std = result["std_metrics"]

        da = config["data_augmentation"]
        da_str = ""
        if da["use_adasyn"]:
            da_str += "A"
        if da["use_smote"]:
            da_str += "S"
        if da["use_tomek_links"]:
            da_str += "T"
        if not da_str:
            da_str = "-"

        row = {
            "实验ID": config["experiment_id"],
            "实验名称": config["experiment_name"],
            "数据增强": da_str,
            "Accuracy": f"{avg['accuracy']:.4f}±{std['accuracy']:.4f}",
            "Precision": f"{avg['precision']:.4f}±{std['precision']:.4f}",
            "Recall": f"{avg['recall']:.4f}±{std['recall']:.4f}",
            "F1": f"{avg['f1']:.4f}±{std['f1']:.4f}",
            "AUC": f"{avg['auc']:.4f}±{std['auc']:.4f}",
            # 数值版本用于统计
            "_accuracy": avg["accuracy"],
            "_precision": avg["precision"],
            "_recall": avg["recall"],
            "_f1": avg["f1"],
            "_auc": avg["auc"],
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def plot_metrics_comparison(
    results: dict, save_dir: str, metrics: list = None
):
    """
    绘制指标对比图

    Args:
        results: 实验结果字典
        save_dir: 保存目录
        metrics: 要绘制的指标列表
    """
    if metrics is None:
        metrics = ["accuracy", "precision", "recall", "f1", "auc"]

    # 准备数据
    exp_names = []
    data = {m: {"mean": [], "std": []} for m in metrics}

    for key, result in results.items():
        if "error" in result:
            continue
        exp_names.append(result["config"]["experiment_name"])
        for m in metrics:
            data[m]["mean"].append(result["avg_metrics"][m])
            data[m]["std"].append(result["std_metrics"][m])

    # 绘制柱状图
    x = np.arange(len(exp_names))
    width = 0.15

    fig, ax = plt.subplots(figsize=(14, 6))

    for i, m in enumerate(metrics):
        means = np.array(data[m]["mean"])
        stds = np.array(data[m]["std"])
        ax.bar(
            x + i * width,
            means,
            width,
            yerr=stds,
            label=m.capitalize(),
            capsize=3,
            alpha=0.8,
        )

    ax.set_xlabel("实验")
    ax.set_ylabel("指标值")
    ax.set_title("消融实验指标对比")
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(exp_names, rotation=45, ha="right")
    ax.legend(loc="upper left")
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "metrics_comparison.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()

    print(f"指标对比图已保存至: {save_dir}/metrics_comparison.png")


def plot_radar_chart(results: dict, save_dir: str):
    """
    绘制雷达图对比

    Args:
        results: 实验结果字典
        save_dir: 保存目录
    """
    metrics = ["accuracy", "precision", "recall", "f1", "auc"]
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # 闭合

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    for idx, (key, result) in enumerate(results.items()):
        if "error" in result:
            continue

        values = [result["avg_metrics"][m] for m in metrics]
        values += values[:1]  # 闭合

        ax.plot(angles, values, "o-", linewidth=2, label=result["config"]["experiment_name"], color=colors[idx])
        ax.fill(angles, values, alpha=0.1, color=colors[idx])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.capitalize() for m in metrics])
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))

    plt.title("消融实验雷达图对比", fontsize=14)
    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "radar_comparison.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()

    print(f"雷达图已保存至: {save_dir}/radar_comparison.png")


def plot_heatmap(results: dict, save_dir: str):
    """
    绘制指标热力图

    Args:
        results: 实验结果字典
        save_dir: 保存目录
    """
    metrics = ["accuracy", "precision", "recall", "f1", "auc"]

    # 准备数据
    data = []
    row_labels = []

    for key, result in results.items():
        if "error" in result:
            continue
        row_labels.append(result["config"]["experiment_name"])
        data.append([result["avg_metrics"][m] for m in metrics])

    df = pd.DataFrame(data, index=row_labels, columns=[m.capitalize() for m in metrics])

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        df,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        ax=ax,
        vmin=0.5,
        vmax=1.0,
        linewidths=0.5,
    )
    ax.set_title("消融实验指标热力图")
    ax.set_xlabel("指标")
    ax.set_ylabel("实验")

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "metrics_heatmap.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()

    print(f"热力图已保存至: {save_dir}/metrics_heatmap.png")


def statistical_significance_test(
    results: dict, baseline_key: str = "Exp0_Baseline"
):
    """
    统计显著性检验

    使用配对t检验比较各实验与baseline

    Args:
        results: 实验结果字典
        baseline_key: baseline实验键

    Returns:
        检验结果DataFrame
    """
    if baseline_key not in results:
        print(f"警告: 未找到baseline实验 {baseline_key}")
        return None

    baseline = results[baseline_key]
    baseline_metrics = baseline["best_val_metrics_per_run"]

    metrics = ["accuracy", "precision", "recall", "f1", "auc"]
    rows = []

    for key, result in results.items():
        if key == baseline_key or "error" in result:
            continue

        exp_metrics = result["best_val_metrics_per_run"]

        row = {
            "实验": result["config"]["experiment_name"],
        }

        for m in metrics:
            baseline_values = [run[m] for run in baseline_metrics]
            exp_values = [run[m] for run in exp_metrics]

            # 配对t检验
            t_stat, p_value = stats.ttest_rel(exp_values, baseline_values)

            # Wilcoxon符号秩检验 (非参数检验)
            try:
                w_stat, w_p_value = stats.wilcoxon(exp_values, baseline_values)
            except:
                w_p_value = np.nan

            row[f"{m}_t_pvalue"] = p_value
            row[f"{m}_wilcoxon_pvalue"] = w_p_value
            row[f"{m}_significant"] = "✓" if p_value < 0.05 else "✗"

        rows.append(row)

    return pd.DataFrame(rows)


def generate_latex_table(summary_df: pd.DataFrame) -> str:
    """
    生成LaTeX表格

    Args:
        summary_df: 汇总DataFrame

    Returns:
        LaTeX表格字符串
    """
    # 选择要显示的列
    display_cols = ["实验ID", "实验名称", "数据增强", "Accuracy", "Precision", "Recall", "F1", "AUC"]
    df = summary_df[display_cols].copy()

    # 重命名列
    df.columns = ["ID", "Experiment", "Aug", "Acc", "Prec", "Rec", "F1", "AUC"]

    latex = df.to_latex(
        index=False,
        escape=False,
        caption="Ablation Study Results",
        label="tab:ablation",
        column_format="l l c c c c c c",
    )

    return latex


def analyze_results(result_dir: str, save_dir: str = None):
    """
    分析消融实验结果

    Args:
        result_dir: 结果目录
        save_dir: 分析结果保存目录，默认为result_dir下的analysis
    """
    if save_dir is None:
        save_dir = os.path.join(result_dir, "analysis")
    os.makedirs(save_dir, exist_ok=True)

    print(f"加载结果: {result_dir}")
    results = load_experiment_results(result_dir)

    if not results:
        print("未找到实验结果文件")
        return

    print(f"找到 {len(results)} 个实验结果")

    # 生成汇总表
    print("\n生成汇总表...")
    summary_df = generate_summary_table(results)

    # 保存表格
    summary_df_display = summary_df.drop(
        columns=[c for c in summary_df.columns if c.startswith("_")]
    )
    summary_df_display.to_excel(
        os.path.join(save_dir, "summary_table.xlsx"), index=False
    )
    summary_df_display.to_csv(os.path.join(save_dir, "summary_table.csv"), index=False)

    # 生成LaTeX表格
    latex_table = generate_latex_table(summary_df)
    with open(os.path.join(save_dir, "summary_table.tex"), "w", encoding="utf-8") as f:
        f.write(latex_table)

    print(f"汇总表已保存至: {save_dir}")

    # 绘制图表
    print("\n绘制对比图...")
    plot_metrics_comparison(results, save_dir)
    plot_radar_chart(results, save_dir)
    plot_heatmap(results, save_dir)

    # 统计检验
    print("\n进行统计显著性检验...")
    sig_df = statistical_significance_test(results)
    if sig_df is not None:
        sig_df.to_excel(
            os.path.join(save_dir, "significance_test.xlsx"), index=False
        )
        print(f"显著性检验结果已保存至: {save_dir}")

    print("\n分析完成!")

    return summary_df


def main():
    parser = argparse.ArgumentParser(description="消融实验结果分析")

    parser.add_argument(
        "--result_dir",
        type=str,
        required=True,
        help="实验结果目录",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="分析结果保存目录",
    )

    args = parser.parse_args()

    analyze_results(args.result_dir, args.save_dir)


if __name__ == "__main__":
    main()
