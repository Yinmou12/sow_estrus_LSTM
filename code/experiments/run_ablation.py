# -*- coding: utf-8 -*-
"""
消融实验批量运行脚本

按顺序执行所有消融实验并汇总结果。
"""

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

pd.set_option("future.no_silent_downcasting", True)

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.ablation_configs import (
    ABLATION_EXPERIMENTS,
    get_all_experiment_keys,
    print_experiment_summary,
)
from ablation_study import run_experiment


def run_all_experiments(
    data_path: str,
    result_dir: str,
    num_features: int = 2,
    num_runs: int = 5,
    experiment_keys: list = None,
    verbose: bool = True,
):
    """
    运行所有或指定的消融实验

    Args:
        data_path: 数据路径
        result_dir: 结果保存目录
        num_features: 特征数量
        num_runs: 每个实验的运行次数
        experiment_keys: 要运行的实验键列表，None表示全部
        verbose: 是否打印详细信息
    """
    # 创建结果目录
    timestamp = datetime.now().strftime("%Y_%m%d_%H%M")
    result_base = os.path.join(result_dir, f"ablation_{timestamp}")
    os.makedirs(result_base, exist_ok=True)

    # 获取要运行的实验
    if experiment_keys is None:
        experiment_keys = get_all_experiment_keys()

    print("=" * 80)
    print("消融实验批量运行")
    print("=" * 80)
    print(f"数据路径: {data_path}")
    print(f"结果目录: {result_base}")
    print(f"实验数量: {len(experiment_keys)}")
    print(f"每个实验运行次数: {num_runs}")
    print("=" * 80)

    # 保存实验配置
    config_summary = {
        "data_path": data_path,
        "result_dir": result_base,
        "num_features": num_features,
        "num_runs": num_runs,
        "experiments": experiment_keys,
        "start_time": datetime.now().isoformat(),
    }

    with open(os.path.join(result_base, "run_config.json"), "w") as f:
        json.dump(config_summary, f, indent=2)

    # 运行实验
    all_results = {}

    for i, key in enumerate(experiment_keys, 1):
        print(f"\n\n{'#' * 80}")
        print(f"进度: {i}/{len(experiment_keys)} - {key}")
        print(f"{'#' * 80}")

        config = ABLATION_EXPERIMENTS[key]
        config.num_runs = num_runs  # 覆盖运行次数

        try:
            result = run_experiment(
                config=config,
                data_path=data_path,
                result_dir=result_base,
                num_features=num_features,
                verbose=verbose,
            )
            all_results[key] = result

        except Exception as e:
            print(f"实验 {key} 失败: {e}")
            all_results[key] = {"error": str(e)}

    # 汇总结果
    print("\n\n" + "=" * 80)
    print("消融实验结果汇总")
    print("=" * 80)

    summary_table = []
    for key in experiment_keys:
        if key in all_results and "error" not in all_results[key]:
            result = all_results[key]
            avg = result["avg_metrics"]
            std = result["std_metrics"]
            summary_table.append(
                {
                    "实验": key,
                    "Accuracy": f"{avg['accuracy']:.4f}±{std['accuracy']:.4f}",
                    "Precision": f"{avg['precision']:.4f}±{std['precision']:.4f}",
                    "Recall": f"{avg['recall']:.4f}±{std['recall']:.4f}",
                    "F1": f"{avg['f1']:.4f}±{std['f1']:.4f}",
                    "AUC": f"{avg['auc']:.4f}±{std['auc']:.4f}",
                }
            )

    # 打印汇总表
    if summary_table:
        import pandas as pd

        summary_df = pd.DataFrame(summary_table)
        print(summary_df.to_string(index=False))

        # 保存汇总表
        summary_df.to_excel(
            os.path.join(result_base, "summary_table.xlsx"), index=False
        )
        summary_df.to_csv(os.path.join(result_base, "summary_table.csv"), index=False)

        print(f"\n汇总表已保存至: {result_base}")

    # 保存完整结果
    config_summary["end_time"] = datetime.now().isoformat()
    config_summary["status"] = "completed"

    with open(os.path.join(result_base, "run_config.json"), "w") as f:
        json.dump(config_summary, f, indent=2)

    print("\n所有实验完成!")

    return all_results


def run_single_experiment(
    experiment_key: str,
    data_path: str,
    result_dir: str,
    num_features: int = 2,
    num_runs: int = 5,
    verbose: bool = True,
):
    """运行单个实验"""
    from configs.ablation_configs import get_experiment_config

    config = get_experiment_config(experiment_key)
    config.num_runs = num_runs

    timestamp = datetime.now().strftime("%Y_%m%d_%H%M")
    result_base = os.path.join(result_dir, f"single_{timestamp}")
    os.makedirs(result_base, exist_ok=True)

    return run_experiment(
        config=config,
        data_path=data_path,
        result_dir=result_base,
        num_features=num_features,
        verbose=verbose,
    )


def main():
    parser = argparse.ArgumentParser(description="消融实验批量运行脚本")

    parser.add_argument(
        "--data_path",
        type=str,
        default=r"D:\_Software_Projects\VSCode\scientific_research\sow_estrus\my_code\data\splited_dataset",
        help="数据路径",
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default=r"D:\_Software_Projects\VSCode\scientific_research\sow_estrus\my_code\result\ablation",
        help="结果保存目录",
    )
    parser.add_argument("--num_runs", type=int, default=5, help="每个实验运行次数")
    parser.add_argument("--num_features", type=int, default=2, help="特征数量")
    parser.add_argument(
        "--experiments",
        type=str,
        nargs="+",
        default=None,
        help="要运行的实验，例如: Exp0_Baseline Exp7_FullPipeline",
    )
    parser.add_argument(
        "--list_experiments",
        action="store_true",
        help="列出所有可用实验并退出",
    )
    parser.add_argument("--quiet", action="store_true", help="减少输出信息")

    args = parser.parse_args()

    if args.list_experiments:
        print_experiment_summary()
        return

    if args.experiments:
        # 运行指定实验
        run_all_experiments(
            data_path=args.data_path,
            result_dir=args.result_dir,
            num_features=args.num_features,
            num_runs=args.num_runs,
            experiment_keys=args.experiments,
            verbose=not args.quiet,
        )
    else:
        # 运行全部实验
        run_all_experiments(
            data_path=args.data_path,
            result_dir=args.result_dir,
            num_features=args.num_features,
            num_runs=args.num_runs,
            verbose=not args.quiet,
        )


if __name__ == "__main__":
    main()
