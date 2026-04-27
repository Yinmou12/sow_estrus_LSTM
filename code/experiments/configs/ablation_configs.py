# -*- coding: utf-8 -*-
"""
消融实验配置文件

定义8组消融实验的数据增强策略配置和模型超参数配置。
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class DataAugmentationConfig:
    """数据增强配置"""

    use_adasyn: bool = False
    use_smote: bool = False
    use_tomek_links: bool = False

    # ADASYN 参数
    adasyn_threshold: float = 0.5
    adasyn_gamma: float = 1.0
    adasyn_k: int = 5

    # SMOTE 参数
    smote_amount: int = 800
    smote_k: int = 5

    # TomekLinks 参数
    tomek_k: int = 3


@dataclass
class ModelConfig:
    """模型配置"""

    model_type: str = "EstrusLSTM"  # EstrusLSTM, EstrusLSTM_Attn, EstrusGRU
    input_size: int = 2  # 输入特征数
    hidden_size: int = 64
    num_layers: int = 4
    dropout_rate: float = 0.2

    # 训练参数
    batch_size: int = 32
    learning_rate: float = 0.0005
    weight_decay: float = 1e-4
    num_epochs: int = 100
    early_patience: int = 7

    # 优化器参数
    optimizer_type: str = "Adam"  # Adam, AdamW, SGD
    scheduler_type: str = "ReduceLROnPlateau"  # ReduceLROnPlateau, CosineAnnealingLR

    # 损失函数
    loss_type: str = "BCELoss"  # BCELoss, BCEWithLogitsLoss, FocalLoss


@dataclass
class ExperimentConfig:
    """完整实验配置"""

    experiment_id: str
    experiment_name: str
    description: str

    data_augmentation: DataAugmentationConfig = field(
        default_factory=DataAugmentationConfig
    )
    model: ModelConfig = field(default_factory=ModelConfig)

    # 实验控制
    num_runs: int = 5  # 多次运行取平均
    random_seed: int = 42


# ============================================================================
# 消融实验配置定义
# ============================================================================

ABLATION_EXPERIMENTS: Dict[str, ExperimentConfig] = {
    # Exp0: Baseline - 无任何数据增强
    "Exp0_Baseline": ExperimentConfig(
        experiment_id="Exp0",
        experiment_name="Baseline",
        description="无数据增强，原始数据直接训练",
        data_augmentation=DataAugmentationConfig(
            use_adasyn=False,
            use_smote=False,
            use_tomek_links=False,
        ),
        model=ModelConfig(),
        num_runs=5,
    ),
    # Exp1: 仅ADASYN
    "Exp1_ADASYN": ExperimentConfig(
        experiment_id="Exp1",
        experiment_name="+ADASYN",
        description="仅使用ADASYN过采样",
        data_augmentation=DataAugmentationConfig(
            use_adasyn=True,
            use_smote=False,
            use_tomek_links=False,
        ),
        model=ModelConfig(),
        num_runs=5,
    ),
    # Exp2: 仅SMOTE
    "Exp2_SMOTE": ExperimentConfig(
        experiment_id="Exp2",
        experiment_name="+SMOTE",
        description="仅使用SMOTE过采样",
        data_augmentation=DataAugmentationConfig(
            use_adasyn=False,
            use_smote=True,
            use_tomek_links=False,
        ),
        model=ModelConfig(),
        num_runs=5,
    ),
    # Exp3: 仅TomekLinks
    "Exp3_TomekLinks": ExperimentConfig(
        experiment_id="Exp3",
        experiment_name="+TomekLinks",
        description="仅使用TomekLinks欠采样",
        data_augmentation=DataAugmentationConfig(
            use_adasyn=False,
            use_smote=False,
            use_tomek_links=True,
        ),
        model=ModelConfig(),
        num_runs=5,
    ),
    # Exp4: ADASYN + SMOTE
    "Exp4_ADASYN_SMOTE": ExperimentConfig(
        experiment_id="Exp4",
        experiment_name="ADASYN+SMOTE",
        description="ADASYN和SMOTE组合过采样",
        data_augmentation=DataAugmentationConfig(
            use_adasyn=True,
            use_smote=True,
            use_tomek_links=False,
        ),
        model=ModelConfig(),
        num_runs=5,
    ),
    # Exp5: ADASYN + TomekLinks
    "Exp5_ADASYN_TomekLinks": ExperimentConfig(
        experiment_id="Exp5",
        experiment_name="ADASYN+TomekLinks",
        description="ADASYN过采样 + TomekLinks欠采样",
        data_augmentation=DataAugmentationConfig(
            use_adasyn=True,
            use_smote=False,
            use_tomek_links=True,
        ),
        model=ModelConfig(),
        num_runs=5,
    ),
    # Exp6: SMOTE + TomekLinks
    "Exp6_SMOTE_TomekLinks": ExperimentConfig(
        experiment_id="Exp6",
        experiment_name="SMOTE+TomekLinks",
        description="SMOTE过采样 + TomekLinks欠采样",
        data_augmentation=DataAugmentationConfig(
            use_adasyn=False,
            use_smote=True,
            use_tomek_links=True,
        ),
        model=ModelConfig(),
        num_runs=5,
    ),
    # Exp7: Full Pipeline - 完整数据增强
    "Exp7_FullPipeline": ExperimentConfig(
        experiment_id="Exp7",
        experiment_name="Full Pipeline",
        description="ADASYN + SMOTE + TomekLinks 完整数据增强",
        data_augmentation=DataAugmentationConfig(
            use_adasyn=True,
            use_smote=True,
            use_tomek_links=True,
        ),
        model=ModelConfig(),
        num_runs=5,
    ),
}


def get_experiment_config(experiment_key: str) -> ExperimentConfig:
    """获取指定实验配置"""
    if experiment_key not in ABLATION_EXPERIMENTS:
        raise ValueError(
            f"未知的实验配置: {experiment_key}. "
            f"可用的配置: {list(ABLATION_EXPERIMENTS.keys())}"
        )
    return ABLATION_EXPERIMENTS[experiment_key]


def get_all_experiment_keys() -> List[str]:
    """获取所有实验配置的键"""
    return list(ABLATION_EXPERIMENTS.keys())


def print_experiment_summary():
    """打印所有实验配置摘要"""
    print("=" * 80)
    print("消融实验配置摘要")
    print("=" * 80)

    header = f"{'实验ID':<8} {'实验名称':<20} {'ADASYN':<8} {'SMOTE':<8} {'TomekLinks':<12} {'描述'}"
    print(header)
    print("-" * 80)

    for key, config in ABLATION_EXPERIMENTS.items():
        da = config.data_augmentation
        row = (
            f"{config.experiment_id:<8} "
            f"{config.experiment_name:<20} "
            f"{'✓' if da.use_adasyn else '✗':<8} "
            f"{'✓' if da.use_smote else '✗':<8} "
            f"{'✓' if da.use_tomek_links else '✗':<12} "
            f"{config.description}"
        )
        print(row)

    print("=" * 80)


if __name__ == "__main__":
    print_experiment_summary()
