import os

from plotter.plotter_AST import (
    plot_AST_horizontal,
    plot_AST_adasyn,
    plot_AST_smote,
    plot_AST_tomek_links,
)
from plotter.plotter_BO import plot_bayesian_convergence_figures
from plotter.plotter_bayesian_heatmap import plot_hidden_sizes_learning_rate_f1_heatmap
from plotter.plotter_final_eval import plot_final_test_evaluation_figures

IMAGE_SAVE_PATH = "D:\\_\u8bba\u6587\\Bi-LSTM\\pictures"

BO_EXPERIMENT_DIR = None
BO_HEATMAP_EXPERIMENT_DIR = r"D:\_Software_Projects\VSCode\scientific_research\sow_estrus\my_code\result\all_experiments\07_bilstm_ast_bayesian_tuning_2026_0701_2159"
MARK_BEST_IN_HEATMAP = False
FINAL_EVAL_EXPERIMENT_DIR = r"D:\_Software_Projects\VSCode\scientific_research\sow_estrus\my_code\result\all_experiments\03_final_model_comparison_ast_2026_0703_1323\run_05\Final_RNN_AST"


def draw_bayesian_convergence():
    output_paths = plot_bayesian_convergence_figures(BO_EXPERIMENT_DIR)
    for output_path in output_paths:
        print(f"Saved: {output_path}")


def draw_bayesian_hidden_lr_heatmap(mark_best=MARK_BEST_IN_HEATMAP):
    output_paths = plot_hidden_sizes_learning_rate_f1_heatmap(
        BO_HEATMAP_EXPERIMENT_DIR,
        mark_best=mark_best,
    )
    for output_path in output_paths:
        print(f"Saved: {output_path}")


def draw_final_test_evaluation():
    output_paths = plot_final_test_evaluation_figures(FINAL_EVAL_EXPERIMENT_DIR)
    for output_path in output_paths:
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    # Uncomment the figures you want to generate.
    # draw_bayesian_convergence()
    # draw_final_test_evaluation()
    # draw_bayesian_hidden_lr_heatmap()

    # plot_AST_horizontal()
    # plot_AST_adasyn()
    # plot_AST_smote()
    # plot_AST_tomek_links()
    pass
