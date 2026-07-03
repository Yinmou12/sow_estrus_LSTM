import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Circle


def plot_AST_horizontal():
    np.random.seed(123)

    n_blue = 60
    blue_x = np.random.uniform(1, 5.5, n_blue)
    blue_y = np.random.uniform(1, 5.5, n_blue)
    mask_blue = (blue_x + blue_y < 7.5) | (np.random.rand(n_blue) < 0.2)
    blue_x = blue_x[mask_blue]
    blue_y = blue_y[mask_blue]

    n_red = 36
    red_x = np.random.uniform(4.5, 8, n_red)
    red_y = np.random.uniform(4.5, 8, n_red)
    border_red_x = np.array([3.8, 4.2])
    border_red_y = np.array([3.5, 4.0])
    red_x = np.concatenate([red_x, border_red_x])
    red_y = np.concatenate([red_y, border_red_y])

    fig, axs = plt.subplots(1, 3, figsize=(18, 6), dpi=1000)

    # Stage 1: ADASYN
    ax = axs[0]
    ax.scatter(
        blue_x, blue_y, c="#4A90E2", s=50, label="Majority", edgecolors="k", zorder=2
    )
    ax.scatter(
        red_x[:-2],
        red_y[:-2],
        c="#E94A4A",
        s=50,
        label="Positive example",
        edgecolors="k",
        zorder=2,
    )
    ax.scatter(
        border_red_x, border_red_y, c="#E94A4A", s=60, edgecolors="k", lw=1.5, zorder=3
    )

    for bx, by in zip(border_red_x, border_red_y):
        ax.add_patch(
            Circle(
                (bx, by),
                0.4,
                edgecolor="#FF9900",
                facecolor="none",
                lw=2,
                linestyle="--",
                zorder=4,
            )
        )

    adasyn_x = []
    adasyn_y = []
    for bx, by in zip(border_red_x, border_red_y):
        for _ in range(3):
            rx = bx + np.random.uniform(-0.3, 0.3)
            ry = by + np.random.uniform(-0.3, 0.3)
            adasyn_x.append(rx)
            adasyn_y.append(ry)
            ax.add_patch(
                Circle(
                    (rx, ry),
                    0.2,
                    edgecolor="none",
                    facecolor="#E94A4A",
                    alpha=0.2,
                    zorder=1,
                )
            )
    adasyn_x = np.array(adasyn_x)
    adasyn_y = np.array(adasyn_y)

    ax.scatter(
        adasyn_x,
        adasyn_y,
        c="#E94A4A",
        s=40,
        edgecolors="k",
        alpha=0.7,
        marker="^",
        label="ADASYN Augmented",
        zorder=2,
    )

    ax.set_title("Stage 1: ADASYN", fontsize=14, fontweight="bold", pad=15)
    ax.legend(loc="upper left")
    ax.grid(True, linestyle=":", alpha=0.6)

    # Stage 2: SMOTE
    ax = axs[1]
    all_blue_x_s1 = blue_x.copy()
    all_blue_y_s1 = blue_y.copy()
    all_red_x_s1 = np.concatenate([red_x, adasyn_x])
    all_red_y_s1 = np.concatenate([red_y, adasyn_y])

    smote_blue_x = []
    smote_blue_y = []
    smote_blue_pairs = []
    for i in range(len(all_blue_x_s1) - 1):
        if np.random.rand() < 0.45:
            smote_blue_x.append((all_blue_x_s1[i] + all_blue_x_s1[i + 1]) / 2)
            smote_blue_y.append((all_blue_y_s1[i] + all_blue_y_s1[i + 1]) / 2)
            smote_blue_pairs.append((i, i + 1))

    smote_red_x = []
    smote_red_y = []
    smote_red_pairs = []
    for i in range(len(all_red_x_s1)):
        for j in range(i + 1, len(all_red_x_s1)):
            dist = np.sqrt(
                (all_red_x_s1[i] - all_red_x_s1[j]) ** 2
                + (all_red_y_s1[i] - all_red_y_s1[j]) ** 2
            )
            if dist < 2.0 and np.random.rand() < 0.25:
                smote_red_x.append((all_red_x_s1[i] + all_red_x_s1[j]) / 2)
                smote_red_y.append((all_red_y_s1[i] + all_red_y_s1[j]) / 2)
                smote_red_pairs.append((i, j))
    smote_blue_x = np.array(smote_blue_x)
    smote_blue_y = np.array(smote_blue_y)
    smote_red_x = np.array(smote_red_x)
    smote_red_y = np.array(smote_red_y)

    for i, j in smote_blue_pairs:
        ax.plot(
            [all_blue_x_s1[i], all_blue_x_s1[j]],
            [all_blue_y_s1[i], all_blue_y_s1[j]],
            c="#4A90E2",
            linestyle="--",
            alpha=0.22,
            lw=0.8,
            zorder=1,
        )
    for i, j in smote_red_pairs:
        ax.plot(
            [all_red_x_s1[i], all_red_x_s1[j]],
            [all_red_y_s1[i], all_red_y_s1[j]],
            c="#E94A4A",
            linestyle="--",
            alpha=0.25,
            lw=0.8,
            zorder=1,
        )

    ax.scatter(
        all_blue_x_s1, all_blue_y_s1, c="#4A90E2", s=50, edgecolors="k", zorder=2
    )
    ax.scatter(
        smote_blue_x,
        smote_blue_y,
        c="#4A90E2",
        s=48,
        edgecolors="k",
        alpha=0.65,
        marker="D",
        label="SMOTE Majority",
        zorder=2,
    )
    ax.scatter(all_red_x_s1, all_red_y_s1, c="#E94A4A", s=50, edgecolors="k", zorder=2)
    ax.scatter(
        smote_red_x,
        smote_red_y,
        c="#E94A4A",
        s=52,
        edgecolors="k",
        alpha=0.65,
        marker="^",
        label="SMOTE Minority",
        zorder=2,
    )

    ax.set_title("Stage 2: SMOTE", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 9)
    ax.legend(loc="upper left")
    ax.grid(True, linestyle=":", alpha=0.6)

    # Stage 3: Tomek Links
    ax = axs[2]
    final_blue_x = np.concatenate([all_blue_x_s1, smote_blue_x])
    final_blue_y = np.concatenate([all_blue_y_s1, smote_blue_y])
    final_red_x = np.concatenate([all_red_x_s1, smote_red_x])
    final_red_y = np.concatenate([all_red_y_s1, smote_red_y])

    tomek_pair_specs = [
        {"red_target": (3.8, 3.5), "blue_target": (3.6, 3.3), "delete_majority": True},
        {"red_target": (4.2, 4.0), "blue_target": (4.4, 4.2), "delete_majority": True},
        {"red_target": (4.8, 4.5), "blue_target": (4.6, 4.7), "delete_majority": False},
    ]

    resolved_pairs = []
    used_red = set()
    used_blue = set()
    for spec in tomek_pair_specs:
        red_dist = (final_red_x - spec["red_target"][0]) ** 2 + (
            final_red_y - spec["red_target"][1]
        ) ** 2
        blue_dist = (final_blue_x - spec["blue_target"][0]) ** 2 + (
            final_blue_y - spec["blue_target"][1]
        ) ** 2
        if used_red:
            red_dist = red_dist.copy()
            red_dist[list(used_red)] = np.inf
        if used_blue:
            blue_dist = blue_dist.copy()
            blue_dist[list(used_blue)] = np.inf
        red_idx = int(np.argmin(red_dist))
        blue_idx = int(np.argmin(blue_dist))
        used_red.add(red_idx)
        used_blue.add(blue_idx)
        resolved_pairs.append(
            {
                "red_idx": red_idx,
                "blue_idx": blue_idx,
                "delete_majority": spec["delete_majority"],
            }
        )

    delete_blue_indices = [
        pair["blue_idx"] for pair in resolved_pairs if pair["delete_majority"]
    ]
    keep_blue_mask = np.ones(len(final_blue_x), dtype=bool)
    keep_blue_mask[delete_blue_indices] = False
    keep_red_mask = np.ones(len(final_red_x), dtype=bool)

    ax.scatter(
        final_blue_x[keep_blue_mask],
        final_blue_y[keep_blue_mask],
        c="#4A90E2",
        s=50,
        edgecolors="k",
        label="Majority Kept",
        zorder=2,
    )
    ax.scatter(
        final_red_x[keep_red_mask],
        final_red_y[keep_red_mask],
        c="#E94A4A",
        s=50,
        edgecolors="k",
        label="Minority Kept",
        zorder=2,
    )

    for pair_no, pair in enumerate(resolved_pairs, start=1):
        rx, ry = final_red_x[pair["red_idx"]], final_red_y[pair["red_idx"]]
        bx, by = final_blue_x[pair["blue_idx"]], final_blue_y[pair["blue_idx"]]
        removed = pair["delete_majority"]
        ring_color = "#666666" if removed else "#2ECC71"

        ax.plot(
            [rx, bx],
            [ry, by],
            c=ring_color,
            linestyle="--",
            alpha=0.55,
            lw=1.0,
            zorder=3,
        )
        ax.add_patch(
            Circle(
                ((rx + bx) / 2, (ry + by) / 2),
                0.42,
                edgecolor=ring_color,
                facecolor="none",
                lw=1.5,
                linestyle="--",
                zorder=4,
            )
        )
        ax.scatter(rx, ry, c="#E94A4A", s=64, edgecolors="k", zorder=5)

        if removed:
            ax.scatter(
                bx,
                by,
                c="#4A90E2",
                s=92,
                edgecolors="k",
                marker="X",
                alpha=0.38,
                label="Majority Removed" if pair_no == 1 else None,
                zorder=5,
            )
            ax.annotate(
                f"T{pair_no}: remove B",
                ((rx + bx) / 2, (ry + by) / 2),
                xytext=(6, 7),
                textcoords="offset points",
                fontsize=5.2,
                color="#444444",
                zorder=6,
            )
        else:
            ax.scatter(
                bx,
                by,
                c="#4A90E2",
                s=64,
                edgecolors="k",
                marker="s",
                label="Tomek Pair Kept",
                zorder=5,
            )
            ax.annotate(
                f"T{pair_no}: kept",
                ((rx + bx) / 2, (ry + by) / 2),
                xytext=(6, 7),
                textcoords="offset points",
                fontsize=5.2,
                color="#1F8A4C",
                zorder=6,
            )

    bd_x = np.linspace(2, 7, 100)
    bd_y = -1.0 * bd_x + 9.0
    ax.plot(
        bd_x,
        bd_y,
        color="#2ECC71",
        linestyle="-",
        lw=2.5,
        label="Decision Boundary",
        zorder=4,
    )

    ax.set_title("Stage 3: Tomek Links", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 9)
    ax.legend(loc="upper left")
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(
        os.path.join(IMAGE_SAVE_PATH, "pipeline_evolution_horizontal_1.svg"),
        format="svg",
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_AST_adasyn():
    np.random.seed(123)

    n_blue = 60
    blue_x = np.random.uniform(1, 5.5, n_blue)
    blue_y = np.random.uniform(1, 5.5, n_blue)
    mask_blue = (blue_x + blue_y < 7.5) | (np.random.rand(n_blue) < 0.2)
    blue_x = blue_x[mask_blue]
    blue_y = blue_y[mask_blue]

    n_red = 36
    red_x = np.random.uniform(4.5, 8, n_red)
    red_y = np.random.uniform(4.5, 8, n_red)
    border_red_x = np.array([3.8, 4.2])
    border_red_y = np.array([3.5, 4.0])
    red_x = np.concatenate([red_x, border_red_x])
    red_y = np.concatenate([red_y, border_red_y])

    adasyn_x = []
    adasyn_y = []
    for bx, by in zip(border_red_x, border_red_y):
        for _ in range(3):
            adasyn_x.append(bx + np.random.uniform(-0.3, 0.3))
            adasyn_y.append(by + np.random.uniform(-0.3, 0.3))
    adasyn_x = np.array(adasyn_x)
    adasyn_y = np.array(adasyn_y)

    plt.figure(figsize=(6, 5.5), dpi=1000)
    ax = plt.gca()
    ax.scatter(
        blue_x,
        blue_y,
        c="#4A90E2",
        s=50,
        label="Negative example",
        edgecolors="k",
        zorder=2,
    )
    ax.scatter(
        red_x[:-2],
        red_y[:-2],
        c="#E94A4A",
        s=50,
        label="Positive example",
        edgecolors="k",
        zorder=2,
    )
    ax.scatter(
        border_red_x, border_red_y, c="#E94A4A", s=60, edgecolors="k", lw=1.5, zorder=3
    )

    for bx, by in zip(border_red_x, border_red_y):
        ax.add_patch(
            Circle(
                (bx, by),
                0.4,
                edgecolor="#FF9900",
                facecolor="none",
                lw=2,
                linestyle="--",
                zorder=4,
            )
        )

    for rx, ry in zip(adasyn_x, adasyn_y):
        ax.add_patch(
            Circle(
                (rx, ry),
                0.22,
                edgecolor="none",
                facecolor="#E94A4A",
                alpha=0.15,
                zorder=1,
            )
        )
    ax.scatter(
        adasyn_x,
        adasyn_y,
        c="#E94A4A",
        s=45,
        edgecolors="k",
        alpha=0.8,
        marker="^",
        label="ADASYN Augmented",
        zorder=2,
    )

    ax.set_title("Stage 1: ADASYN", fontsize=12, fontweight="bold", pad=10)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(
        os.path.join(IMAGE_SAVE_PATH, "stage1_adasyn.svg"),
        format="svg",
        bbox_inches="tight",
    )
    plt.close()


def plot_AST_smote():
    color_maj_ori = "#0072B2"
    color_maj_smote = "#56B4E9"
    color_min_ori = "#D55E00"
    color_min_smote = "#E69F00"
    line_maj = "#0072B2"
    line_min = "#D55E00"

    np.random.seed(123)

    n_blue = 60
    blue_x = np.random.uniform(1, 5.5, n_blue)
    blue_y = np.random.uniform(1, 5.5, n_blue)
    mask_blue = (blue_x + blue_y < 7.5) | (np.random.rand(n_blue) < 0.2)
    blue_x = blue_x[mask_blue]
    blue_y = blue_y[mask_blue]

    n_red = 36
    red_x = np.random.uniform(4.5, 8, n_red)
    red_y = np.random.uniform(4.5, 8, n_red)
    border_red_x = np.array([3.8, 4.2])
    border_red_y = np.array([3.5, 4.0])
    red_x = np.concatenate([red_x, border_red_x])
    red_y = np.concatenate([red_y, border_red_y])

    adasyn_x = []
    adasyn_y = []
    for bx, by in zip(border_red_x, border_red_y):
        for _ in range(3):
            adasyn_x.append(bx + np.random.uniform(-0.3, 0.3))
            adasyn_y.append(by + np.random.uniform(-0.3, 0.3))
    adasyn_x = np.array(adasyn_x)
    adasyn_y = np.array(adasyn_y)

    all_blue_x_s1 = blue_x.copy()
    all_blue_y_s1 = blue_y.copy()
    all_red_x_s1 = np.concatenate([red_x, adasyn_x])
    all_red_y_s1 = np.concatenate([red_y, adasyn_y])

    smote_blue_x = []
    smote_blue_y = []
    smote_blue_pairs = []
    for i in range(len(all_blue_x_s1) - 1):
        if np.random.rand() < 0.45:
            smote_blue_x.append((all_blue_x_s1[i] + all_blue_x_s1[i + 1]) / 2)
            smote_blue_y.append((all_blue_y_s1[i] + all_blue_y_s1[i + 1]) / 2)
            smote_blue_pairs.append((i, i + 1))

    smote_red_x = []
    smote_red_y = []
    smote_red_pairs = []
    for i in range(len(all_red_x_s1)):
        for j in range(i + 1, len(all_red_x_s1)):
            dist = np.sqrt(
                (all_red_x_s1[i] - all_red_x_s1[j]) ** 2
                + (all_red_y_s1[i] - all_red_y_s1[j]) ** 2
            )
            if dist < 2.0 and np.random.rand() < 0.25:
                smote_red_x.append((all_red_x_s1[i] + all_red_x_s1[j]) / 2)
                smote_red_y.append((all_red_y_s1[i] + all_red_y_s1[j]) / 2)
                smote_red_pairs.append((i, j))
    smote_blue_x = np.array(smote_blue_x)
    smote_blue_y = np.array(smote_blue_y)
    smote_red_x = np.array(smote_red_x)
    smote_red_y = np.array(smote_red_y)

    plt.figure(figsize=(6, 5.5), dpi=1000)
    ax = plt.gca()
    for i, j in smote_blue_pairs:
        ax.plot(
            [all_blue_x_s1[i], all_blue_x_s1[j]],
            [all_blue_y_s1[i], all_blue_y_s1[j]],
            c=line_maj,
            linestyle="--",
            alpha=0.4,
            lw=0.8,
            zorder=1,
        )
    for i, j in smote_red_pairs:
        ax.plot(
            [all_red_x_s1[i], all_red_x_s1[j]],
            [all_red_y_s1[i], all_red_y_s1[j]],
            c=line_min,
            linestyle="--",
            alpha=0.4,
            lw=0.8,
            zorder=1,
        )

    ax.scatter(
        all_blue_x_s1,
        all_blue_y_s1,
        c=color_maj_ori,
        s=50,
        label="Existing negative example",
        edgecolors="k",
        zorder=2,
    )
    ax.scatter(
        smote_blue_x,
        smote_blue_y,
        c=color_maj_smote,
        s=48,
        edgecolors="k",
        alpha=0.8,
        marker="D",
        label="SMOTE negative example",
        zorder=2,
    )
    ax.scatter(
        all_red_x_s1,
        all_red_y_s1,
        c=color_min_ori,
        s=50,
        label="Existing positive example",
        edgecolors="k",
        zorder=2,
    )
    ax.scatter(
        smote_red_x,
        smote_red_y,
        c=color_min_smote,
        s=52,
        edgecolors="k",
        alpha=0.8,
        marker="^",
        label="SMOTE positive example",
        zorder=2,
    )

    ax.set_title("Stage 2: SMOTE", fontsize=12, fontweight="bold", pad=10)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(
        os.path.join(IMAGE_SAVE_PATH, "stage2_smote.svg"),
        format="svg",
        bbox_inches="tight",
    )
    plt.close()


def plot_AST_tomek_links():
    np.random.seed(123)

    n_blue = 60
    blue_x = np.random.uniform(1, 5.5, n_blue)
    blue_y = np.random.uniform(1, 5.5, n_blue)
    mask_blue = (blue_x + blue_y < 7.5) | (np.random.rand(n_blue) < 0.2)
    blue_x = blue_x[mask_blue]
    blue_y = blue_y[mask_blue]

    n_red = 36
    red_x = np.random.uniform(4.5, 8, n_red)
    red_y = np.random.uniform(4.5, 8, n_red)
    border_red_x = np.array([3.8, 4.2])
    border_red_y = np.array([3.5, 4.0])
    red_x = np.concatenate([red_x, border_red_x])
    red_y = np.concatenate([red_y, border_red_y])

    adasyn_x = []
    adasyn_y = []
    for bx, by in zip(border_red_x, border_red_y):
        for _ in range(3):
            adasyn_x.append(bx + np.random.uniform(-0.3, 0.3))
            adasyn_y.append(by + np.random.uniform(-0.3, 0.3))
    adasyn_x = np.array(adasyn_x)
    adasyn_y = np.array(adasyn_y)

    all_blue_x_s1 = blue_x.copy()
    all_blue_y_s1 = blue_y.copy()
    all_red_x_s1 = np.concatenate([red_x, adasyn_x])
    all_red_y_s1 = np.concatenate([red_y, adasyn_y])

    smote_blue_x = []
    smote_blue_y = []
    for i in range(len(all_blue_x_s1) - 1):
        if np.random.rand() < 0.45:
            smote_blue_x.append((all_blue_x_s1[i] + all_blue_x_s1[i + 1]) / 2)
            smote_blue_y.append((all_blue_y_s1[i] + all_blue_y_s1[i + 1]) / 2)

    smote_red_x = []
    smote_red_y = []
    for i in range(len(all_red_x_s1)):
        for j in range(i + 1, len(all_red_x_s1)):
            dist = np.sqrt(
                (all_red_x_s1[i] - all_red_x_s1[j]) ** 2
                + (all_red_y_s1[i] - all_red_y_s1[j]) ** 2
            )
            if dist < 2.0 and np.random.rand() < 0.25:
                smote_red_x.append((all_red_x_s1[i] + all_red_x_s1[j]) / 2)
                smote_red_y.append((all_red_y_s1[i] + all_red_y_s1[j]) / 2)
    smote_blue_x = np.array(smote_blue_x)
    smote_blue_y = np.array(smote_blue_y)
    smote_red_x = np.array(smote_red_x)
    smote_red_y = np.array(smote_red_y)

    final_blue_x = np.concatenate([all_blue_x_s1, smote_blue_x])
    final_blue_y = np.concatenate([all_blue_y_s1, smote_blue_y])
    final_red_x = np.concatenate([all_red_x_s1, smote_red_x])
    final_red_y = np.concatenate([all_red_y_s1, smote_red_y])

    tomek_pair_specs = [
        {"red_target": (3.8, 3.5), "blue_target": (3.6, 3.3), "delete_majority": True},
        {"red_target": (4.2, 4.0), "blue_target": (4.4, 4.2), "delete_majority": True},
        {"red_target": (4.8, 4.5), "blue_target": (4.6, 4.7), "delete_majority": False},
    ]

    resolved_pairs = []
    used_red = set()
    used_blue = set()
    for spec in tomek_pair_specs:
        red_dist = (final_red_x - spec["red_target"][0]) ** 2 + (
            final_red_y - spec["red_target"][1]
        ) ** 2
        blue_dist = (final_blue_x - spec["blue_target"][0]) ** 2 + (
            final_blue_y - spec["blue_target"][1]
        ) ** 2
        if used_red:
            red_dist = red_dist.copy()
            red_dist[list(used_red)] = np.inf
        if used_blue:
            blue_dist = blue_dist.copy()
            blue_dist[list(used_blue)] = np.inf
        red_idx = int(np.argmin(red_dist))
        blue_idx = int(np.argmin(blue_dist))
        used_red.add(red_idx)
        used_blue.add(blue_idx)
        resolved_pairs.append(
            {
                "red_idx": red_idx,
                "blue_idx": blue_idx,
                "delete_majority": spec["delete_majority"],
            }
        )

    delete_blue_indices = [
        pair["blue_idx"] for pair in resolved_pairs if pair["delete_majority"]
    ]
    keep_blue_mask = np.ones(len(final_blue_x), dtype=bool)
    keep_blue_mask[delete_blue_indices] = False
    keep_red_mask = np.ones(len(final_red_x), dtype=bool)

    plt.figure(figsize=(6, 5.5), dpi=1000)
    ax = plt.gca()
    ax.scatter(
        final_blue_x[keep_blue_mask],
        final_blue_y[keep_blue_mask],
        c="#4A90E2",
        s=50,
        edgecolors="k",
        label="Negative example kept",
        zorder=2,
    )
    ax.scatter(
        final_red_x[keep_red_mask],
        final_red_y[keep_red_mask],
        c="#E94A4A",
        s=50,
        edgecolors="k",
        label="Positive example kept",
        zorder=2,
    )

    for pair_no, pair in enumerate(resolved_pairs, start=1):
        rx, ry = final_red_x[pair["red_idx"]], final_red_y[pair["red_idx"]]
        bx, by = final_blue_x[pair["blue_idx"]], final_blue_y[pair["blue_idx"]]
        removed = pair["delete_majority"]
        ring_color = "#666666" if removed else "#2ECC71"

        ax.plot(
            [rx, bx],
            [ry, by],
            c=ring_color,
            linestyle="--",
            alpha=0.55,
            lw=1.0,
            zorder=3,
        )
        ax.add_patch(
            Circle(
                ((rx + bx) / 2, (ry + by) / 2),
                0.42,
                edgecolor=ring_color,
                facecolor="none",
                lw=1.5,
                linestyle="--",
                zorder=4,
            )
        )
        ax.scatter(rx, ry, c="#E94A4A", s=64, edgecolors="k", zorder=5)

        if removed:
            ax.scatter(
                bx,
                by,
                c="#4A90E2",
                s=92,
                edgecolors="k",
                marker="X",
                alpha=0.6,
                label="Negative example removed" if pair_no == 1 else None,
                zorder=5,
            )
            ax.annotate(
                f"Removed",
                ((rx + bx) / 2, (ry + by) / 2 - 0.5),
                xytext=(6, 7),
                textcoords="offset points",
                fontsize=12,
                color="#444444",
                zorder=6,
            )
        else:
            ax.scatter(
                bx,
                by,
                c="#4A90E2",
                s=64,
                edgecolors="k",
                marker="s",
                label="Tomek Pair Kept",
                zorder=5,
            )
            ax.annotate(
                f" ",
                ((rx + bx) / 2, (ry + by) / 2),
                xytext=(6, 7),
                textcoords="offset points",
                fontsize=6,
                color="#1F8A4C",
                zorder=6,
            )

    bd_x = np.linspace(2, 7, 100)
    bd_y = -1.0 * bd_x + 9.0
    ax.plot(
        bd_x,
        bd_y,
        color="#2ECC71",
        linestyle="-",
        lw=2.5,
        label="Decision Boundary",
        zorder=4,
    )

    ax.set_title("Stage 3: Tomek Links", fontsize=12, fontweight="bold", pad=10)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(
        os.path.join(IMAGE_SAVE_PATH, "stage3_tomek.svg"),
        format="svg",
        bbox_inches="tight",
    )
    plt.close()
