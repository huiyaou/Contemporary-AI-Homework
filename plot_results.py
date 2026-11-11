import json
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figures"
BLUE = "#0F4D92"
GREEN = "#42949E"
GRAY = "#CFCECE"

plt.rcParams.update({
    "font.family": ["Microsoft YaHei", "Arial", "DejaVu Sans"],
    "font.size": 12,
    "axes.linewidth": 1.4,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.unicode_minus": False,
    "legend.frameon": False,
    "svg.fonttype": "none",
})


def save(fig, name):
    fig.tight_layout(pad=1.4)
    fig.savefig(OUT / f"{name}.png", dpi=300, facecolor="white")
    fig.savefig(OUT / f"{name}.svg", facecolor="white")
    plt.close(fig)


def draw_path(case, name, title):
    grid = case["grid"]
    path = case["runs"][1]["path"]
    rows, cols = len(grid), len(grid[0])
    fig, ax = plt.subplots(figsize=(5.6, 5.1))
    walls = [[int(cell == "X") for cell in row] for row in grid]
    ax.imshow(walls, cmap=ListedColormap(["#F5F7FA", "#4D4D4D"]), vmin=0, vmax=1)
    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
            if cell != ".":
                ax.text(c, r, str(cell), ha="center", va="center",
                        color="white" if cell == "X" else "#272727", fontsize=14, zorder=4)
    for (r1, c1), (r2, c2) in zip(path, path[1:]):
        ax.annotate("", xy=(c2, r2), xytext=(c1, r1),
                    arrowprops={"arrowstyle": "->", "color": BLUE, "lw": 2.4,
                                "shrinkA": 12, "shrinkB": 12})
    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_xlabel("列 col")
    ax.set_ylabel("行 row")
    ax.set_title(title, pad=12)
    save(fig, name)


def maze_scaling(cases):
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.4), sharey=True)
    for ax, density in zip(axes, (0.0, 0.15, 0.30)):
        for method, color, marker, label in ((0, "#767676", "s", "h=0"), (1, BLUE, "o", "曼哈顿距离")):
            groups = [[c["runs"][method]["expanded_nodes"] for c in cases
                       if c.get("size") == size and c.get("obstacle_probability") == density]
                      for size in (20, 40, 80)]
            centers = np.array([mean(group) for group in groups])
            errors = np.array([[center - min(group) for center, group in zip(centers, groups)],
                               [max(group) - center for center, group in zip(centers, groups)]])
            ax.errorbar((20, 40, 80), centers, yerr=errors, color=color,
                        marker=marker, lw=2.1, markersize=6, capsize=4, label=label,
                        markeredgecolor="#4D4D4D" if method == 0 else BLUE)
        ax.set_title(f"障碍概率 p={density:.2f}")
        ax.set_xticks((20, 40, 80))
        ax.set_xlabel("方形地图边长")
        ax.set_ylim(0, 7000)
        ax.grid(axis="y", alpha=0.15)
    axes[0].set_ylabel("扩展节点数")
    axes[0].legend(loc="upper left")
    save(fig, "q1_scaling")


def tree_scaling(cases):
    fig, ax = plt.subplots(figsize=(8.8, 4.7))
    settings = ((2, 6), (3, 4), (3, 6))
    x = np.arange(len(settings))
    for index, (algorithm, order, color, hatch, label) in enumerate((
        ("minimax", "left", GRAY, "", "Minimax"),
        ("alpha_beta", "left", BLUE, "", "Alpha-beta 正序"),
        ("alpha_beta", "reverse", GREEN, "//", "Alpha-beta 逆序"),
    )):
        groups = [[run["visited_nodes"] for c in cases
                   if c.get("branching_factor") == b and c.get("depth") == d
                   for run in c["runs"] if run["algorithm"] == algorithm and run["order"] == order]
                  for b, d in settings]
        centers = np.array([mean(group) for group in groups])
        errors = np.array([[center - min(group) for center, group in zip(centers, groups)],
                           [max(group) - center for center, group in zip(centers, groups)]])
        bars = ax.bar(x + (index - 1) * 0.24, centers, width=0.24, color=color,
                      edgecolor="#272727", linewidth=0.8, hatch=hatch, label=label,
                      yerr=errors, capsize=3)
        ax.bar_label(bars, labels=[f"{value:.1f}" for value in centers], padding=5, fontsize=10)
    ax.set_xticks(x, [f"b={b}, H={d}" for b, d in settings])
    ax.set_ylim(0, 1350)
    ax.set_ylabel("访问节点数")
    ax.set_xlabel("树的分支数与深度")
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.15)
    ax.set_axisbelow(True)
    save(fig, "q3_scaling")


def treasure_comparison(case):
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8))
    for index, detail in enumerate(case["depths"]):
        color = GREEN if index == 0 else BLUE
        offset = (index - 0.5) * 0.34
        values = list(detail["root_action_values"].values())
        bars = axes[0].bar(np.arange(4) + offset, values, width=0.34, color=color,
                           edgecolor="#272727", linewidth=0.7, label=f"H={detail['depth']}")
        axes[0].bar_label(bars, padding=3, fontsize=11)
        runs = detail["runs"]
        nodes = [runs[0]["visited_nodes"], runs[1]["visited_nodes"],
                 runs[3]["visited_nodes"], runs[5]["visited_nodes"]]
        bars = axes[1].bar(np.arange(4) + offset, nodes, width=0.34, color=color,
                           edgecolor="#272727", linewidth=0.7, label=f"H={detail['depth']}")
        axes[1].bar_label(bars, padding=3, fontsize=11)
    axes[0].axhline(0, color="#767676", linewidth=1)
    axes[0].set_xticks(range(4), ["T1", "T2", "T3", "T4"])
    axes[0].set_ylim(-9, 5)
    axes[0].set_ylabel("首步的最优收益差 U")
    axes[0].set_title("(a) 各首步的精确博弈值")
    axes[1].set_xticks(range(4), ["Minimax", "AB fixed", "AB reverse", "AB gain"])
    axes[1].set_ylim(0, 78)
    axes[1].set_ylabel("访问节点数")
    axes[1].set_title("(b) 算法与行动顺序")
    for ax in axes:
        ax.grid(axis="y", alpha=0.15)
        ax.set_axisbelow(True)
    axes[0].legend(loc="upper left")
    axes[1].legend(loc="upper right")
    save(fig, "q4_comparison")


def main():
    data = json.loads((ROOT / "results" / "experiments.json").read_text(encoding="utf-8"))
    OUT.mkdir(exist_ok=True)
    draw_path(data["q1"]["cases"][0], "q1_path", "题目一：7 步最短路径")
    draw_path(data["q2"]["cases"][0], "q2_path", "题目二：最小代价为 7 的路径")
    maze_scaling(data["q1"]["cases"])
    tree_scaling(data["q3"]["cases"])
    treasure_comparison(data["q4"]["cases"][0])
    print(f"Generated 5 PNG and 5 SVG figures in {OUT}")


if __name__ == "__main__":
    main()
