#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import warnings

# 1. 屏蔽 PyTorch 加载时的 Future Warning
warnings.filterwarnings("ignore", category=FutureWarning)

# =========================
# 确保能引用到 util 目录
# =========================
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.abspath(os.path.join(CUR_DIR, "../.."))
if PROJ_DIR not in sys.path:
    sys.path.insert(0, PROJ_DIR)

from util.utils import robust_accuracy_curve

# =========================
# 配置路径 (MNIST)
# =========================
result_dir = os.path.join("Results", "Mnist", "Nature_Attack")
output_dir = os.path.join("Results", "Mnist", "Comparison", "L2_RSA")
os.makedirs(output_dir, exist_ok=True)

# =========================
# 全局绘图样式
# =========================
USE_TEX = True
PNG_DPI = 300

try:
    if USE_TEX:
        plt.rc("text", usetex=True)
        plt.rc("font", family="serif")
    else:
        plt.rc("text", usetex=False)
        plt.rc("font", family="sans-serif")
except Exception:
    print("[WARN] LaTeX not available, falling back to standard fonts.")
    plt.rc("text", usetex=False)

plt.rcParams["lines.linewidth"] = 1.2

fontsize = 9
plt.rcParams.update(
    {
        "axes.labelsize": fontsize,
        "xtick.labelsize": fontsize - 1,
        "ytick.labelsize": fontsize - 1,
        "axes.titlesize": fontsize,
        "legend.fontsize": fontsize - 2,
    }
)

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

YTICKS = [0, 0.1, 0.25, 0.5, 0.75, 1]
YTICKLABELS = [0, 10, 25, 50, 75, 100]

DISPLAY_NAME = {
    "SmallCNN_regular": "SmallCNN-Regular",
    "SmallCNN_ddn_l2": "SmallCNN-DDN-L2",
    "SmallCNN_trades_linf": "SmallCNN-TRADES",
    "IBP_large_linf": "CROWN-IBP",
}

MODELS_CONFIG = [
    ("SmallCNN_regular", 8),
    ("SmallCNN_ddn_l2", 8),
    ("SmallCNN_trades_linf", 8),
    ("IBP_large_linf", 8),
]

# ============================================================
# Attacks List 配置
# ============================================================
attacks_list = [
    # --- 实线组 (主要展示的算法) ---
    ("DF_l2_100", r"DF $\ell_2$ 100", "-", colors[0]),
    ("GSF_l2_100", r"GSF $\ell_2$ 100", "-", colors[1]),
    ("SDF_l2_100", r"SDF $\ell_2$ 100", "-", colors[2]),
    ("SGSF_l2_100", r"SGSF $\ell_2$ 100", "-", colors[3]),
    ("APGD_l2", r"$\mathrm{APGD}^\mathrm{DLR}_\mathrm{T}$ $\ell_2$", "-", colors[8]),
    ("FAB_l2_1000", r"FAB $\ell_2$ 1000", "-", colors[4]),
    ("FMN_l2_1000", r"FMN $\ell_2$ 1000", "-", colors[5]),
    ("DDN_l2_1000", r"DDN $\ell_2$ 1000", "-", colors[6]),
    ("ALMA_l2_1000", r"ALMA $\ell_2$ 1000", "-", colors[7]),
    ("C&W_l2_9x10000", r"C\&W $\ell_2$ $9\times 10000$", "-", colors[8]),
    # --- 点线组 (仅作背景对比，不进图例) ---
    ("FAB_l2_100", r"FAB $\ell_2$ 100", ":", colors[4]),
    ("FMN_l2_100", r"FMN $\ell_2$ 100", ":", colors[5]),
    ("DDN_l2_100", r"DDN $\ell_2$ 100", ":", colors[6]),
    ("ALMA_l2_100", r"ALMA $\ell_2$ 100", ":", colors[7]),
    ("C&W_l2_9x1000", r"C\&W $\ell_2$ $9\times 1000$", ":", colors[8]),
]


# =========================
# 工具函数：读取并生成曲线
# =========================
def load_curve(model: str, attack_id: str):
    path = os.path.join(result_dir, f"NT_{model}_{attack_id}.pt")
    if not os.path.exists(path):
        return None

    try:
        data = torch.load(path, map_location="cpu", weights_only=False)

        distances = data.get("distances")
        if not isinstance(distances, dict):
            return None

        # 兼容大小写 key
        adv_dist = distances.get("l2")
        if adv_dist is None:
            adv_dist = distances.get("L2")

        success = data.get("success")

        if adv_dist is None or success is None:
            return None

        x, y = robust_accuracy_curve(distances=adv_dist, successes=success)
        return x, y
    except Exception:
        return None


# =========================
# 主程序逻辑
# =========================
def main():
    # 预加载缓存
    print("Caching results from disk...")
    curve_cache = {}
    missing = []

    for model, _ in MODELS_CONFIG:
        curve_cache[model] = {}
        for attack_id, legend, linestyle, color in attacks_list:
            curve = load_curve(model, attack_id)
            if curve is None:
                missing.append((model, attack_id))
                continue
            x, y = curve
            curve_cache[model][attack_id] = (x, y, legend, linestyle, color)

    if missing:
        print(f"[INFO] Missing files/curves: {len(missing)} (showing up to 12)")
        for m, a in missing[:12]:
            print("   ", m, a)

    # 内部绘图函数
    def draw_one(
        ax,
        model,
        x_limit,
        *,
        show_title=False,
        show_xlabel=True,
        show_ylabel=True,
        show_legend=True,
    ):
        # 绘制所有曲线
        for attack_id, _, _, _ in attacks_list:
            if attack_id not in curve_cache[model]:
                continue
            x, y, legend, linestyle, color = curve_cache[model][attack_id]

            # 处理图例标签：实线显示，点线隐藏
            plot_label = legend if linestyle == "-" else "_nolegend_"
            ax.plot(x, y, label=plot_label, linestyle=linestyle, c=color)

        # 设置坐标轴
        ax.set_yticks(YTICKS)
        ax.set_xlim(0, x_limit)
        ax.set_ylim(0, 1)
        ax.grid(True, linestyle="--", c="lightgray", which="both")

        if show_title:
            ax.set_title(DISPLAY_NAME.get(model, model), pad=4)

        if show_ylabel:
            ax.set_ylabel(r"Robust Acc (\%)", labelpad=2)
            ax.set_yticklabels(YTICKLABELS)
        else:
            ax.set_ylabel("")
            ax.set_yticklabels([])

        if show_xlabel:
            ax.set_xlabel(r"$\ell_2$-norm", labelpad=2)
        else:
            ax.set_xlabel("")

        if show_legend:
            # 单张图时的图例位置
            ax.legend(loc="upper right", framealpha=0.85, fontsize=fontsize - 3)

    # =========================
    # 任务一：生成单模型图 (Single Figures)
    # =========================
    print("Generating single model curves (pdf+png)...")
    for model, x_limit in MODELS_CONFIG:
        fig, ax = plt.subplots(figsize=(3.4, 2.6))
        draw_one(
            ax,
            model,
            x_limit,
            show_title=False,
            show_xlabel=True,
            show_ylabel=True,
            show_legend=True,
        )

        plt.tight_layout()
        base = os.path.join(output_dir, f"single_mnist_l2_{model}")

        # 【优化点】 pad_inches=0 去除白边
        fig.savefig(base + ".pdf", bbox_inches="tight", pad_inches=0)
        fig.savefig(base + ".png", bbox_inches="tight", pad_inches=0, dpi=PNG_DPI)

        plt.close(fig)

    # =========================
    # 任务二：生成合并图 (Combined 2x2 Figure)
    # =========================
    print("Generating combined 2x2 figure...")

    # 初始化 2x2 画布
    fig, axes = plt.subplots(2, 2, figsize=(6.0, 4.5))
    axes_flat = axes.flatten()

    # 遍历每个模型，画在对应的子图上
    for idx, (model, x_limit) in enumerate(MODELS_CONFIG):
        ax = axes_flat[idx]

        # 计算行列位置以控制坐标轴标签
        row = idx // 2
        col = idx % 2

        show_ylabel = col == 0  # 仅左列显示 Y 轴标签
        show_xlabel = row == 1  # 仅底行显示 X 轴标签

        # 调用绘图函数 (注意：这里关闭 show_legend，统一在最后加)
        draw_one(
            ax,
            model,
            x_limit,
            show_title=False,
            show_xlabel=show_xlabel,
            show_ylabel=show_ylabel,
            show_legend=False,
        )

    # 手动构建顶部图例 (仅实线)
    proxies = []
    labels = []
    seen = set()

    for _, legend, linestyle, color in attacks_list:
        if linestyle != "-":
            continue
        if legend in seen:
            continue
        seen.add(legend)
        # 创建图例代理对象
        proxies.append(
            Line2D([0], [0], color=color, linestyle=linestyle, linewidth=1.2)
        )
        labels.append(legend)

    # 在顶部添加全局图例
    if proxies:
        fig.legend(
            proxies,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.995),
            ncol=3,
            frameon=True,
            columnspacing=1.0,
            handlelength=2.2,
            fontsize=fontsize - 2,
        )

    # 调整布局：给顶部图例留出空间，并调整子图间距
    plt.subplots_adjust(
        left=0.08, right=0.98, bottom=0.08, top=0.84, wspace=0.15, hspace=0.15
    )

    combined_base = os.path.join(output_dir, "combined_mnist_l2_2x2")

    # 【优化点】 pad_inches=0 去除白边
    fig.savefig(combined_base + ".pdf", bbox_inches="tight", pad_inches=0)
    fig.savefig(combined_base + ".png", bbox_inches="tight", pad_inches=0, dpi=PNG_DPI)

    plt.close(fig)

    print("Done. Results saved in:", output_dir)


if __name__ == "__main__":
    main()
