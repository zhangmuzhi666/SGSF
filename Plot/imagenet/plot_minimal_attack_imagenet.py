#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import warnings

# 1. 屏蔽 PyTorch 加载时的 Future Warning，保持终端清爽
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
# 配置路径 (ImageNet)
# =========================
result_dir = os.path.join("Results", "ImageNet", "Nature_Attack")
output_dir = os.path.join("Results", "ImageNet", "Comparison", "L2_RSA")
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

plt.rcParams["lines.linewidth"] = 1

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

# =========================
# ImageNet 模型配置
# =========================
MODELS_CONFIG = [
    ("ResNet50", 2.0),
    ("ResNet50_l2_3", 15.0),
    ("ResNet50_linf_4", 15.0),
]

# =========================
# 攻击列表配置
# =========================
attacks_list = [
    # --- 实线组（进图例）---
    ("DF_l2_100_0.02", r"DF $\ell_2$ 100", "-", colors[0]),
    ("GSF_l2_100_0.02", r"GSF $\ell_2$ 100", "-", colors[1]),
    ("SDF_l2_100_0.02", r"SDF $\ell_2$ 100", "-", colors[2]),
    ("SGSF_l2_100", r"SGSF $\ell_2$ 100", "-", colors[3]),
    ("APGD_l2", r"$\mathrm{APGD}^\mathrm{DLR}_\mathrm{T}$ $\ell_2$", "-", colors[8]),
    ("FMN_l2_1000", r"FMN $\ell_2$ 1000", "-", colors[5]),
    ("DDN_l2_1000", r"DDN $\ell_2$ 1000", "-", colors[6]),
    ("ALMA_l2_1000", r"ALMA $\ell_2$ 1000", "-", colors[9]),
    ("FAB_l2_1000", r"FAB $\ell_2$ 100", "-", colors[4]),
    ("C&W_l2_9x1000", r"C\&W $\ell_2$ $9\times 1000$", "-", colors[7]),
    # --- 点线组（仅背景对比，不进图例）---
    ("FAB_l2_100", r"FAB $\ell_2$ 100", ":", colors[4]),
    ("FMN_l2_100", r"FMN $\ell_2$ 100", ":", colors[5]),
    ("DDN_l2_100", r"DDN $\ell_2$ 100", ":", colors[6]),
    ("ALMA_l2_100", r"ALMA $\ell_2$ 100", ":", colors[9]),
]


# =========================
# 工具函数：加载数据
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
# 绘图逻辑封装
# =========================
def main():
    # Step 1: 预加载缓存
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
        print(f"[INFO] Missing files: {len(missing)} (showing first 5)")
        for m, a in missing[:5]:
            print("   ", m, a)

    # 内部函数：绘制单个坐标轴
    def draw_one_ax(
        ax, model, x_limit, *, show_xlabel=True, show_ylabel=True, show_legend=True
    ):
        # 绘制曲线
        for attack_id, _, _, _ in attacks_list:
            if attack_id not in curve_cache[model]:
                continue
            x, y, legend, linestyle, color = curve_cache[model][attack_id]

            # 实线进图例，虚线/点线不进
            plot_label = legend if linestyle == "-" else "_nolegend_"
            ax.plot(x, y, label=plot_label, linestyle=linestyle, c=color)

        # 坐标轴设置
        ax.set_yticks(YTICKS)
        ax.set_xlim(0, x_limit)
        ax.set_ylim(0, 1)
        ax.grid(True, linestyle="--", c="lightgray", which="both")

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
            ax.legend(loc="upper right", framealpha=0.85, fontsize=fontsize - 3)

    # Task 1: 生成单张图 (Single Plots)
    print("Generating single model curves (pdf+png)...")
    for model, x_limit in MODELS_CONFIG:
        fig, ax = plt.subplots(figsize=(3.4, 2.6))
        draw_one_ax(
            ax, model, x_limit, show_xlabel=True, show_ylabel=True, show_legend=True
        )

        plt.tight_layout()
        base = os.path.join(output_dir, f"single_imagenet_l2_{model}")

        # 【优化点】 pad_inches=0 去除白边
        fig.savefig(base + ".pdf", bbox_inches="tight", pad_inches=0)
        fig.savefig(base + ".png", bbox_inches="tight", pad_inches=0, dpi=PNG_DPI)

        plt.close(fig)

    # Task 2: 生成合并图 (Combined 2x2)
    print("Generating combined 2x2 figure...")

    fig, axes = plt.subplots(2, 2, figsize=(6.0, 4.5))
    axes_flat = axes.flatten()

    # 遍历 3 个模型
    for idx, (model, x_limit) in enumerate(MODELS_CONFIG):
        ax = axes_flat[idx]
        row = idx // 2
        col = idx % 2

        # 布局控制
        show_ylabel = col == 0
        show_xlabel = row == 1

        draw_one_ax(
            ax,
            model,
            x_limit,
            show_xlabel=show_xlabel,
            show_ylabel=show_ylabel,
            show_legend=False,
        )

    # 隐藏第 4 个格子 (右下角)
    if len(MODELS_CONFIG) < 4:
        for i in range(len(MODELS_CONFIG), 4):
            axes_flat[i].axis("off")

    # 顶部全局图例
    proxies, labels = [], []
    seen = set()
    for _, legend, linestyle, color in attacks_list:
        if linestyle != "-":
            continue
        if legend in seen:
            continue
        seen.add(legend)
        proxies.append(Line2D([0], [0], color=color, linestyle=linestyle, linewidth=1))
        labels.append(legend)

    ncol = 3 if len(labels) <= 9 else 4

    if proxies:
        fig.legend(
            proxies,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.995),
            ncol=ncol,
            frameon=True,
            columnspacing=1.0,
            handlelength=2.2,
            fontsize=fontsize - 2,
        )

    plt.subplots_adjust(
        left=0.03, right=0.97, bottom=0.03, top=0.84, wspace=0.15, hspace=0.15
    )

    combined_base = os.path.join(output_dir, "combined_imagenet_l2_2x2")

    # 【优化点】 pad_inches=0 去除白边
    fig.savefig(combined_base + ".pdf", bbox_inches="tight", pad_inches=0)
    fig.savefig(combined_base + ".png", bbox_inches="tight", pad_inches=0, dpi=PNG_DPI)

    plt.close(fig)

    print("Done. Results saved in:", output_dir)


if __name__ == "__main__":
    main()
