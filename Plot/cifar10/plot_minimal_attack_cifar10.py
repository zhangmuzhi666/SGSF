#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compile_cifar10_ablation_2.py (PDF + PNG 版 - 无白边)
- 功能：绘制 L2 vs Overshoot 曲线
- 输出：同时生成 .png (400dpi) 和 .pdf (矢量图)
- 修复：自动清洗文件名，防止重复运行
- 样式：pad_inches=0 去除白边
"""

import os
import glob
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# 屏蔽 FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

# ======================= 1. 路径配置 =======================
# 请确认此输入路径正确
NT_DIR = os.path.join("Results", "Cifar10", "Nature_Attack")

# 输出路径
OUT_DIR = os.path.join("Results", "Cifar10", "Ablation", "Plots_L2_vs_Overshoot")
os.makedirs(OUT_DIR, exist_ok=True)

# ======================= 2. 绘图风格设置 =======================
sns.set_theme(style="whitegrid", font_scale=1.2)
palette = sns.color_palette("Set1", n_colors=4)

STYLE_MAP = {
    "DF": {"color": palette[1], "marker": "o", "ls": "-"},  # 蓝
    "SDF": {"color": palette[0], "marker": "s", "ls": "--"},  # 红
    "GSF": {"color": palette[2], "marker": "^", "ls": "-."},  # 绿
    "SGSF": {"color": palette[3], "marker": "D", "ls": ":"},  # 紫
}

COLORS = {k: v["color"] for k, v in STYLE_MAP.items()}
MARKERS = {k: v["marker"] for k, v in STYLE_MAP.items()}
LINESTYLES = {k: v["ls"] for k, v in STYLE_MAP.items()}
ALGO_ORDER = ["DF", "SDF", "GSF", "SGSF"]

# ======================= 3. 核心函数 =======================


def parse_filename(filename):
    """解析并清洗文件名"""
    name = filename.replace(".pt", "")
    if name.startswith("NT_"):
        name = name[3:]

    try:
        parts = name.split("_")

        # 提取 Overshoot
        os_part = parts[-1]
        if not os_part.startswith("os"):
            return None
        overshoot = float(os_part.replace("os", ""))

        # 提取 Algorithm
        algo_idx = -4
        algorithm = parts[algo_idx]

        # 提取 Model Name 并清洗
        raw_model_parts = parts[:algo_idx]
        # 过滤掉 "cifar10" 防止重复
        clean_model_parts = [p for p in raw_model_parts if p.lower() != "cifar10"]
        model_name = "_".join(clean_model_parts)

        return {"Model": model_name, "Algorithm": algorithm, "Overshoot": overshoot}
    except Exception:
        return None


def load_data(input_dir):
    data_list = []
    pt_files = glob.glob(os.path.join(input_dir, "*.pt"))

    if not pt_files:
        print(f"❌ 错误：在 {input_dir} 下没找到 .pt 文件！")
        return pd.DataFrame()

    print(f"📂 正在读取 {len(pt_files)} 个数据文件...")

    for pt_path in pt_files:
        filename = os.path.basename(pt_path)
        meta = parse_filename(filename)
        if meta is None:
            continue

        try:
            metrics = torch.load(pt_path, map_location="cpu", weights_only=False)
            success = metrics.get("success", None)
            l2_dist = metrics.get("distances", {}).get("l2", None)

            if success is None or l2_dist is None:
                continue

            successful_l2 = l2_dist[success]
            l2_median = (
                float("nan")
                if len(successful_l2) == 0
                else successful_l2.median().item()
            )

            entry = meta.copy()
            entry["L2 Median"] = l2_median
            data_list.append(entry)
        except Exception:
            continue

    return pd.DataFrame(data_list)


def plot_single_model(df, model_name):
    subset = df[df["Model"] == model_name].copy()
    if subset.empty:
        return

    subset.sort_values(by="Overshoot", inplace=True)
    plt.figure(figsize=(8, 6))

    available = subset["Algorithm"].unique()
    for algo in ALGO_ORDER:
        if algo in available and algo in COLORS:
            data = subset[subset["Algorithm"] == algo]
            if data.empty:
                continue
            plt.plot(
                data["Overshoot"],
                data["L2 Median"],
                label=algo,
                color=COLORS[algo],
                marker=MARKERS[algo],
                linestyle=LINESTYLES[algo],
                linewidth=2.5,
                markersize=8,
                alpha=0.9,
            )

    plt.xlabel(r"overshoot ($\eta$)", fontsize=16)
    plt.ylabel(r"$\ell_2$-norm", fontsize=16)
    plt.legend(frameon=False, fontsize=13)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    # ====== 保存逻辑 (PNG + PDF) ======

    # 基础文件名 (无后缀)
    base_name = f"Plot_Clean_Cifar10_{model_name}"

    # 1. 保存 PNG
    png_path = os.path.join(OUT_DIR, base_name + ".png")
    # 【关键修改】 pad_inches=0 去除白边
    plt.savefig(png_path, dpi=400, bbox_inches="tight", pad_inches=0)

    # 2. 保存 PDF
    pdf_path = os.path.join(OUT_DIR, base_name + ".pdf")
    # 【关键修改】 pad_inches=0 去除白边
    plt.savefig(pdf_path, bbox_inches="tight", pad_inches=0)

    plt.close()
    print(f" -> ✅ 已保存: {base_name} (.png & .pdf)")


# ======================= 4. 主程序入口 =======================

if __name__ == "__main__":
    # 1. 读取数据
    df = load_data(NT_DIR)

    if not df.empty:
        models = df["Model"].unique()
        print(f"\n📊 检测到 {len(models)} 个模型: {models}\n")

        # 3. 循环绘图
        for model in models:
            plot_single_model(df, model)

        print(f"\n✨ 全部完成！图片 (PNG+PDF) 保存在: {OUT_DIR}")
    else:
        print("没有有效数据，程序退出。")
