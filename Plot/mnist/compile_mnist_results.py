#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import torch
import numpy as np
import warnings

# 忽略警告
warnings.filterwarnings("ignore", category=UserWarning)

# ======================= 0. 基础配置与路径 =======================
DATASET_NAME = "Mnist"

# 固定路径配置
result_dir = os.path.join("Results", "Mnist", "Nature_Attack")
output_dir = os.path.join("Results", "Mnist", "Comparison", "CSV")
os.makedirs(output_dir, exist_ok=True)

print(f"当前数据集: {DATASET_NAME}")
print(f"输入目录: {result_dir}")
print(f"输出目录: {output_dir}")
print("-" * 50)

# ======================= 1. 模型与攻击列表 (MNIST) =======================
MODELS = [
    "SmallCNN_regular",
    "SmallCNN_ddn_l2",
    "SmallCNN_trades_linf",
    "IBP_large_linf",
]

# 【自定义模型显示名称】
MODEL_DISPLAY_NAME = {
    "SmallCNN_regular": "SmallCNN",
    "SmallCNN_ddn_l2": "SmallCNN-DDN",
    "SmallCNN_trades_linf": "SmallCNN-TRADES",
    "IBP_large_linf": "CROWN-IBP",
}

MNIST_ATTACKS = [
    # --- Geometric / Base ---
    "DF_l2_100",
    "GSF_l2_100",
    "SDF_l2_100",
    "SGSF_l2_100",
    # --- Minimal Norm Baselines ---
    "FAB_l2_100",
    "FAB_l2_1000",
    "FMN_l2_100",
    "FMN_l2_1000",
    "DDN_l2_100",
    "DDN_l2_1000",
    "C&W_l2_9x1000",
    "C&W_l2_9x10000",
    # --- ALMA L2 ---
    "ALMA_l2_100",
    "ALMA_l2_1000",
    # --- SOTA ---
    "APGD_l2",
]

# 攻击名称映射
ATTACK_SHORT_NAME = {
    "DF_l2_100": r"DF$\ell_2$ 100",
    "GSF_l2_100": r"GSF$\ell_2$ 100",
    "SDF_l2_100": r"SDF $\ell_2$ 100",
    "SGSF_l2_100": r"SGSF $\ell_2$ 100",
    "FAB_l2_100": r"FAB$^\dagger$ $\ell_2$ 100",
    "FAB_l2_1000": r"FAB$^\dagger$ $\ell_2$ 1000",
    "FMN_l2_100": r"FMN $\ell_2$ 100",
    "FMN_l2_1000": r"FMN $\ell_2$ 1000",
    "DDN_l2_100": r"DDN 100",
    "DDN_l2_1000": r"DDN 1000",
    "C&W_l2_9x1000": r"C&W $\ell_2$ 9x1000",
    "C&W_l2_9x10000": r"C&W $\ell_2$ 9x10000",
    "ALMA_l2_100": r"ALMA $\ell_2$ 100",
    "ALMA_l2_1000": r"ALMA $\ell_2$ 1000",
    "APGD_l2": r"APGD$_{\text{DLR}}^{\text{T}}$ $\ell_2^\ddagger$",
}


# ======================= 2. 辅助函数 =======================
def get_target_norm(attack_name):
    name_lower = attack_name.lower()
    if "l2" in name_lower:
        return "l2", "L2"
    elif "linf" in name_lower or "inf" in name_lower:
        return "linf", "Linf"
    else:
        return "l2", "L2"


# ======================= 3. 核心逻辑：读取 .pt 文件 =======================
all_rows = []
print(f"正在读取数据 ...")

for model in MODELS:
    display_model = MODEL_DISPLAY_NAME.get(model, model)

    for attack in MNIST_ATTACKS:
        file_path = os.path.join(result_dir, f"NT_{model}_{attack}.pt")

        if not os.path.exists(file_path):
            continue

        try:
            metrics = torch.load(file_path, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue

        if "success" not in metrics:
            continue

        success = metrics["success"].bool()
        if success.numel() == 0:
            continue

        # --- 数据提取 (逻辑优化版) ---
        asr = success.float().mean().item() * 100
        short_name = ATTACK_SHORT_NAME.get(attack, attack)

        # 兼容各种 Time Key
        t_val = metrics.get("time", metrics.get("runtimes", metrics.get("total_time")))
        time_taken = 0.0
        if t_val is not None:
            if isinstance(t_val, torch.Tensor):
                time_taken = t_val.float().mean().item()
            elif isinstance(t_val, (list, np.ndarray)):
                time_taken = np.mean(t_val)
            else:
                time_taken = float(t_val)

        # 距离提取 (带维度检查)
        target_key, target_display = get_target_norm(attack)
        distances_dict = metrics.get("distances", {})
        dist_tensor = distances_dict.get(target_key, None)

        mean_val = float("nan")
        median_val = float("nan")

        if dist_tensor is not None:
            if not isinstance(dist_tensor, torch.Tensor):
                dist_tensor = torch.tensor(dist_tensor)

            # 维度匹配检查
            if dist_tensor.shape == success.shape:
                if success.any():
                    succ_dists = dist_tensor[success]
                    mean_val = succ_dists.mean().item()
                    median_val = succ_dists.median().item()
            elif dist_tensor.numel() > 0:
                # 备用方案：如果存的是已经筛选过的距离
                mean_val = dist_tensor.float().mean().item()
                median_val = dist_tensor.float().median().item()

        num_forwards = int(metrics.get("num_forwards", 0))
        num_backwards = int(metrics.get("num_backwards", 0))

        row = {
            "MODEL": model,
            "Model_Display": display_model,
            "ATTACK_ID": attack,
            "Attack": short_name,
            "ASR (%)": float(f"{asr:.2f}"),
            "Distance Type": target_display,
            "Median Distance": median_val,
            "Mean Distance": mean_val,
            "Forwards": num_forwards,
            "Backwards": num_backwards,
            "Time (s)": float(f"{time_taken:.2f}"),
        }
        all_rows.append(row)

if not all_rows:
    print("未找到有效数据，请检查路径和文件名。")
    exit()

df_all = pd.DataFrame(all_rows)
print(f"读取完成，共 {len(df_all)} 行数据。")

# ======================= 4. 生成 CSV 1: 原始详细表 =======================
detailed_csv_name = f"{DATASET_NAME}_Detailed_Clean.csv"
detailed_csv_path = os.path.join(output_dir, detailed_csv_name)

cols_order = [
    "MODEL",
    "Attack",
    "Distance Type",
    "ASR (%)",
    "Median Distance",
    "Mean Distance",
    "Forwards",
    "Backwards",
    "Time (s)",
]
df_detailed = df_all[[c for c in cols_order if c in df_all.columns]]
df_detailed.to_csv(detailed_csv_path, index=False)
print(f"\n[1] 已生成详细数据表: {detailed_csv_path}")

# ======================= 5. 生成 CSV 2: 论文汇总表 (Paper Style) =======================
print("正在生成论文格式汇总表 (Avg)...")
df_avg = (
    df_all.groupby("ATTACK_ID", sort=False)
    .agg(
        {
            "Attack": "first",
            "Distance Type": "first",
            "ASR (%)": "mean",
            "Median Distance": "mean",
            "Forwards": "mean",
            "Backwards": "mean",
            "Time (s)": "mean",
        }
    )
    .reset_index()
)

df_avg["ASR (%)"] = df_avg["ASR (%)"].map(lambda x: f"{x:.2f}")
df_avg["Median Distance"] = df_avg["Median Distance"].map(
    lambda x: f"{x:.4f}" if pd.notnull(x) else "-"
)
df_avg["Avg Time (s)"] = df_avg["Time (s)"].map(lambda x: f"{x:.2f}")

# 【核心优化】加入 \t 防止 Excel 日期格式化
df_avg["Forwards / Backwards"] = df_avg.apply(
    lambda row: f"\t{int(row['Forwards'])} / {int(row['Backwards'])}", axis=1
)

# 排序
attack_order_map = {k: i for i, k in enumerate(MNIST_ATTACKS)}
df_avg["_sort_key"] = df_avg["ATTACK_ID"].map(attack_order_map)
df_avg = df_avg.sort_values("_sort_key")

final_paper_cols = [
    "Distance Type",
    "Attack",
    "ASR (%)",
    "Median Distance",
    "Forwards / Backwards",
    "Avg Time (s)",
]
paper_csv_name = f"{DATASET_NAME}_Paper_Style.csv"
paper_csv_path = os.path.join(output_dir, paper_csv_name)
df_avg[final_paper_cols].to_csv(paper_csv_path, index=False)
print(f"[2] 已生成论文汇总表: {paper_csv_path}")

# ======================= 6. 生成 CSV 3: 图片同款格式 (Image Style) =======================
print("正在生成图片同款格式表...")
image_style_rows = []

for model in MODELS:
    df_sub = df_all[df_all["MODEL"] == model].copy()
    if df_sub.empty:
        continue

    display_model_name = df_sub["Model_Display"].iloc[0]

    for _, row in df_sub.iterrows():
        med_val = row["Median Distance"]
        med_str = "-" if pd.isna(med_val) else f"{med_val:.4f}"

        # 【核心优化】加入 \t 防止 Excel 日期格式化
        fwd_bwd_str = f"\t{int(row['Forwards'])} / {int(row['Backwards'])}"

        new_row = {
            "Model": display_model_name,
            "Attack": row["Attack"],
            "ASR (%)": f"{row['ASR (%)']:.2f}",
            "Median Distance": med_str,
            "Forwards / Backwards": fwd_bwd_str,
            "Time (s)": f"{row['Time (s)']:.2f}",
        }
        image_style_rows.append(new_row)

image_style_csv_name = f"{DATASET_NAME}_Image_Style.csv"
image_style_path = os.path.join(output_dir, image_style_csv_name)
pd.DataFrame(image_style_rows).to_csv(image_style_path, index=False)

print(f"[3] 已生成图片同款格式表: {image_style_path}")
print("-" * 50)
print("所有任务完成。")
