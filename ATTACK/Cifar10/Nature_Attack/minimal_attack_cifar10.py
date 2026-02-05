#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NT_attack_cifar10.py - 使用 CIFAR-10 1k 均衡子集进行攻击评估 (严格复现版)
- 适配: RTX 3090/4090 等新架构 GPU
- 特性: 强制确定性计算 (Deterministic)，结果完全可复现
- 数据: 每类 100 张，共 1000 张测试图片
"""

import os
import sys
import random
from functools import partial
from pathlib import Path
import argparse
import warnings
import logging
import traceback
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm

# ========== 0. 严格确定性环境设置 (必须放在最前面) ==========
# 设置 CUBLAS 环境变量，确保 CUDA 算子确定性 (针对 Torch 1.7+ 和 Ampere 架构)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTHONHASHSEED"] = str(42)

# ========== 项目路径 ==========
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ========== 警告抑制 ==========
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

# ========== 导入工具 ==========
from util.download_cifar10 import download_cifar10, get_dataset
from util.build_cifar10_1k import build_cifar10_1k_subset
from util.utils import validate_and_combine_attack_data, evaluate_accuracy

from adv_lib.distances.lpips import LPIPS
from adv_lib.attacks.auto_pgd import minimal_apgd
from adv_lib.utils import requires_grad_
from adv_lib.utils.attack_utils import (
    run_attack,
    compute_attack_metrics,
    _default_metrics,
    print_metrics,
)
from adv_lib.utils.lagrangian_penalties import all_penalties
from adv_lib.distances.color_difference import ciede2000_loss
from adv_lib.distances.structural_similarity import (
    compute_ssim,
    compute_ms_ssim,
)
from robustbench.utils import load_model

# ========== 攻击方法导入 ==========
from adv_lib.attacks import alma, ddn, sdf, df, fmn
from adv_lib.attacks.perceptual_color_attacks import perc_al

from attacks.GeoSensFool import GSF as gsf
from attacks.SGSF import SGSF as sgsf
from attacks.original_fab import original_fab as fab
from adv_lib.attacks import carlini_wagner_l2 as cw


# ========== 核心工具函数 (复现性关键) ==========


def setup_seed(seed=42):
    """
    设置所有可能的随机种子以保证结果可复现
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 强制 CuDNN 使用确定性算法
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # 禁止 TF32 (虽然慢一点，但精度更高且稳定，避免 A100/4090 上的精度波动)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def seed_worker(worker_id):
    """DataLoader 的 worker 初始化函数，确保多进程加载数据时随机性固定"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def str2bool(v):
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ("yes", "true", "t", "y", "1"):
        return True
    if v in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def main():
    # ========== 参数 ==========
    parser = argparse.ArgumentParser(
        description="Run multiple attacks on CIFAR-10 (Strict Determinism)"
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="device id")
    parser.add_argument(
        "--batch-size", type=int, default=512, help="evaluation batch size"
    )
    parser.add_argument(
        "--model-dir", type=str, default="models", help="directory for models"
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="Results/Cifar10/Nature_Attack",
        help="output directory",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="减少终端输出",
    )
    parser.add_argument(
        "--resume",
        type=str2bool,
        default=True,
        help="是否断点续跑",
    )
    parser.add_argument(
        "--rebuild-subset",
        action="store_true",
        help="强制重新构建 CIFAR-10 1k 子集",
    )

    args = parser.parse_args()

    # ========== 1. 初始化确定性环境 ==========
    setup_seed(args.seed)

    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (Path(args.model_dir) / "cifar10").mkdir(parents=True, exist_ok=True)

    # ========== 日志 ==========
    log_path = out_dir / "NT_attack_cifar10_1.log"
    logger = logging.getLogger("NT_attack_cifar10")
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    def log_info(msg: str, print_also: bool = True):
        logger.info(msg)
        if print_also and not args.quiet:
            print(msg)

    log_info(
        f"[INIT] Device: {device} | Batch Size: {args.batch_size} | Seed: {args.seed}"
    )
    log_info("[INIT] Mode: Strict Determinism (TF32 Disabled, Benchmark Disabled)")

    # ========== 数据 ==========
    log_info("[DATA] Loading CIFAR-10 dataset...")
    download_cifar10(download=True)
    transform = transforms.Compose([transforms.ToTensor()])
    test_ds = get_dataset(train=False, transform=transform)

    # 构建或加载 1k 均衡子集索引
    subset_pth = build_cifar10_1k_subset(
        dataset=test_ds, per_class_k=100, overwrite=args.rebuild_subset
    )

    # 读取索引
    subset_info = torch.load(subset_pth, weights_only=False)
    indices = subset_info["indices"]
    total_samples = len(indices)
    log_info(f"[DATA] Using 1k subset indices from: {subset_pth}")

    # 手动构建 Subset 和 DataLoader 以确保确定性 (替换原有的 subset_loader_from_pth)
    subset_ds = Subset(test_ds, indices)

    # 创建 DataLoader 的生成器
    g = torch.Generator()
    g.manual_seed(args.seed)

    loader = DataLoader(
        subset_ds,
        batch_size=args.batch_size,
        shuffle=False,  # 必须为 False
        num_workers=2,
        pin_memory=True,
        worker_init_fn=seed_worker,  # 关键：固定 worker 种子
        generator=g,  # 关键：固定生成器
    )

    # ========== 模型 ==========
    log_info("[MODEL] Loading models from RobustBench...")
    models = {
        "WideResNet_28-10": load_model(
            model_name="Standard",
            model_dir=args.model_dir,
            dataset="cifar10",
            threat_model="Linf",
        ),
        "Carmon2019": load_model(
            model_name="Carmon2019Unlabeled",
            model_dir=args.model_dir,
            dataset="cifar10",
            threat_model="Linf",
        ),
        "Augustin2020": load_model(
            model_name="Augustin2020Adversarial",
            model_dir=args.model_dir,
            dataset="cifar10",
            threat_model="L2",
        ),
    }

    for k, m in models.items():
        m.eval()
        m.to(device)
        requires_grad_(m, False)

    # ========== 评估干净准确率 ==========
    log_info("\n[INFO] Evaluating Clean Accuracy (Deterministic)...")
    for model_name, model in models.items():
        acc_results = evaluate_accuracy(model, loader, device, topk=(1, 5))
        top1 = acc_results.get(1, 0.0)
        top5 = acc_results.get(5, 0.0)
        log_info(
            f"  > Model: {model_name:20s} | Top-1: {top1:5.2f}% | Top-5: {top5:5.2f}%"
        )
    log_info("[INFO] Clean accuracy evaluation completed.\n")

    penalty = all_penalties["P2"]

    # ========== 攻击集合 ==========
    attacks = [
        ("DF_l2_100", partial(df, steps=100)),
        ("GSF_l2_100", partial(gsf, steps=100)),
        ("SDF_l2_100", partial(sdf, steps=100)),
        ("SGSF_l2_100", partial(sgsf, steps=100)),
        ("FAB_l2_100", partial(fab, norm="L2", n_iter=100)),
        ("FAB_l2_1000", partial(fab, norm="L2", n_iter=1000)),
        ("FMN_l2_100", partial(fmn, norm=2, steps=100)),
        ("FMN_l2_1000", partial(fmn, norm=2, steps=1000)),
        ("DDN_l2_100", partial(ddn, steps=100)),
        ("DDN_l2_1000", partial(ddn, steps=1000)),
        (
            "ALMA_l2_100",
            partial(
                alma,
                penalty=penalty,
                distance="l2",
                init_lr_distance=0.1,
                α=0.5,
                num_steps=100,
            ),
        ),
        (
            "ALMA_l2_1000",
            partial(
                alma,
                penalty=penalty,
                distance="l2",
                init_lr_distance=0.1,
                num_steps=1000,
            ),
        ),
        # ("C&W_l2_9x10000", partial(cw, binary_search_steps=9, max_iterations=10000)),
        (
            "APGD_l2",
            partial(
                minimal_apgd,
                norm=2,
                targeted_version=True,
                max_eps=2.5,
                binary_search_steps=12,
            ),
        ),
        (
            "ALMA_SSIM_100",
            partial(
                alma,
                penalty=penalty,
                distance="ssim",
                init_lr_distance=0.0001,
                α=0.5,
                num_steps=100,
            ),
        ),
        (
            "ALMA_SSIM_1000",
            partial(
                alma,
                penalty=penalty,
                distance="ssim",
                init_lr_distance=0.0001,
                num_steps=1000,
            ),
        ),
        ("Perc-AL_100", partial(perc_al, num_classes=10, max_iterations=100)),
        ("Perc-AL_1000", partial(perc_al, num_classes=10, max_iterations=1000)),
        (
            "ALMA_CIEDE2000_100",
            partial(
                alma,
                penalty=penalty,
                distance="ciede2000",
                init_lr_distance=0.05,
                α=0.5,
                num_steps=100,
            ),
        ),
        (
            "ALMA_CIEDE2000_1000",
            partial(
                alma,
                penalty=penalty,
                distance="ciede2000",
                init_lr_distance=0.05,
                num_steps=1000,
            ),
        ),
        (
            "ALMA_LPIPS_100",
            partial(
                alma,
                penalty=penalty,
                distance="lpips",
                init_lr_distance=0.01,
                α=0.5,
                num_steps=100,
            ),
        ),
        (
            "ALMA_LPIPS_1000",
            partial(
                alma,
                penalty=penalty,
                distance="lpips",
                init_lr_distance=0.01,
                num_steps=1000,
            ),
        ),
        # ("C&W_l2_9x1000", partial(cw, binary_search_steps=9, max_iterations=1000)),
        # ("C&W_l2_9x10000", partial(cw, binary_search_steps=9, max_iterations=10000)),
    ]

    # ========== metrics 扩展 ==========
    metrics = _default_metrics.copy()
    metrics["ssim"] = compute_ssim
    metrics["ciede2000"] = ciede2000_loss
    lpips_path = Path("models/alex/alex.pth")
    if lpips_path.exists():
        metrics["lpips"] = partial(LPIPS, linear_mapping=str(lpips_path))
    else:
        log_info(f"[WARN] LPIPS model not found: {lpips_path}, trying default load...")
        try:
            metrics["lpips"] = partial(LPIPS, linear_mapping="alex.pth")
        except:
            pass

    # ========== 主循环 ==========
    for atk_name, atk_fn in attacks:
        # 【关键】每个攻击配置开始前，重新强制设置随机种子
        # 确保 attack A 和 attack B 之间的随机状态互不干扰，且单跑 attack B 时结果与连跑一致
        setup_seed(args.seed)

        for model_name, model in models.items():
            save_name = f"NT_{model_name}_{atk_name}.pt"
            save_path = out_dir / save_name

            # 断点续跑
            if args.resume and save_path.exists():
                log_info(
                    f"[SKIP] Attack={atk_name} | Model={model_name} 结果已存在，跳过"
                )
                continue

            log_info(f"\n[RUN] Attack={atk_name} | Model={model_name}")

            attack_data_list = []
            processed_samples = 0

            # 进度条
            for batch_idx, (images, labels) in enumerate(
                tqdm(loader, desc=f"{model_name}-{atk_name}", leave=False)
            ):
                images = images.to(device)
                labels = labels.to(device)

                attack_data = run_attack(
                    model=model,
                    inputs=images,
                    labels=labels,
                    attack=atk_fn,
                    batch_size=args.batch_size,
                )
                attack_data_list.append(attack_data)
                processed_samples += len(images)

                if (batch_idx + 1) % 5 == 0:
                    logger.info(
                        f"{model_name}-{atk_name}: 已处理 {processed_samples}/{total_samples} 样本"
                    )

            # 验证与合并
            is_valid, combined_data = validate_and_combine_attack_data(
                attack_data_list, verbose=False
            )
            if not is_valid:
                msg = f"[ERROR] combine failed for {model_name}/{atk_name}"
                log_info(msg)
                logger.error(msg)
                continue

            # 计算指标
            attack_metrics = compute_attack_metrics(
                model=model,
                attack_data=combined_data,
                batch_size=args.batch_size,
                metrics=metrics,
            )

            # 打印 + 保存
            print_metrics(attack_metrics)
            torch.save(attack_metrics, str(save_path))
            log_info(f"[SAVED] {save_path}")

            # 清理显存
            del attack_data, attack_data_list, combined_data, attack_metrics
            torch.cuda.empty_cache()

    # ========== SUMMARY ==========
    log_info("\n===== SUMMARY (CIFAR-10 1k Subset Deterministic) =====")

    for atk_name, _ in attacks:
        for model_name in models.keys():
            file_path = out_dir / f"NT_{model_name}_{atk_name}.pt"
            if not file_path.exists():
                continue

            try:
                metrics_blob = torch.load(
                    str(file_path), map_location="cpu", weights_only=False
                )
            except Exception:
                continue

            success = metrics_blob.get("success", None)
            if success is None:
                continue

            success_rate = float(success.float().mean().item()) * 100.0
            log_info(
                f"\n[MODEL] {model_name:20s} [ATTACK] {atk_name:18s} "
                f"ASR={success_rate:6.2f}%"
            )

            dists_all = metrics_blob.get("distances", {})
            if not dists_all:
                continue

            prefer_order = [
                "l2",
                "linf",
                "l1",
                "l0",
                "ssim",
                "msssim",
                "ciede2000",
                "lpips",
            ]
            ordered_keys = [k for k in prefer_order if k in dists_all.keys()]
            for k in dists_all.keys():
                if k not in ordered_keys:
                    ordered_keys.append(k)

            for dist_name in ordered_keys:
                dists = dists_all[dist_name]
                if dists is None or not success.any():
                    continue

                dists_succ = dists[success]
                mean_dist = dists_succ.mean().item()
                median_dist = dists_succ.median().item()
                std_dist = dists_succ.std(unbiased=False).item()

                log_info(
                    f"  [{dist_name:9s}] "
                    f"Mean={mean_dist:.4f} | Median={median_dist:.4f} | Std={std_dist:.4f}"
                )

    log_info("\n[ALL DONE] - CIFAR-10 评估完成")


if __name__ == "__main__":

    main()
    # os.system("shutdown")
