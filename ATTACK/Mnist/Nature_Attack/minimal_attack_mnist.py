#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NT_attack_mnist.py - MNIST 攻击评估 (严格复现版)
- 适配: RTX 3090/4090 等新架构 GPU
- 特性: 强制确定性计算 (Deterministic)，结果完全可复现
- 注意: 速度会略微下降 (禁用 TF32/Benchmark)，但换来绝对一致的结果
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
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

# ========== 0. 严格确定性环境设置 (必须放在最前面) ==========
# 设置 CUBLAS 环境变量，确保 CUDA 算子确定性 (针对 Torch 1.7+ 和 Ampere 架构)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTHONHASHSEED"] = str(42)

# ========== 项目路径准备 ==========
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

warnings.filterwarnings(
    "ignore", category=UserWarning, message="pkg_resources is deprecated"
)
warnings.filterwarnings("ignore", category=FutureWarning)

# ========== 本地 download/get helpers ==========
from util.download_mnist import download_mnist, get_dataset

# ========== adv-lib / utils ==========
from adv_lib.utils import requires_grad_
from adv_lib.utils.attack_utils import (
    run_attack,
    compute_attack_metrics,
    _default_metrics,
)

# 距离/感知指标
from adv_lib.distances.structural_similarity import compute_ssim, compute_ms_ssim
from adv_lib.distances.color_difference import ciede2000_loss
from adv_lib.utils.lagrangian_penalties import all_penalties

from util.utils import validate_and_combine_attack_data, evaluate_accuracy

# ========== 模型 ==========
from models.mnist.mnist import SmallCNN, IBP_large

# ========== 攻击集合 ==========
from adv_lib.attacks import alma, ddn, sdf, df, fmn
from attacks.GeoSensFool import GSF as gsf
from attacks.original_fab import original_fab as fab
from adv_lib.attacks.auto_pgd import minimal_apgd
from attacks.SGSF import SGSF as sgsf
from attacks.SuperDDN import sddn
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

    # 禁止 TF32 (虽然慢一点，但精度更高且稳定)
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
    # ========== 参数解析 ==========
    parser = argparse.ArgumentParser(
        description="Run multiple attacks on MNIST (Strict Determinism)"
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="device, e.g. cuda:0"
    )
    parser.add_argument(
        "--batch-size", type=int, default=1024, help="evaluation batch size"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models",
        help="directory for models",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="Results/Mnist/Nature_Attack",
        help="output directory",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument(
        "--download", action="store_true", help="download MNIST before running"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="减少终端输出（不影响 tqdm 进度条）",
    )
    parser.add_argument(
        "--resume",
        type=str2bool,
        default=True,
        help="是否断点续跑",
    )

    args = parser.parse_args()

    # ========== 1. 初始化确定性环境 ==========
    setup_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 输出/模型目录
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (Path(args.model_dir) / "mnist").mkdir(parents=True, exist_ok=True)

    # ========== 日志 ==========
    log_path = out_dir / "NT_attack_mnist.log"
    logger = logging.getLogger("NT_attack_mnist")
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

    # ========== 数据准备 ==========
    if args.download:
        try:
            download_mnist(download=True)
            log_info("[INFO] MNIST download attempted")
        except Exception as e:
            log_info(f"[WARN] download_mnist failed: {e}")

    transform = transforms.ToTensor()
    test_ds = get_dataset(train=False, transform=transform)
    total_samples = len(test_ds)
    log_info(f"[DATA] MNIST test samples: {total_samples}")

    # 创建 DataLoader 的生成器 (关键步骤)
    g = torch.Generator()
    g.manual_seed(args.seed)

    # 手动构建 DataLoader 以确保确定性
    loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,  # 必须为 False
        num_workers=2,
        pin_memory=True,
        worker_init_fn=seed_worker,  # 固定 worker 种子
        generator=g,  # 固定生成器
    )

    # ========== 模型加载 ==========
    models = {
        "SmallCNN_regular": SmallCNN(),
        "SmallCNN_ddn_l2": SmallCNN(),
        "SmallCNN_trades_linf": SmallCNN(),
        "IBP_large_linf": IBP_large(in_ch=1, in_dim=28),
    }

    model_paths = {
        "SmallCNN_regular": os.path.join(args.model_dir, "mnist", "mnist_regular.pth"),
        "SmallCNN_ddn_l2": os.path.join(
            args.model_dir, "mnist", "mnist_robust_ddn.pth"
        ),
        "SmallCNN_trades_linf": os.path.join(
            args.model_dir, "mnist", "mnist_robust_trades.pt"
        ),
        "IBP_large_linf": os.path.join(args.model_dir, "IBP", "IBP_large_best.pth"),
    }

    for name, model in models.items():
        if name in model_paths and os.path.exists(model_paths[name]):
            try:
                state_dict = torch.load(
                    model_paths[name], map_location=device, weights_only=False
                )
                if isinstance(state_dict, dict) and "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
                model.load_state_dict(state_dict)
                log_info(f"[INFO] Loaded weights for {name}")
            except Exception as e:
                log_info(f"[ERROR] Failed to load {name}: {e}")
        else:
            log_info(f"[WARN] No weights found for {name}, using random init")

        model.eval()
        model.to(device)
        requires_grad_(model, False)

    # ========== 评估干净准确率 ==========
    log_info("\n[INFO] Evaluating Clean Accuracy (Deterministic)...")
    for model_name, model in models.items():
        acc_results = evaluate_accuracy(model, loader, device, topk=(1,))
        top1 = acc_results.get(1, 0.0)
        log_info(f"  > Model: {model_name:20s} | Top-1: {top1:5.2f}%")
    log_info("[INFO] Clean accuracy evaluation completed.\n")

    # ========== 攻击配置 ==========
    penalty = all_penalties["P2"]

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
        (
            "APGD_l2",
            partial(
                minimal_apgd,
                norm=2,
                targeted_version=True,
                max_eps=5,
                binary_search_steps=13,
            ),
        ),
        ("C&W_l2_9x1000", partial(cw, binary_search_steps=9, max_iterations=1000)),
        ("C&W_l2_9x10000", partial(cw, binary_search_steps=9, max_iterations=10000)),
    ]

    # ========== metrics 扩展 ==========
    metrics = _default_metrics.copy()
    metrics["ssim"] = compute_ssim
    metrics["ciede2000"] = ciede2000_loss

    # ========== 自定义 metrics 打印 ==========
    def print_metrics_verbose(m: dict) -> None:
        np.set_printoptions(
            formatter={"float": "{:0.3f}".format},
            threshold=16,
            edgeitems=3,
            linewidth=120,
        )

        def _out(line: str):
            logger.info(line)
            if not args.quiet:
                print(line)

        _out("Attack success: {:.2%}".format(m["success"].float().mean().item()))
        for distance, values in m["distances"].items():
            data = values.cpu().numpy()
            success = m["success"].cpu().numpy()
            msg = "{}: Mean={:.3f} | Median={:.3f}".format(
                distance, data.mean(), np.median(data)
            )
            if success.any():
                msg += " | Succ Avg={:.3f}".format(data[success].mean())
            _out(msg)

    # ========== 主循环 ==========
    for atk_name, atk_fn in attacks:
        # 【关键】每个攻击配置开始前，重新强制设置随机种子
        setup_seed(args.seed)

        for model_name, model in models.items():
            save_name = f"NT_{model_name}_{atk_name}.pt"
            save_path = out_dir / save_name

            if args.resume and save_path.exists():
                log_info(
                    f"[SKIP] Attack={atk_name} | Model={model_name} 已存在结果，跳过"
                )
                continue

            log_info(f"\n[RUN] Attack={atk_name} | Model={model_name}")

            attack_data_list = []
            processed_samples = 0

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

                if (batch_idx + 1) % 10 == 0:
                    logger.info(
                        f"{model_name}-{atk_name}: processed {processed_samples}/{total_samples}"
                    )

            is_valid, combined_data = validate_and_combine_attack_data(
                attack_data_list, verbose=False
            )
            if not is_valid:
                msg = f"[ERROR] combine failed for {model_name}/{atk_name}"
                log_info(msg)
                logger.error(msg)
                continue

            attack_metrics = compute_attack_metrics(
                model=model,
                attack_data=combined_data,
                batch_size=args.batch_size,
                metrics=metrics,
            )

            print_metrics_verbose(attack_metrics)

            torch.save(attack_metrics, str(save_path))
            log_info(f"[SAVED] {save_path}")

            # 清理显存
            del attack_data, attack_data_list, combined_data, attack_metrics
            torch.cuda.empty_cache()

    # ========== SUMMARY ==========
    log_info("\n===== SUMMARY (MNIST Deterministic) =====")

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

            dists_all = metrics_blob.get("distances", {})
            # 优先显示 L2
            dist_key = (
                "l2"
                if "l2" in dists_all
                else (list(dists_all.keys())[0] if dists_all else None)
            )

            if dist_key and success.any():
                dists = dists_all[dist_key]
                dists_succ = dists[success]
                mean_dist = dists_succ.mean().item()
                median_dist = dists_succ.median().item()
                line = (
                    f"{model_name:20s} {atk_name:18s} "
                    f"ASR={success_rate:6.2f}% | "
                    f"Mean {dist_key}={mean_dist:.4f} | Med={median_dist:.4f}"
                )
            else:
                line = f"{model_name:20s} {atk_name:18s} " f"ASR={success_rate:6.2f}%"
            log_info(line)

    log_info("\n[ALL DONE]")


if __name__ == "__main__":
    main()
    os.system("shutdown")
