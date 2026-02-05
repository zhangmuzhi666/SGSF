"""
NT_attack_imagenet_final.py
- 场景：ImageNet 1k 样本攻击
- 特性：
  1. 严格确定性 (Deterministic)：锁定随机种子，结果可复现。
  2. Batch 级断点续传 (Resumable)：防止 Cloud Studio 断连导致前功尽弃。
  3. 自动清理：跑完后自动删除临时文件。
  4. 纯净终端：Batch 详细信息只记入 Log，终端只显示进度条。
  5. 自动汇总：运行结束后自动输出 Summary 报表。
  6. [修复]：兼容 run_attack 不返回 success 键的情况。
"""

import os
import sys
import random
import argparse
import warnings
import logging
import time
import numpy as np
import torch
from functools import partial
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# ========== 0. 严格确定性环境设置 (必须最先执行) ==========
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTHONHASHSEED"] = str(42)

# ========== 项目路径 ==========
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

warnings.filterwarnings("ignore", category=UserWarning)

# ========== 导入工具 ==========
from adv_lib.utils import requires_grad_
from adv_lib.utils.attack_utils import (
    run_attack,
    compute_attack_metrics,
    _default_metrics,
    print_metrics,
)
from adv_lib.utils.lagrangian_penalties import all_penalties

# 距离与指标
from adv_lib.distances.lpips import LPIPS
from adv_lib.distances.color_difference import ciede2000_loss
from adv_lib.distances.structural_similarity import (
    compute_ssim,
    compute_ms_ssim,
)

# 攻击方法
from adv_lib.attacks import alma, ddn, sdf, df, fmn
from adv_lib.attacks.auto_pgd import minimal_apgd
from attacks.GeoSensFool import GSF as gsf
from attacks.SGSF import SGSF as sgsf
from attacks.original_fab import original_fab as fab
from adv_lib.attacks import carlini_wagner_l2 as cw
from models.imagenet import imagenet_model_factory
from util.utils import validate_and_combine_attack_data

# ========== 核心工具函数 ==========


def setup_seed(seed):
    """设置全局随机种子，确保确定性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


# ========== 主程序 ==========


def main():
    parser = argparse.ArgumentParser(
        description="Deterministic Resumable ImageNet Attack"
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--data-path", type=str, default=r"models/imagenet/imagenet_1000_random.pth"
    )
    parser.add_argument(
        "--out-dir", type=str, default=r"Results/ImageNet/Nature_Attack"
    )
    parser.add_argument("--seed", type=int, default=42, help="Global random seed")
    parser.add_argument("--quiet", action="store_true", help="Minimal console output")
    parser.add_argument(
        "--resume", type=str2bool, default=True, help="Enable resume from temp files"
    )

    args = parser.parse_args()

    # 1. 基础设置
    setup_seed(args.seed)
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 2. 日志设置
    log_path = out_dir / "NT_attack_final.log"
    logger = logging.getLogger("NT_attack")
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)

    def log_info(msg, print_also=True):
        """
        :param msg: 日志内容
        :param print_also: 是否同时打印到终端 (默认 True)
        """
        logger.info(msg)
        if print_also and not args.quiet:
            print(msg)

    log_info(
        f"[INIT] Device: {device} | Batch Size: {args.batch_size} | Global Seed: {args.seed}"
    )

    # 3. 数据加载
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"{args.data_path} not found")

    data = torch.load(args.data_path, weights_only=False)
    images_raw, labels_raw = data["images"], data["labels"]

    if images_raw.max() > 1.0:
        images_raw = images_raw.float() / 255.0
    else:
        images_raw = images_raw.float()

    dataset = TensorDataset(images_raw, labels_raw)
    g = torch.Generator()
    g.manual_seed(args.seed)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
    )
    log_info(f"[DATA] Loaded {len(dataset)} samples | {len(loader)} batches.")

    # 4. 模型加载
    log_info("[MODEL] Loading models...")
    model_configs = {
        "ResNet50": "resnet50",
        "ResNet50_l2_3": "resnet50",
        "ResNet50_linf_4": "resnet50",
    }
    models = {}
    for name, factory_name in model_configs.items():
        try:
            if name == "ResNet50":
                m = imagenet_model_factory("resnet50")[0]
            elif name == "ResNet50_l2_3":
                m = imagenet_model_factory(
                    "resnet50", state_dict_path="models/imagenet/imagenet_l2_3_0.pt"
                )[0]
            elif name == "ResNet50_linf_4":
                m = imagenet_model_factory(
                    "resnet50", state_dict_path="models/imagenet/imagenet_linf_4.pt"
                )[0]

            m.to(device).eval()
            requires_grad_(m, False)

            # 【关键】禁用 torch.compile，直接使用原模型，以保证结果一致性
            models[name] = m

        except Exception as e:
            log_info(f"[WARN] Failed to load {name}: {e}")

    log_info(f"[MODEL] Loaded: {list(models.keys())}")

    # 5. 攻击配置
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
        ("C&W_l2_9x10000", partial(cw, binary_search_steps=9, max_iterations=10000)),
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
        ("C&W_l2_9x1000", partial(cw, binary_search_steps=9, max_iterations=1000)),
        ("C&W_l2_9x10000", partial(cw, binary_search_steps=9, max_iterations=10000)),
    ]

    metrics_config = _default_metrics.copy()
    metrics_config["ssim"] = compute_ssim
    metrics_config["ciede2000"] = ciede2000_loss
    try:
        metrics_config["lpips"] = partial(LPIPS, linear_mapping="models/alex/alex.pth")
    except:
        pass

    # 6. 主循环
    for atk_name, atk_fn in attacks:
        for model_name, model in models.items():
            save_name = f"NT_{model_name}_{atk_name}.pt"
            save_path = out_dir / save_name
            temp_save_path = out_dir / f"TEMP_{model_name}_{atk_name}.pt"

            if args.resume and save_path.exists():
                log_info(f"[SKIP] {save_name} already exists.")
                continue

            log_info(f"\n[START] {model_name} - {atk_name}")

            attack_data_list = []
            start_batch_idx = 0
            processed_samples = 0
            # 断点恢复
            if args.resume and temp_save_path.exists():
                try:
                    attack_data_list = torch.load(
                        str(temp_save_path), weights_only=False
                    )
                    start_batch_idx = len(attack_data_list)
                    log_info(f"[RESUME] Resuming from Batch {start_batch_idx}...")
                except Exception as e:
                    logger.error(f"Temp file corrupted: {e}. Restarting.")
                    attack_data_list = []
                    start_batch_idx = 0

            pbar = tqdm(
                loader,
                desc=f"{model_name}-{atk_name}",
                total=len(loader),
                initial=start_batch_idx,
                leave=False,
            )
            data_iter = iter(loader)
            for _ in range(start_batch_idx):
                next(data_iter)

            for batch_idx_rel, (images, labels) in enumerate(data_iter):
                real_batch_idx = start_batch_idx + batch_idx_rel
                batch_start_time = time.time()
                setup_seed(args.seed + real_batch_idx)
                images, labels = images.to(device), labels.to(device)

                try:
                    # 运行攻击
                    attack_data = run_attack(
                        model=model,
                        inputs=images,
                        labels=labels,
                        attack=atk_fn,
                        batch_size=args.batch_size,
                    )

                    # [修复 KeyError] 手动计算 success
                    if "success" not in attack_data:
                        with torch.no_grad():
                            adv_inputs = attack_data["adv_inputs"]
                            outputs = model(adv_inputs)
                            preds = outputs.argmax(dim=1)
                            # 如果是非定向攻击
                            is_success = preds != labels
                            attack_data["success"] = is_success

                    # 计算简要统计
                    batch_asr = attack_data["success"].float().mean().item() * 100

                    with torch.no_grad():
                        diff = (attack_data["adv_inputs"] - images).view(
                            len(images), -1
                        )
                        batch_l2 = torch.norm(diff, p=2, dim=1).mean().item()
                    batch_time = time.time() - batch_start_time

                    # 【核心修改】 print_also=False: 只写 Log，不打印到终端
                    log_info(
                        f"[{atk_name}] Batch {real_batch_idx + 1}/{len(loader)} | ASR: {batch_asr:>5.1f}% | Avg L2: {batch_l2:>6.4f} | Time: {batch_time:>5.2f}s",
                        print_also=False,
                    )

                    attack_data_list.append(attack_data)
                    processed_samples += len(images)
                    if processed_samples >= 20:
                        break
                    torch.save(attack_data_list, str(temp_save_path))
                    pbar.update(1)
                except Exception as e:
                    logger.error(f"Batch {real_batch_idx} failed: {e}")
                    import traceback

                    logger.error(traceback.format_exc())
                    torch.save(attack_data_list, str(temp_save_path))
                    pbar.close()
                    raise e

            pbar.close()

            # 合并与结果记录
            is_valid, combined_data = validate_and_combine_attack_data(
                attack_data_list, verbose=False
            )
            if not is_valid:
                log_info(f"[ERROR] Combine failed for {save_name}")
                continue

            setup_seed(args.seed)
            attack_metrics = compute_attack_metrics(
                model=model,
                attack_data=combined_data,
                batch_size=args.batch_size,
                metrics=metrics_config,
            )
            torch.save(attack_metrics, str(save_path))
            log_info(f"[SAVED] {save_path}")

            if temp_save_path.exists():
                os.remove(str(temp_save_path))
            del attack_data_list, combined_data, attack_metrics
            torch.cuda.empty_cache()

    # ========== 7. Summary (汇总统计) ==========
    log_info("\n===== SUMMARY (ImageNet 1k ) =====")

    for atk_name, _ in attacks:
        for model_name in models.keys():
            save_name = f"NT_{model_name}_{atk_name}.pt"
            file_path = out_dir / save_name

            if not file_path.exists():
                continue

            try:
                metrics_blob = torch.load(str(file_path), map_location="cpu")
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {e}")
                continue

            success = metrics_blob.get("success", None)
            if success is None:
                continue

            success_rate = float(success.float().mean().item()) * 100.0
            log_info(
                f"\n[MODEL] {model_name:20s} [ATTACK] {atk_name:20s} "
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

                log_info(
                    f"  [{dist_name:9s}] Mean={mean_dist:.4f} | Median={median_dist:.4f}"
                )

            del metrics_blob

    log_info("\n[ALL DONE] - ImageNet evaluations completed.")


if __name__ == "__main__":
    main()
