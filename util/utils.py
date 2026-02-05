from typing import Tuple
import torch
from torch import Tensor
import numpy as np


import torch
import os
import sys
import random
import pathlib
from pathlib import Path
from torch import Tensor
from typing import Tuple
from typing import Any, Dict, Tuple, Optional
import torch.nn as nn
import numpy as np
import types

m = types.ModuleType("torchvision.models.utils")
m.load_state_dict_from_url = torch.hub.load_state_dict_from_url
sys.modules["torchvision.models.utils"] = m
# Robustness 库导入
from robustness.datasets import ImageNet
from robustness.model_utils import make_and_restore_model


def str2bool(v):
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ("yes", "true", "t", "y", "1"):
        return True
    if v in ("no", "false", "f", "n", "0"):
        return False
    raise ValueError("Boolean value expected.")


# ========== 辅助类：Wrapper 处理 Tuple 输出 ==========


class RobModelWrapper(nn.Module):
    """
    包装 robustness 模型，使其输出仅返回 logits (tuple[0])，
    适配 adv_lib 的接口要求。
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # robustness 模型通常返回 (logits, features)，我们只取 logits
        out = self.model(x)
        return out[0] if isinstance(out, tuple) else out


def load_robustness_model(arch, resume_path):
    """
    加载 Robustness 库训练的模型。
    使用假路径初始化 ImageNet 结构，仅用于加载权重。
    """
    # 使用临时路径，因为我们只关心模型架构，不加载实际数据
    ds = ImageNet("/tmp")

    # make_and_restore_model 返回 (model, checkpoint)
    model, _ = make_and_restore_model(arch=arch, dataset=ds, resume_path=resume_path)
    return model


# ========== 原子保存 ==========
def atomic_torch_save(obj, path: Path):
    """
    先写临时文件，再用 os.replace 原子替换目标文件。
    避免中断导致的半成品文件“存在但损坏”，从而被 resume 误判为完成。
    """
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, str(tmp))
    os.replace(str(tmp), str(path))


# ========== 完整性检查 ==========
def is_complete_metrics(path: Path, expected_samples: int) -> bool:
    """
    轻量完整性验证：
    - success 存在且长度匹配 expected_samples
    - distances 中（如存在）任意一个距离向量长度匹配 expected_samples
    """
    try:
        m = torch.load(str(path), map_location="cpu")
        if not isinstance(m, dict):
            return False

        s = m.get("success", None)
        if s is None or (not torch.is_tensor(s)):
            return False
        if s.numel() != expected_samples:
            return False

        dists = m.get("distances", {})
        if isinstance(dists, dict) and len(dists) > 0:
            k = next(iter(dists.keys()))
            v = dists[k]
            if torch.is_tensor(v) and v.numel() != expected_samples:
                return False

        return True
    except Exception:
        return False


def robust_accuracy_curve(
    distances: Tensor, successes: Tensor, worst_distance: float = float("inf")
) -> Tuple[Tensor, Tensor]:
    # 把攻击失败的样本赋值为 worst_distance
    worst_case_distances = distances.clone()
    worst_case_distances[~successes] = worst_distance

    # 排序后的唯一半径
    unique_distances, _ = torch.sort(worst_case_distances.unique())

    # 逐个计算鲁棒精度
    robust_accuracies = (
        (worst_case_distances.unsqueeze(0) > unique_distances.unsqueeze(1))
        .float()
        .mean(1)
    )

    # 确保单调递减（避免数值抖动）
    for i in range(1, len(robust_accuracies)):
        robust_accuracies[i] = min(robust_accuracies[i], robust_accuracies[i - 1])

    return unique_distances, robust_accuracies


# def robust_accuracy_curve(
#     distances: Tensor, successes: Tensor, worst_distance: float = float("inf")
# ) -> Tuple[Tensor, Tensor]:
#     """
#     计算鲁棒性曲线 (Memory Efficient Version).
#     复杂度: O(N log N) 由于排序，而不是 O(N^2)
#     """
#     # 1. 处理攻击失败的样本
#     # 使用 clone 防止修改原始数据
#     dists = distances.clone()
#     dists[~successes] = worst_distance

#     # 2. 排序
#     # sorted_dists 即为 x 轴坐标
#     sorted_dists, _ = torch.sort(dists)

#     # 3. 计算鲁棒精度 (Y轴)
#     # 逻辑: 对于排好序的第 i 个距离 d[i]，
#     # 意味着有 i 个样本的距离 <= d[i] (即被攻击成功)
#     # 那么剩下的 (N - 1 - i) 个样本距离 > d[i] (即鲁棒)
#     n = len(dists)

#     # 生成从 0 到 N-1 的索引
#     indices = torch.arange(n, device=dists.device)

#     # 计算大于当前阈值的比例
#     # 公式: (总数 - 当前排位 - 1) / 总数
#     # 例如: 最小的距离(idx=0)，有 (N-1)/N 的样本比它大
#     robust_accuracies = (n - 1 - indices).float() / n

#     return sorted_dists, robust_accuracies


def validate_and_combine_attack_data(attack_data_list, verbose=False):
    """
    验证并合并多个攻击数据字典，允许最后一个 batch size 不同

    Args:
        attack_data_list: run_attack 返回的字典列表
        verbose: 是否打印详细信息

    Returns:
        tuple: (is_valid, combined_data)
    """
    import torch

    if not attack_data_list:
        if verbose:
            print("Error: Empty attack data list")
        return False, None

    if verbose:
        print(
            f"Validating and combining {len(attack_data_list)} attack data entries..."
        )

    # 1. 验证所有字典有相同的键
    first_keys = set(attack_data_list[0].keys())
    for i, data in enumerate(attack_data_list[1:], 1):
        if set(data.keys()) != first_keys:
            if verbose:
                print(f"Error: Data entry {i} has different keys")
            return False, None

    validation_results = {}
    for key in first_keys:
        values = [data[key] for data in attack_data_list if key in data]

        if key in ["inputs", "labels", "adv_inputs"]:
            if not all(isinstance(v, torch.Tensor) for v in values):
                if verbose:
                    print(f"Error: Not all values are tensors for key '{key}'")
                return False, None

            # ✅ 只验证特征维度是否一致，忽略 batch 维度
            feature_shape = values[0].shape[1:]
            for i, v in enumerate(values, 1):
                if v.shape[1:] != feature_shape:
                    if verbose:
                        print(f"Error: Feature shape mismatch for key '{key}'")
                        print(f"Expected: {feature_shape}, got {v.shape[1:]}")
                    return False, None

        elif key == "targets":
            if not all(v is None or isinstance(v, torch.Tensor) for v in values):
                if verbose:
                    print(f"Error: Invalid targets values")
                return False, None

        elif key in ["time", "num_forwards", "num_backwards"]:
            if not all(isinstance(v, (int, float)) for v in values):
                if verbose:
                    print(f"Error: Not all values are numeric for key '{key}'")
                return False, None
            if any(v < 0 for v in values):
                if verbose:
                    print(f"Error: Negative values found for key '{key}'")
                return False, None

        validation_results[key] = True

    # 2. 合并数据
    if all(validation_results.values()):
        combined_data = {}
        total_entries = len(attack_data_list)

        for key in first_keys:
            values = [data[key] for data in attack_data_list if key in data]

            if key in ["inputs", "labels", "adv_inputs"]:
                combined_data[key] = torch.cat(values, dim=0)  # ✅ 允许最后 batch 更小
                if verbose:
                    print(f"\nConcatenated {key}: shape {combined_data[key].shape}")

            elif key == "targets":
                combined_data[key] = (
                    values[0] if all(v == values[0] for v in values) else None
                )

            elif key in ["time", "num_forwards", "num_backwards"]:
                combined_data[key] = sum(values) / total_entries
                if verbose:
                    print(f"\nAverage {key}: {combined_data[key]:.4f}")

            else:
                combined_data[key] = values[0]

        if verbose:
            print("\nData validation and combination completed successfully")
        return True, combined_data

    if verbose:
        print("\nData validation failed")
    return False, None


# augmented_lagrangian_adversarial_attacks/util/_subset_base.py
# -*- coding: utf-8 -*-

import os
import sys
from typing import Sequence, Dict, Any, Tuple

import torch
from torch.utils.data import DataLoader, Subset

# 项目根
_CUR = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.dirname(_CUR)
sys.path.insert(0, _PROJ)

# 全局确定性
torch.backends.cudnn.deterministic = True
torch.manual_seed(42)

# 构建子集时用于扫描 dataset 的 batch
DEFAULT_BUILD_BATCH = 1024


# ---------- 通用路径/缓存 ----------
def cache_dir(dataset_key: str) -> str:
    p = os.path.join(_PROJ, "data", "cache", dataset_key)
    os.makedirs(p, exist_ok=True)
    return p


def subset_pth_path(dataset_key: str, model_key: str, k: int = 1000) -> str:
    return os.path.join(
        cache_dir(dataset_key), f"{dataset_key}_{k}_clean_{model_key}.pth"
    )


# ---------- 通用工具 ----------
def extract_labels_from_dataset(dataset, indices: Sequence[int]) -> torch.Tensor:
    idx = torch.as_tensor(indices, dtype=torch.long)
    if hasattr(dataset, "targets"):
        all_targets = torch.as_tensor(dataset.targets)
        return all_targets[idx]
    ys = [dataset[i][1] for i in idx.tolist()]
    return torch.as_tensor(ys, dtype=torch.long)


def save_indices_labels(
    dataset_key: str,
    model_key: str,
    indices: Sequence[int],
    labels: torch.Tensor,
    k: int,
    out_path: str | None = None,
) -> str:
    out_path = out_path or subset_pth_path(dataset_key, model_key, k=k)
    blob: Dict[str, Any] = {
        "indices": torch.as_tensor(indices, dtype=torch.long),
        "labels": labels.to(torch.long).cpu(),
        "meta": {
            "type": "clean",
            "k": len(indices),
            "model": model_key,
            "dataset": dataset_key,
            "seed": 42,
        },
    }
    torch.save(blob, out_path)
    return out_path


def subset_loader_from_existing_dataset(
    dataset,
    subset_pth: str,
    batch_size_loader: int = 1024,
    shuffle: bool = False,
) -> DataLoader:
    blob = torch.load(subset_pth, map_location="cpu")
    indices = blob["indices"].tolist()
    return DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size_loader,
        shuffle=shuffle,
        num_workers=0,
    )


# ---------- 通用“用模型生成干净 K 子集” ----------
DEFAULT_BUILD_BATCH = 1024  # 若你已有同名常量，可删掉这行


def _set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def build_clean_k_with_model(
    dataset_key: str,
    model_key: str,
    dataset,
    model: torch.nn.Module,  # 已 .eval().to(device) 且载入权重
    k: int = 1000,
    batch_size_eval: int = DEFAULT_BUILD_BATCH,
    device: torch.device | str = "cuda",
    force_rebuild: bool = False,
    require_full_k: bool = True,  # 若不足 k，默认继续扫描直至遍历全集
    seed: int = 42,  # ✅ 固定种子
    score_metric: str = "margin",  # ✅ 打分方式: "margin" 或 "confidence"
    tau: Optional[float] = None,  # ✅ margin/置信度的门槛（例如 3.5）
    per_class: Optional[int] = None,  # ✅ 类均衡：每类取多少（MNIST 可设 100）
    num_classes: int = 10,
) -> Tuple[str, torch.Tensor, torch.Tensor]:
    """
    用传入模型在传入数据集上筛出“预测正确”的样本，并按 score 排序+确定性 tie-break 选出前 k。
    若 per_class 不为 None，则改为每类取 per_class（总 k = per_class * num_classes）。
    返回：(out_path, indices, labels)
    """
    out_path = subset_pth_path(
        dataset_key, model_key, k=k if per_class is None else per_class * num_classes
    )

    # ---------- 缓存判定与回收 ----------
    if os.path.exists(out_path) and not force_rebuild:
        try:
            blob = torch.load(out_path, map_location="cpu")
            old_idx = blob.get("indices", None)
            old_lbl = blob.get("labels", None)
            if old_idx is not None and old_lbl is not None:
                target_k = k if per_class is None else per_class * num_classes
                if len(old_idx) >= target_k:
                    if len(old_idx) != target_k:
                        # 截断并回写一致的 k（保持缓存可重用）
                        new_blob = dict(blob)
                        new_blob["indices"] = old_idx[:target_k]
                        new_blob["labels"] = old_lbl[:target_k]
                        meta = dict(new_blob.get("meta", {}))
                        meta["k"] = target_k
                        new_blob["meta"] = meta
                        torch.save(new_blob, out_path)
                    return out_path, old_idx[:target_k], old_lbl[:target_k]
        except Exception:
            pass  # 缓存坏了则忽略

    # ---------- 正式构建（确定性） ----------
    _set_all_seeds(seed)
    model = model.eval().to(device)

    loader_eval = DataLoader(
        dataset=dataset,
        batch_size=batch_size_eval,
        shuffle=False,  # ✅ 不打乱，保证遍历顺序稳定
        num_workers=0,  # ✅ 多进程可能引入不确定性
        pin_memory=False,
    )

    # 收集 (global_index, label, score)
    scored: list[Tuple[int, int, float]] = []
    base = 0
    for images, labels in loader_eval:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        preds = logits.argmax(1)
        ok = preds.eq(labels)

        # 计算分数
        if score_metric == "margin":
            # margin = logit[y] - max_{j!=y} logit[j]
            top2 = torch.topk(logits, k=2, dim=1).values  # shape [B,2]
            # 注意：top2[:,0] 对应最大 logit；若预测正确，top2[:,0] 就是 y 的 logit
            # 更稳妥：用 gather 获取 y 的 logit
            y_logit = logits.gather(1, labels.view(-1, 1)).squeeze(1)
            # 其它类别最大 logit
            # 使用 -inf 屏蔽 y 类
            mask = torch.ones_like(logits, dtype=torch.bool)
            mask.scatter_(1, labels.view(-1, 1), False)
            other_max = logits.masked_fill(~mask, float("-inf")).max(dim=1).values
            score = (y_logit - other_max).detach().float()
        elif score_metric == "confidence":
            # 置信度：softmax 的 y 概率
            prob = torch.softmax(logits, dim=1)
            score = prob.gather(1, labels.view(-1, 1)).squeeze(1).detach().float()
        else:
            raise ValueError(f"Unsupported score_metric: {score_metric}")

        # 只收正确的
        ok_idx = torch.nonzero(ok, as_tuple=True)[0]
        for i in ok_idx.tolist():
            gidx = base + i
            s = float(score[i].cpu())
            y = int(labels[i].cpu())
            # tau 门槛（如 margin >= tau）
            if (tau is None) or (s >= tau):
                scored.append((gidx, y, s))

        base += images.size(0)

    if len(scored) == 0:
        raise RuntimeError(
            f"[{dataset_key}:{model_key}] 没有样本通过筛选条件（正确性 + tau）"
        )

    # ---------- 确定性排序：(-score, global_index) ----------
    scored.sort(key=lambda x: (-x[2], x[0]))

    # ---------- 选样（全局或按类均衡） ----------
    if per_class is None:
        # 全局前 k
        selected = scored[:k]
    else:
        # 类均衡：每类 per_class
        buckets: Dict[int, list[Tuple[int, int, float]]] = {
            c: [] for c in range(num_classes)
        }
        for triplet in scored:
            c = triplet[1]
            if len(buckets[c]) < per_class:
                buckets[c].append(triplet)
            # 所有类都满了即可提前结束
            if all(len(buckets[c]) >= per_class for c in range(num_classes)):
                break
        # 汇总
        selected = []
        for c in range(num_classes):
            if len(buckets[c]) < (per_class or 0):
                msg = (
                    f"[WARN] {dataset_key}:{model_key}: 类 {c} 候选不足 "
                    f"{len(buckets[c])}/{per_class}（可考虑下调 tau 或取消 per_class）。"
                )
                print(msg)
            selected.extend(buckets[c])
        # 若总数不足且 require_full_k，则发警告
        target_k = (per_class or 0) * num_classes
        if len(selected) < target_k and require_full_k:
            print(
                f"[WARN] {dataset_key}:{model_key}: 仅收集到 {len(selected)}/{target_k} 样本。"
            )

    # ---------- 输出 indices/labels ----------
    sel_idx = [g for (g, y, s) in selected]
    sel_lbl = [y for (g, y, s) in selected]
    labels_k = extract_labels_from_dataset(dataset, sel_idx)

    # 保险：labels_k 与 sel_lbl 一致性（正常应一致）
    if len(labels_k) == len(sel_lbl):
        # 不一致时以真实 labels_k 为准，但打印提示
        mismatch = labels_k.cpu().tolist() != sel_lbl
        if mismatch:
            print(
                f"[WARN] {dataset_key}:{model_key}: 发现 label 不一致，已以 dataset 中的 label 为准。"
            )

    final_k = len(sel_idx)
    save_path = save_indices_labels(
        dataset_key, model_key, sel_idx, labels_k, k=final_k, out_path=out_path
    )

    # 写回 meta（把 seed/score/tau 等签进缓存，便于复现）
    try:
        blob = torch.load(save_path, map_location="cpu")
        meta = dict(blob.get("meta", {}))
        meta.update(
            {
                "k": final_k,
                "seed": seed,
                "score_metric": score_metric,
                "tau": tau,
                "per_class": per_class,
                "num_classes": num_classes,
                "deterministic_sort": "(-score, global_index)",
            }
        )
        blob["meta"] = meta
        torch.save(blob, save_path)
    except Exception as e:
        print(f"[WARN] 写回 meta 失败：{e}")

    return save_path, torch.as_tensor(sel_idx, dtype=torch.long), labels_k


def build_clean_k_for_models(
    dataset_key: str,
    dataset,
    models: Dict[str, torch.nn.Module],  # {model_key: model(with weights)}
    k: int = 1000,
    batch_size_eval: int = DEFAULT_BUILD_BATCH,
    device: torch.device | str = "cuda",
    force_rebuild: bool = False,
    require_full_k: bool = True,
    seed: int = 42,
    score_metric: str = "margin",
    tau: Optional[float] = None,
    per_class: Optional[int] = None,
    num_classes: int = 10,
) -> Dict[str, Dict[str, Any]]:
    """
    逐模型构建干净子集；确定性（固定 seed + 排序 + tie-break）。
    - 若 per_class 非 None，则每类取 per_class，总样本数 = per_class * num_classes，k 参数将被忽略。
    """
    results: Dict[str, Dict[str, Any]] = {}
    for model_key, model in models.items():
        out_path, idx, lbl = build_clean_k_with_model(
            dataset_key=dataset_key,
            model_key=model_key,
            dataset=dataset,
            model=model,
            k=k,
            batch_size_eval=batch_size_eval,
            device=device,
            force_rebuild=force_rebuild,
            require_full_k=require_full_k,
            seed=seed,
            score_metric=score_metric,
            tau=tau,
            per_class=per_class,
            num_classes=num_classes,
        )
        print(f"[OK] {dataset_key}:{model_key} -> {out_path} (n={len(idx)})")
        results[model_key] = {"path": out_path, "indices": idx, "labels": lbl}
    return results


import torch


import torch


def print_balance_summary(
    results: dict,
    expect_per_class: int = None,
    num_classes: int = 10,
    tol: int = 0,
):
    """
    打印各模型子集的类别分布，并判断是否均衡。
    （不保存文件，仅控制台输出）

    参数：
        results           : build_clean_k_for_models 的返回值
        expect_per_class  : 理论每类样本数（如每类100张）。None 则仅显示分布
        num_classes       : 类别总数
        tol               : 允许的浮动范围（例如 tol=2 表示 ±2 内视为均衡）
    """
    all_balanced = True

    print("\n====== 类别分布检查 ======\n")
    for model_name, info in results.items():
        labels = info["labels"].view(-1).cpu()
        cnt = torch.bincount(labels, minlength=num_classes)
        total = int(cnt.sum().item())

        # 输出每个类别数量
        print(f"[{model_name}] total={total}")
        for c in range(num_classes):
            print(f"  class {c}: {int(cnt[c])}")

        # 判断是否均衡
        balanced = True
        if expect_per_class is not None:
            for c in range(num_classes):
                if abs(int(cnt[c]) - expect_per_class) > tol:
                    balanced = False
            if total != expect_per_class * num_classes:
                balanced = False

        # 打印结论
        if expect_per_class is None:
            print(f"  ℹ️ 未设置期望样本数，仅显示分布。")
        elif balanced:
            print(f"  ✅ 数据集类别分布均衡（±{tol}）")
        else:
            print(f"  ⚠️ 数据集类别分布不均衡！建议检查样本筛选逻辑。")

        all_balanced &= balanced
        print("")

    if expect_per_class is not None:
        print("====== 检查结果汇总 ======")
        print("✅ 全部均衡" if all_balanced else "❌ 存在不均衡模型")
    print("")


import torch
from tqdm import tqdm
from torch.utils.data import Subset, DataLoader
from typing import Dict, Tuple, Optional

def evaluate_accuracy(model, data_loader, device, topk=(1, 5)):
    """
    计算模型在给定数据集上的 Top-k 准确率 (函数版)
    
    参数:
        model: PyTorch 模型
        data_loader: DataLoader
        device: 运行设备
        topk: tuple, 需要计算的 Top-k 值，默认 (1, 5)
    
    返回:
        results: dict, key为k值, value为对应的准确率(百分比 float)
    """
    model.eval()
    maxk = max(topk)
    batch_size = data_loader.batch_size
    
    # 初始化计数器
    correct_counts = {k: 0 for k in topk}
    total = 0
    
    # 进度条描述
    desc = f"Eval Acc (Top-{topk})"
    
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc=desc, leave=False):
            images = images.to(device)
            labels = labels.to(device)
            total += labels.size(0)

            outputs = model(images)
            
            # --- Top-k 计算核心逻辑 ---
            # 获取前 maxk 个预测值的索引 (values, indices)
            _, pred = outputs.topk(maxk, 1, True, True)
            pred = pred.t() # 转置为 (maxk, batch_size)
            
            # 将标签扩展为相同形状以便广播对比
            correct = pred.eq(labels.view(1, -1).expand_as(pred))

            for k in topk:
                # 取前 k 行，看是否包含 True（只要前 k 个里有一个对，就是 Top-k 正确）
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                correct_counts[k] += correct_k.item()

    # 计算百分比
    results = {}
    for k in topk:
        results[k] = 100.0 * correct_counts[k] / total
        
    return results
@torch.no_grad()
def evaluate_clean_subset_accuracy(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    indices: torch.Tensor,
    labels_gt: torch.Tensor,
    device: torch.device | str = "cuda",
    batch_size: int = 1024,
    quiet: bool = False,
) -> Tuple[float, list]:
    """
    在指定的干净样本子集上评估模型分类准确率。

    参数:
        model        - 已加载权重并 .eval().to(device) 的模型
        dataset      - PyTorch Dataset 对象（MNIST/CIFAR10 等）
        indices      - Tensor[int]，所选样本的全局索引
        labels_gt    - Tensor[int]，对应的真实标签
        device       - 设备（"cuda" 或 "cpu"）
        batch_size   - 推理 batch 大小
        quiet        - 若 True，则不打印详细错误样本信息

    返回:
        acc          - 准确率 (float)
        err_info     - 错误样本列表 [(global_idx, y_true, y_pred), ...]
    """
    model.eval()
    indices = indices.view(-1).cpu().long()
    labels_gt = labels_gt.view(-1).cpu().long()

    sub_ds = Subset(dataset, indices.tolist())
    loader = DataLoader(
        sub_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False
    )

    all_preds = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1).detach().cpu()
            all_preds.append(preds)
    preds_all = torch.cat(all_preds, dim=0)

    assert len(preds_all) == len(labels_gt), "预测长度与标签长度不一致！"

    correct = (preds_all == labels_gt).sum().item()
    acc = correct / len(labels_gt)

    # 收集错误样本信息
    err_mask = preds_all != labels_gt
    err_indices = torch.nonzero(err_mask, as_tuple=False).view(-1)
    err_info = []
    for i in err_indices.tolist():
        global_idx = int(indices[i])
        err_info.append((global_idx, int(labels_gt[i]), int(preds_all[i])))

    if not quiet:
        print(f"[ACC] 子集准确率: {acc*100:.2f}% （正确 {correct}/{len(labels_gt)}）")
        if acc < 1.0 and len(err_info) > 0:
            print("  前 20 个错样示例 (全局索引, 真标签, 预测值)：")
            for g, y, yp in err_info[:20]:
                print(f"   - ({g}, {y} -> {yp})")

    return acc, err_info


def evaluate_all_models_accuracy(
    models: Dict[str, torch.nn.Module],
    dataset: torch.utils.data.Dataset,
    results: Dict[str, Dict[str, torch.Tensor]],
    device: torch.device | str = "cuda",
    batch_size: int = 1024,
):
    """
    依次评估多个模型在各自干净子集上的准确率。
    results 应为 build_clean_k_for_models 的返回值。
    """
    summary = {}
    print("\n[STEP] 开始批量评估各模型 1k 子集准确率...\n")
    for model_name, model in models.items():
        info = results[model_name]
        acc, err_info = evaluate_clean_subset_accuracy(
            model=model,
            dataset=dataset,
            indices=info["indices"],
            labels_gt=info["labels"],
            device=device,
            batch_size=batch_size,
            quiet=False,
        )
        summary[model_name] = {"acc": acc, "n_err": len(err_info)}
    print("\n[SUMMARY]")
    for name, s in summary.items():
        print(f"  {name:<25}  acc={s['acc']*100:.2f}%  错样数={s['n_err']}")
    return summary
