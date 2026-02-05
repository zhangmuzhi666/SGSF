"""
build_mnist_1k.py
构建或加载 MNIST 的 1k 均衡子集（每类100样本，保持原始顺序）。
保存路径：data/cache/mnist/mnist_1000_balanced.pth
"""

from pathlib import Path
import torch
from torch.utils.data import Subset
import os
import sys

# 添加项目根目录到路径
# ========== 修复导入路径 ==========
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
try:
    from util.download_cifar10 import get_dataset, get_loader
except ImportError:
    # 如果 util 模块不存在，尝试直接导入
    from download_cifar10 import get_dataset, get_loader

# === 路径设置 ===
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
CACHE_DIR = DATA_ROOT / "cache" / "mnist"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

SUBSET_FILE = CACHE_DIR / "mnist_1k_balanced.pth"


def build_mnist_1k_subset(dataset=None, per_class_k=100, overwrite=False):
    """
    构建 MNIST 均衡子集（每类 per_class_k 个，默认100，共1000个）。
    如果缓存存在且未指定 overwrite=True，则直接返回路径。
    """
    if SUBSET_FILE.exists() and not overwrite:
        print(f"[跳过] 已存在: {SUBSET_FILE}")
        return SUBSET_FILE

    if dataset is None:
        dataset = get_dataset(train=False)

    class_counts = {}
    indices, labels = [], []
    for i, (_, lbl) in enumerate(dataset):
        lbl = int(lbl)
        if class_counts.get(lbl, 0) < per_class_k:
            indices.append(i)
            labels.append(lbl)
            class_counts[lbl] = class_counts.get(lbl, 0) + 1
        if len(class_counts) == 10 and all(
            v >= per_class_k for v in class_counts.values()
        ):
            break

    # 保存结构化信息
    torch.save(
        {
            "indices": indices,
            "labels": labels,
            "meta": {"dataset": "mnist", "per_class_k": per_class_k},
        },
        SUBSET_FILE,
    )

    print(f"[OK] 构建完成，保存至 {SUBSET_FILE}")
    return SUBSET_FILE


def subset_loader_from_pth(
    dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True
):
    """
    根据缓存文件创建 DataLoader。
    """
    info = torch.load(SUBSET_FILE)
    sub = Subset(dataset, info["indices"])
    return get_loader(
        sub,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


# === Demo ===
if __name__ == "__main__":
    from download_mnist import download_mnist
    import collections

    print("==== 构建 MNIST 1k 均衡子集 ====")
    download_mnist()
    ds = get_dataset(train=False)
    subset_pth = build_mnist_1k_subset(dataset=ds)
    print(f"子集路径: {subset_pth}")

    loader = subset_loader_from_pth(ds)
    imgs, labels = next(iter(loader))
    cnt = collections.Counter(int(x) for x in labels)
    print("首批标签分布:", dict(sorted(cnt.items())))
    print("==== 完成 ====")
