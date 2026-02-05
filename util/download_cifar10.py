"""
download_cifar10.py
仅负责下载和加载 CIFAR-10。
数据根目录固定为项目根的 data/。
"""

from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 项目根：.../GEOSENSFOOL/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
TORCHVISION_DIR = DATA_ROOT / "torchvision"
TORCHVISION_DIR.mkdir(parents=True, exist_ok=True)

# 默认增强
TRAIN_TRANSFORM = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
)
TEST_TRANSFORM = transforms.ToTensor()


def download_cifar10(download: bool = True) -> None:
    """下载 CIFAR-10 到 GEOSENSFOOL/data/torchvision 下。"""
    datasets.CIFAR10(TORCHVISION_DIR, train=True, download=download)
    datasets.CIFAR10(TORCHVISION_DIR, train=False, download=download)
    print(f"CIFAR-10 已准备好：{TORCHVISION_DIR}")


def get_dataset(train: bool = True, transform=None):
    """获取 CIFAR-10 数据集（可自定义 transform）。"""
    transform = transform or (TRAIN_TRANSFORM if train else TEST_TRANSFORM)
    return datasets.CIFAR10(
        TORCHVISION_DIR, train=train, transform=transform, download=False
    )


def get_loader(
    dataset,
    batch_size: int = 128,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


# ---------------- demo ----------------
if __name__ == "__main__":
    # 简单演示：下载 -> 取 test dataset -> 构造 loader -> 取一个 batch 并打印信息
    print("==== CIFAR-10 demo ====")
    download_cifar10(download=True)

    ds = get_dataset(train=False)  # test set, 默认 transform: ToTensor()
    print(f"数据集长度 (test): {len(ds)}")

    loader = get_loader(ds, batch_size=64, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    imgs, labels = batch
    print(f"取到一个 batch: imgs.shape={imgs.shape}, labels.shape={labels.shape}")
    # 统计该 batch 的 label 分布（示例）
    import collections

    cnt = collections.Counter(int(x) for x in labels)
    print("labels count in this batch (partial):", dict(list(cnt.items())[:10]))

    # 显示一个样本的像素统计（tensor）
    s0 = imgs[0]
    print(
        "第一个样本张量：min, max, mean =",
        float(s0.min()),
        float(s0.max()),
        float(s0.mean()),
    )
    print("==== demo 完成 ====")
