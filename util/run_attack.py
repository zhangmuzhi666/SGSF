import torch
import tqdm
from typing import Callable, Optional
from torch import nn, Tensor
from functools import partial


def run_attack(
    model: nn.Module,
    inputs: Tensor,
    labels: Tensor,
    attack: Callable,
    targets: Optional[Tensor] = None,
    batch_size: Optional[int] = None,
) -> Tensor:
    """
    极简版 run_attack：
    - 对所有样本都执行 attack（不做 already-adv 检查）
    - 返回 adv_inputs（每个样本都被扰动，无论攻击是否成功）
    """
    model_device = next(model.parameters()).device
    to_device = lambda t: t.to(model_device)
    targeted, adv_labels = False, labels
    if targets is not None:
        targeted, adv_labels = True, targets
    batch_size = batch_size or len(inputs)

    advs_chunks = []
    chunks = [tensor.split(batch_size) for tensor in [inputs, adv_labels]]

    for inputs_chunk, label_chunk in tqdm.tqdm(
        zip(*chunks), ncols=80, total=len(chunks[0])
    ):
        batch_d, label_d = [to_device(t.clone()) for t in (inputs_chunk, label_chunk)]
        advs_chunk_d = attack(model, batch_d, label_d, targeted=targeted)
        advs_chunks.append(advs_chunk_d.to(inputs.device))

        # 若 attack 是 partial 且带 callback，照旧重置窗口（与原版一致）
        if (
            isinstance(attack, partial)
            and (cb := attack.keywords.get("callback")) is not None
        ):
            cb.reset_windows()

    adv_inputs = torch.cat(advs_chunks, 0) if len(advs_chunks) > 1 else advs_chunks[0]
    return adv_inputs
