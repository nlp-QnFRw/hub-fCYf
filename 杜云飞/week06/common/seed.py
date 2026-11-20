#region 0. 模块说明
"""
随机性控制。**所有**训练 / 抽样 / 评估脚本开头都要 `set_seed(42)`。

为什么必须集中：
  前一个项目（text_classification）忘了设 seed，同一条命令跑两次 val_acc
  分别是 0.5646 / 0.5595（差 0.51pp）——这恰好是很多"消融结论"的量级，
  等于所有小数点后两位的对比都不可信。设了之后两次逐位一致。

设计重点：
  - torch / numpy / random 三者都要设，只设 torch 不够（数据抽样走 random）；
  - `cudnn.deterministic=True` 会让部分卷积变慢，但本任务是 Transformer，代价可忽略；
  - 提供 `seed_worker` 给 DataLoader（Windows 无 fork，worker 不继承主进程随机状态）。

使用方式：
  from common.seed import set_seed
  set_seed(42)
"""
#endregion

#region 1. 导入依赖
from __future__ import annotations

import os
import random

import numpy as np
import torch

#endregion

#region 2. 主函数
DEFAULT_SEED: int = 42


def set_seed(seed: int = DEFAULT_SEED, deterministic: bool = True) -> None:
    """一次性固定所有随机源。返回 None（不需要链式调用）。

    参数：
      seed          : 随机种子，默认 42（全项目统一，方便跨方法对比）
      deterministic : 是否开启 cudnn 确定性算法（评估与复现实验必须为 True）
    """
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  # 开了 benchmark 会自动选最快 kernel，结果不可复现


def seed_worker(worker_id: int) -> None:
    """DataLoader 的 worker_init_fn：让每个 worker 的随机状态也确定。

    numpy 种子用 `seed + worker_id`，避免所有 worker 抽到同一批数据。
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
#endregion

#region 3. 自检
if __name__ == "__main__":
    set_seed(42)
    a = [random.random() for _ in range(3)]
    set_seed(42)
    b = [random.random() for _ in range(3)]
    assert a == b, "set_seed 不可复现"
    print("set_seed 可复现 ✓", [round(x, 6) for x in a])
#endregion
