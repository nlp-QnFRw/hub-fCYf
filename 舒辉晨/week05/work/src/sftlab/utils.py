"""配置、随机种子、设备选择等通用工具。

配置文件的路径约定：所有相对路径都以**项目根目录**（work/）为基准，
这样无论从哪个工作目录启动脚本，路径都能对上。
"""
from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

# utils.py -> sftlab -> src -> work
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ── 路径与配置 ────────────────────────────────────────────────────────────────

def resolve(path: str | os.PathLike) -> Path:
    """把配置里的相对路径按项目根目录解析成绝对路径。"""
    p = Path(path)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def load_config(path: str | os.PathLike) -> dict[str, Any]:
    """读取 yaml 配置，并做一次最小校验。"""
    cfg_path = resolve(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"找不到配置文件: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    for key in ("data", "model", "train"):
        if key not in cfg:
            raise KeyError(f"配置 {cfg_path.name} 缺少必需的段落: {key}")
    cfg.setdefault("stage", "sft")
    cfg.setdefault("seed", 1337)
    return cfg


# ── 随机性 ────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    """固定一切随机源，保证同配置两次训练结果可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ── 设备 ──────────────────────────────────────────────────────────────────────

def get_device(prefer: str = "auto") -> torch.device:
    """按 auto / cuda / cpu 选择设备；cuda 不可用时安静地退回 cpu。"""
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer in ("cuda", "auto") and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def describe_device(device: torch.device) -> str:
    """打印一行设备信息，方便确认这次到底跑在 GPU 还是 CPU 上。"""
    if device.type == "cuda":
        idx = device.index or 0
        name = torch.cuda.get_device_name(idx)
        mem = torch.cuda.get_device_properties(idx).total_memory / 1024**3
        return f"cuda:{idx} · {name} · {mem:.1f} GB"
    return f"cpu · {os.cpu_count()} 线程"


# ── 杂项 ──────────────────────────────────────────────────────────────────────

def human_time(seconds: float) -> str:
    """秒 → 人类可读时长。"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s"


class AverageMeter:
    """滑动平均，用来把零散的 loss 汇总成一行日志。"""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count else 0.0


def ppl(loss: float) -> float:
    """困惑度 = exp(loss)。超过 50 时继续算意义不大，做个截断防止溢出。"""
    return math.exp(min(loss, 20.0))


def save_json(obj: Any, path: str | os.PathLike) -> None:
    path = resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str | os.PathLike) -> Any:
    with open(resolve(path), "r", encoding="utf-8") as f:
        return json.load(f)


class Logger:
    """同时往终端和日志文件写，训练完还能回看曲线数字。"""

    def __init__(self, log_path: str | os.PathLike | None = None) -> None:
        self.path = resolve(log_path) if log_path else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("", encoding="utf-8")

    def __call__(self, msg: str = "") -> None:
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        if self.path:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
