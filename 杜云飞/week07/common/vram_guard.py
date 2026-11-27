# -*- coding: utf-8 -*-
"""
vram_guard.py — 8GB 笔记本显存保护（本机先例：RTX 5050 Laptop 8GB 跑满会卡死/掉驱动）

策略（三层防线，全部落日志，降档本身也是实验数据）：
  1. 训练前检查可用显存：free < 阈值（默认总显存 85%）→ 自动调低 batch、等比提高梯度累积；
  2. 训练中捕获 torch.cuda.OutOfMemoryError → 清缓存、降一档 batch 重试该 epoch；
  3. 每次调整都把 (目标batch, 实际batch, grad_accum, 峰值显存, 降档次数) 记进 meta。

用法：
  from vram_guard import VramGuard
  guard = VramGuard()
  batch_size, grad_accum = guard.adjust(args.batch_size, args.grad_accum)
  ... 训练循环里：
  try:
      run_epoch(...)
  except torch.cuda.OutOfMemoryError:
      batch_size, grad_accum = guard.on_oom(model, optimizer)   # 降档重试
"""
import os
import sys

import torch

# 预留的系统/桌面显存：Windows 桌面合成器 + 其它进程常驻约占 0.8~1.2GB
DEFAULT_RATIO = 0.85


class VramGuard:
    def __init__(self, ratio: float = None, device=None):
        self.ratio = float(ratio or os.environ.get("VRAM_RATIO", DEFAULT_RATIO))
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.downgrades = 0
        self.history = []  # [(batch, accum, free_mb_before)]

    def _free_mb(self) -> float:
        if self.device.type != "cuda":
            return float("inf")
        free, total = torch.cuda.mem_get_info()
        return free / 1024 ** 2

    def budget_mb(self) -> float:
        if self.device.type != "cuda":
            return float("inf")
        _, total = torch.cuda.mem_get_info()
        return total / 1024 ** 2 * self.ratio

    def adjust(self, batch_size: int, grad_accum: int) -> tuple[int, int]:
        """可用显存低于阈值时逐级降 batch（最小 1），grad_accum 等比放大，保持等效 batch 不变。"""
        if self.device.type != "cuda":
            return batch_size, grad_accum
        effective = batch_size * grad_accum
        orig = batch_size
        free = self._free_mb()
        while batch_size > 1 and free < self.budget_mb():
            batch_size = max(1, batch_size // 2)
            grad_accum = max(1, effective // batch_size)
            self.downgrades += 1
            free = self._free_mb()
        self.history.append({"orig_batch": orig, "batch": batch_size, "grad_accum": grad_accum,
                             "free_mb": round(self._free_mb(), 0),
                             "downgrades": self.downgrades})
        if batch_size != orig:
            print(f"[vram_guard] 可用显存 {self._free_mb():.0f} MiB < 阈值 {self.budget_mb():.0f} MiB"
                  f"（{self.ratio:.0%}）→ batch {orig}→{batch_size}，grad_accum→{grad_accum}", flush=True)
        return batch_size, grad_accum

    def on_oom(self, batch_size: int, grad_accum: int) -> tuple[int, int]:
        """训练中 OOM：清缓存、降一档 batch，由调用方重试当前 epoch。"""
        self.downgrades += 1
        torch.cuda.empty_cache()
        effective = batch_size * grad_accum
        new_batch = max(1, batch_size // 2)
        new_accum = max(1, effective // new_batch)
        print(f"[vram_guard] 捕获 OOM：batch {batch_size}→{new_batch}，grad_accum→{new_accum}，重试本 epoch", flush=True)
        self.history.append({"oom_at_batch": batch_size, "batch": new_batch,
                             "grad_accum": new_accum, "downgrades": self.downgrades})
        return new_batch, new_accum

    def meta(self) -> dict:
        peak = 0.0
        if self.device.type == "cuda":
            peak = torch.cuda.max_memory_allocated() / 1024 ** 2
        return {"vram_ratio": self.ratio, "downgrades": self.downgrades,
                "peak_gpu_mb": round(peak, 1), "history": self.history}
