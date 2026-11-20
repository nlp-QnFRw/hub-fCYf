#region 0. 模块说明
"""
产物契约与落盘的**唯一实现**（规范 §8 跨模块契约）。

为什么要有这一层：
  六条方法路线各写各的 json，字段名一漂移，compare.py 就得写六套分支，
  而且最容易出现"这个 run 有 macro_f1、那个 run 只有 f1"的静默错误。
  统一走 write_run()，字段名与目录结构都由这里保证。

产物结构：
  runs/<run_id>/
    meta.json          run 元信息（谁、什么配置、多久、多少显存、多少钱）
    metrics.json       指标（由 eval/evaluate.py 写入，方法脚本不直接写）
    pred_test.jsonl    在 test_nat_10k 上的逐条预测
    pred_test_bal.jsonl 在 test_bal 上的逐条预测
    train_log.json     每 epoch 的曲线（免训练方法写 []）
    best.pt / adapter/ 权重（大文件，入 .gitignore）

pred 行格式（**字段名固定**，下游 McNemar / 错例分析依赖）：
  {"id": int, "gold": str, "pred": str|null, "raw": str, "status": "ok|abstain|fail", "latency_ms": float}

使用方式：
  from common.contract import new_run_id, write_preds, write_meta, append_index
"""
#endregion

#region 1. 导入依赖
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Sequence

from common.paths import RUNS, RUNS_INDEX, run_dir

#endregion

#region 2. run_id 与落盘
def new_run_id(tier: str, method: str, variant: str) -> str:
    """生成 run_id：{tier}__{method}__{variant}，例 t2__bert__l256-30k-2ep。

    三个字段都只用小写字母、数字、连字符，保证目录名在 Windows / Linux 都合法。
    """
    safe = lambda s: "".join(ch if (ch.isalnum() or ch in "-_") else "-" for ch in s.lower())
    return f"{safe(tier)}__{safe(method)}__{safe(variant)}"


def write_preds(run_id: str, split_name: str, preds: Sequence[dict]) -> Path:
    """把逐条预测写到 runs/<run_id>/pred_<split_name>.jsonl（按 id 升序，便于 diff）。"""
    d = run_dir(run_id)
    path = d / f"pred_{split_name}.jsonl"
    rows = sorted(preds, key=lambda r: r["id"])
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({
                "id": int(r["id"]),
                "gold": r["gold"],
                "pred": r.get("pred"),
                "raw": (r.get("raw") or "")[:200],   # 截断原始输出，避免单行几十 KB
                "status": r.get("status", "ok"),
                "latency_ms": round(float(r.get("latency_ms", 0.0)), 2),
            }, ensure_ascii=False) + "\n")
    return path


def write_meta(run_id: str, **fields) -> Path:
    """写 meta.json。必含字段由调用方保证，这里只补时间戳与路径。"""
    d = run_dir(run_id)
    meta = {"run_id": run_id, **fields, "written_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    path = d / "meta.json"
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_train_log(run_id: str, log: list[dict]) -> Path:
    """写 train_log.json（免训练方法传 []）。"""
    path = run_dir(run_id) / "train_log.json"
    path.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def append_index(run_id: str, extra: dict | None = None) -> None:
    """把 run 追加进 runs/runs.json 索引（append-only，用于"跑过哪些"的一览）。"""
    idx: list[dict] = []
    if RUNS_INDEX.exists():
        try:
            idx = json.loads(RUNS_INDEX.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            idx = []
    if any(r.get("run_id") == run_id for r in idx):
        return  # 重复入队不重复记
    idx.append({"run_id": run_id, "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"), **(extra or {})})
    RUNS_INDEX.write_text(json.dumps(idx, ensure_ascii=False, indent=2), encoding="utf-8")
#endregion

#region 3. 读取（compare / paired_test 共用）
def load_metrics(run_id: str) -> dict | None:
    """读 runs/<run_id>/metrics.json；缺失返回 None（**不抛异常**）。

    汇总脚本的核心约定：任何一个 run 缺失都不能影响其它 run 的展示，
    所以这里必须"有则返回、无则 None"。
    """
    p = RUNS / run_id / "metrics.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def load_preds(run_id: str, split_name: str) -> list[dict] | None:
    """读 pred_<split>.jsonl；缺失返回 None。"""
    p = RUNS / run_id / f"pred_{split_name}.jsonl"
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def list_runs() -> list[str]:
    """列出所有产物目录名（不含 runs.json 这类文件）。"""
    if not RUNS.exists():
        return []
    return sorted(p.name for p in RUNS.iterdir() if p.is_dir() and not p.name.startswith("_"))
#endregion

#region 4. 自检
if __name__ == "__main__":
    rid = new_run_id("t1", "bert", "bert-base-l256-100k-3ep")
    print("run_id 示例：", rid)
    assert rid == "t1__bert__bert-base-l256-100k-3ep"
    print("现有 run：", list_runs()[:5] or "（空）")
#endregion
