"""把每个实验实际用的超参落成 configs/*.json，方便复现时不用翻日志。

    python scripts/dump_configs.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import CONFIGS, OUTPUTS, read_json, write_json  # noqa: E402


def main() -> None:
    n = 0
    for ds in ("resume", "cluener", "msra"):
        for method in ("bilstm", "bert", "roberta"):
            p = OUTPUTS / ds / method / "metrics.json"
            if not p.exists():
                continue
            m = read_json(p)
            cfg = {
                "script": f"src.train_{'bilstm' if method == 'bilstm' else 'transformer'}",
                "dataset": ds,
                "method": method,
                "args": m.get("args", {}),
                "result": {
                    "dev_f1_best": m.get("dev_f1_best"),
                    "test_f1": m["test"]["overall"]["f1"],
                    "n_params": m.get("n_params"),
                    "train_seconds": m.get("train_seconds"),
                },
            }
            if method != "bilstm":
                cfg["model_dir"] = f"models/{method}"
            write_json(CONFIGS / f"{ds}_{method}.json", cfg)
            n += 1

    # LLM：把 prompt 版本、抽样、并发等关键参数记下来
    for ds in ("resume", "cluener", "msra"):
        for tag in (f"zero_n{'all' if ds == 'resume' else 300}", f"few_n{'all' if ds == 'resume' else 300}"):
            rp = OUTPUTS / ds / "llm" / f"report_{tag}.json"
            mp = OUTPUTS / ds / "llm" / f"metrics_{tag}.json"
            if not (rp.exists() and mp.exists()):
                continue
            r, m = read_json(rp), read_json(mp)
            write_json(CONFIGS / f"{ds}_llm_{tag}.json", {
                "script": "src.llm_extract",
                "dataset": ds,
                "tag": tag,
                "args": {"model": r["model"], "mode": r["mode"], "n": r["n"], "temperature": 0,
                         "max_tokens": 512, "prompt_version": "v1"},
                "result": {"test_f1": m["test"]["overall"]["f1"], "json_ok_rate": r["json_ok_rate"],
                           "prompt_tokens": r["prompt_tokens"], "completion_tokens": r["completion_tokens"],
                           "cost_cny": r["cost_cny"]},
            })
            n += 1
    print(f"写出 {n} 个 config 到 {CONFIGS}")


if __name__ == "__main__":
    main()
