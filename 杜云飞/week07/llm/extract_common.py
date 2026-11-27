# -*- coding: utf-8 -*-
"""
extract_common.py — LLM 抽取公共流程（本地模型与 API 共用，保证两边只差一个 generate_fn）

口径纪律（与课程一致）：
  - 所有 LLM 方法在**同一批 n=200 句**上评测（data/processed/dev_n200.jsonl 的句子集合），
    与判别式方法的 n200 投影完全同批；
  - 预测失败的句子**保留** events:[]（删了分母变小、F1 虚高）；
  - 工程指标（JSON 合法率/修复数/幻觉数/token 量/延迟）与三级 F1 一起进 meta。

断点续跑（11-26 教训）：上一版只在整轮结束后落盘，0.6B few-shot 跑到 128/200 被
时间截止杀掉后中间结果全部丢失，30 分钟的 GPU 时间白烧。现在每批解析完立刻追加写
_partial_<run>.jsonl（原子到"句"粒度），重启后按前缀校验自动跳过已完成的句子。
"""
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "common"))
sys.path.insert(0, str(ROOT / "llm"))

from eval_utils import evaluate_predictions  # noqa: E402
from prompt import parse_model_output  # noqa: E402

PRED_DIR = ROOT / "outputs" / "disc"


def load_schema() -> dict:
    with open(ROOT / "data" / "processed" / "event_schema.json", encoding="utf-8") as f:
        return {k: v for k, v in json.load(f).items()}


def load_dev_sentences(n: int = 200) -> list[dict]:
    """读 dev_n200（按句去重后的固定 200 句），每句聚合全部 gold 事件。"""
    path = ROOT / "data" / "processed" / "dev_n200.jsonl"
    sents: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            g = sents.setdefault(r["id"], {"id": r["id"], "text": r["text"], "events": []})
            g["events"].append(r["event"])
    rows = list(sents.values())
    assert len(rows) == n, f"dev_n200 应为 {n} 句，实际 {len(rows)}"
    return rows


def _partial_paths(out_name: str) -> tuple[Path, Path]:
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    return (PRED_DIR / f"_partial_{out_name}.jsonl",
            PRED_DIR / f"_partial_{out_name}.stats.json")


def run_extract(schema: dict, sentences: list[dict], generate_fn, out_name: str,
                meta: dict, batch_size: int = 8, resume: bool = True) -> dict:
    """
    generate_fn(batch_texts: list[str]) -> list[str]（原始模型输出，与输入等长）
    返回评测结果 dict；预测与指标分别落盘 outputs/disc/ 与 outputs/metrics/disc/。
    """
    part, statp = _partial_paths(out_name)
    preds: list[dict] = []
    stats: Counter = Counter()
    start = 0

    if resume and part.exists():
        rows = [json.loads(x) for x in part.read_text(encoding="utf-8").splitlines() if x.strip()]
        same = (len(rows) <= len(sentences) and
                all(str(r.get("id")) == str(sentences[i]["id"]) for i, r in enumerate(rows)))
        if same and rows:
            preds = rows
            start = len(rows)
            if statp.exists():
                try:
                    stats = Counter(json.loads(statp.read_text(encoding="utf-8")))
                except json.JSONDecodeError:
                    pass
            print(f"[resume] {out_name}：已完成 {start}/{len(sentences)} 句，续跑", flush=True)
        else:
            print(f"[resume] {out_name}：断点与本次句子集不匹配，丢弃重跑", flush=True)
            part.unlink(missing_ok=True)

    t0 = time.perf_counter()
    for i in range(start, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        try:
            raws = generate_fn([s["text"] for s in batch])
        except Exception as e:  # 一批失败不能拖垮整轮：记空输出继续
            print(f"[warn] 第 {i} 批生成失败（{e}），该批按空输出处理", flush=True)
            raws = [""] * len(batch)
        new_rows = []
        for s, raw in zip(batch, raws):
            events, info = parse_model_output(raw or "", s["text"], schema)
            for k, v in info.items():
                if isinstance(v, bool):
                    stats[k] += int(v)
                else:
                    stats[k] += v
            # 解析失败的行保留 events:[]（评测纪律：不删分母）
            new_rows.append({"uid": str(s["id"]), "id": s["id"], "events": events})
        preds.extend(new_rows)
        # 逐批落盘：被杀掉也不丢已完成的句子
        with open(part, "a", encoding="utf-8") as f:
            for p in new_rows:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        statp.write_text(json.dumps(dict(stats), ensure_ascii=False), encoding="utf-8")
        if (i // batch_size) % 5 == 0:
            done = min(i + batch_size, len(sentences))
            el = time.perf_counter() - t0
            print(f"  抽取进度 {done}/{len(sentences)}（本轮 {el:.0f}s，"
                  f"json_ok={stats['json_ok']}）", flush=True)

    n = len(sentences)
    runtime = time.perf_counter() - t0
    meta.update({
        "group": "llm",
        "n_sentences": n,
        "resumed_from": start,
        "runtime_s": round(runtime, 1),
        "sec_per_sentence": round(runtime / max(n - start, 1), 3),
        "parse_stats": dict(stats),
        "json_valid_rate": round(stats["json_ok"] / n, 4),
    })
    res = evaluate_predictions(out_name, preds, meta=meta,
                               gold_path=ROOT / "data" / "processed" / "dev_n200.jsonl")
    part.unlink(missing_ok=True)
    statp.unlink(missing_ok=True)
    return res
