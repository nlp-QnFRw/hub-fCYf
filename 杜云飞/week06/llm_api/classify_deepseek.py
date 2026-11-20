#region 0. 模块说明
"""
L3-b：DeepSeek-V4.1-Flash 云端 zero/few-shot 分类。

它是"上限参照"——如果云端强模型 + few-shot 都赢不过微调后的 BERT，
那本任务的瓶颈就不在模型能力（很可能在标注体系），这个结论比多涨 1pp 更有价值。

**必须知道的坑（前项目实测，直接继承）**：
  `deepseek-flash` 是**推理模型**，默认会先吐 `reasoning_content`，`content` 为空。
  如果不显式关闭思考，`max_tokens=8` 时 100% 拿不到答案，看起来像"模型不会分类"。
  关闭方式：请求体加 `{"thinking": {"type": "disabled"}}`；
  开启方式：`{"reasoning_effort": "low" | "high"}`（参数名不是 "effort"，写错会 400）。

工程约束：
  1. 密钥只走环境变量 `DEEPSEEK_API_KEY`，**绝不写进代码/argparse 默认值/日志**；
     本机该变量是用户级环境变量，当前 shell 可能没继承，因此加了注册表兜底；
  2. 成本护栏：`--estimate` 先发 1 条真实请求，按返回的 usage 推算总 token 与金额；
  3. **响应磁盘缓存**：同一 (模型, thinking, prompt) 重跑不花钱；
  4. 并发 12 + 指数退避重试，单条失败返回 None 而不是让整批中断。

使用方式：
  python llm_api/classify_deepseek.py --estimate --n 50
  python llm_api/classify_deepseek.py --shots 0 --thinking off --n 1000 --splits test_bal
  python llm_api/classify_deepseek.py --shots 2 --thinking high --splits test_bal
"""
#endregion

#region 1. 导入依赖
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.contract import append_index, new_run_id, write_meta, write_preds, write_train_log  # noqa: E402
from common.data import load_split  # noqa: E402
from common.paths import CACHE  # noqa: E402
from common.prompt import CLASS_LIST, SYSTEM_PROMPT, build_messages, parse_label  # noqa: E402
from common.text import build_text  # noqa: E402

#endregion

#region 2. 全局配置
BASE_URL: str = "https://api.deepseek.com/v1"
MODEL_ID: str = "deepseek-flash"  # 即 DeepSeek-V4.1-Flash
CONCURRENCY: int = 12
CACHE_DIR: Path = CACHE / "deepseek"
# 参考价（元 / 百万 token），仅用于成本估算，实际以账单为准
PRICE_IN: float = 0.5
PRICE_OUT: float = 2.0
#endregion

#region 3. 函数：密钥读取
def get_api_key() -> str:
    """按 环境变量 → Windows 用户级注册表 的顺序取密钥。

    为什么要读注册表：用户把密钥设在"用户环境变量"里，已开的终端不会继承，
    `os.environ` 取不到。直接读 HKCU\\Environment 可以免去"重启终端"的来回。
    """
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if key:
        return key
    if os.name == "nt":
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
                key = str(winreg.QueryValueEx(k, "DEEPSEEK_API_KEY")[0]).strip()
                if key:
                    return key
        except OSError:
            pass
    raise SystemExit("找不到 DEEPSEEK_API_KEY：请设置用户环境变量后重开终端，或临时 export。")
#endregion

#region 4. 函数：单次调用（带缓存与重试）
def _cache_path(payload_key: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{payload_key}.json"


def call_once(messages: list[dict], thinking: str, api_key: str,
              timeout: int = 90) -> dict:
    """发一次 chat/completions，返回 {content, reasoning, usage, latency, error}。

    **失败也返回 dict**（不抛异常）：API 偶发 429/5xx 是常态，不该让整批 4000 条中断。
    """
    body = {
        "model": MODEL_ID,
        "messages": messages,
        "temperature": 0.0,  # 分类任务要确定性
        "stream": False,
    }
    if thinking == "off":
        body["thinking"] = {"type": "disabled"}  # 关键：不关思考，max_tokens=8 时 content 为空
        body["max_tokens"] = 16
    else:
        body["reasoning_effort"] = thinking  # low / high；注意参数名不是 "effort"
        body["max_tokens"] = 2048

    key = hashlib.md5(json.dumps(body, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    cp = _cache_path(key)
    if cp.exists():
        try:
            cached = json.loads(cp.read_text(encoding="utf-8"))
            cached["cached"] = True
            return cached
        except json.JSONDecodeError:
            pass

    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    last_err = ""
    for attempt in range(3):
        t0 = time.time()
        try:
            req = urllib.request.Request(
                f"{BASE_URL}/chat/completions", data=data,
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {api_key}"}, method="POST")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                obj = json.loads(resp.read().decode("utf-8"))
            msg = obj["choices"][0]["message"]
            out = {"content": msg.get("content") or "",
                   "reasoning": msg.get("reasoning_content") or "",
                   "usage": obj.get("usage", {}),
                   "latency": round(time.time() - t0, 3), "error": ""}
            cp.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
            return out
        except urllib.error.HTTPError as e:
            last_err = f"HTTP {e.code}: {e.read()[:200].decode('utf-8', 'ignore')}"
            # 402 = 余额不足，重试一万次也不会成功；必须立刻中止而不是继续把 4000 条写成"拒答"。
            # 实测踩过：余额耗尽后 3126 条被记成 abstain，macro-F1 从 0.82 掉到 0.25，
            # 表面看像"few-shot 崩了"，实际是账单问题。
            if e.code == 402:
                raise SystemExit(
                    f"API 余额/配额不足（HTTP 402），已中止。请充值后重跑。\n{last_err}")
            if e.code in (401, 403):
                raise SystemExit(f"鉴权失败：{last_err}")  # 这类错误重试没用
            if e.code == 429:
                time.sleep(3 * (2 ** attempt))  # 限流要多等，别和普通 5xx 用同一档退避
        except (urllib.error.URLError, TimeoutError) as e:
            last_err = f"{type(e).__name__}: {e}"
        time.sleep(2 ** attempt)  # 指数退避
    return {"content": "", "reasoning": "", "usage": {}, "latency": 0.0, "error": last_err}
#endregion

#region 5. 函数：渲染 prompt（与本地 LLM 同一套）
def render_messages(rec: dict, shots: int, max_chars: int, truncation: str) -> list[dict]:
    """复用 common.prompt.build_messages，保证本地与云端的 prompt 完全一致。"""
    return build_messages(rec, shots=shots, max_chars=max_chars, truncation=truncation)
#endregion

#region 6. 主流程
def main() -> None:
    ap = argparse.ArgumentParser(description="DeepSeek 云端 zero/few-shot 分类")
    ap.add_argument("--shots", type=int, default=0)
    ap.add_argument("--thinking", default="off", choices=["off", "low", "high"])
    ap.add_argument("--splits", nargs="+", default=["test_bal"])
    ap.add_argument("--n", type=int, default=1000, help="每个 split 的条数（默认 1000，控成本）")
    ap.add_argument("--max_chars", type=int, default=600)
    ap.add_argument("--truncation", default="head_tail")
    ap.add_argument("--parse", default="lenient", choices=["strict", "lenient"])
    ap.add_argument("--concurrency", type=int, default=CONCURRENCY)
    ap.add_argument("--max_errors", type=int, default=50,
                    help="单次运行允许的接口错误条数上限，超过即中止（防止把账单/网络故障写成'拒答'）")
    ap.add_argument("--estimate", action="store_true", help="只发 1 条请求，试算总成本")
    ap.add_argument("--price_in", type=float, default=PRICE_IN, help="元/百万输入 token")
    ap.add_argument("--price_out", type=float, default=PRICE_OUT, help="元/百万输出 token")
    ap.add_argument("--run_id", default=None)
    ap.add_argument("--variant", default=None)
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()

    if args.demo:
        args.n, args.splits = 20, ["test_bal"]
        print("[Demo 模式] 20 条，产物不可用于对比")

    api_key = get_api_key()
    variant = args.variant or f"ds-{args.thinking}-{args.shots}shot-{args.n}"
    run_id = args.run_id or new_run_id("t1" if args.shots == 0 else "t3", "llm-api", variant)

    recs = load_split(args.splits[0], limit=args.n)
    msgs = render_messages(recs[0], args.shots, args.max_chars, args.truncation)

    # ---- 成本试算 ----
    if args.estimate:
        out = call_once(msgs, args.thinking, api_key)
        u = out.get("usage", {})
        pin = u.get("prompt_tokens", len(json.dumps(msgs, ensure_ascii=False)))
        pout = u.get("completion_tokens", 4)
        total = len(recs)
        cost = (pin * total / 1e6 * args.price_in) + (pout * total / 1e6 * args.price_out)
        print(f"单条：输入 {pin} tok / 输出 {pout} tok | 单条 {out['latency']}s")
        print(f"按 {total} 条估算：输入 {pin * total / 1e6:.3f}M / 输出 {pout * total / 1e6:.3f}M "
              f"→ **约 {cost:.2f} 元**（参考价 {args.price_in}/{args.price_out} 元每百万）")
        print(f"缓存目录 {CACHE_DIR}：同配置重跑 0 成本")
        return

    t_all = time.time()
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0}
    for split in args.splits:
        recs = load_split(split, limit=args.n)
        print(f"\n[{split}] {len(recs)} 条，并发 {args.concurrency}，thinking={args.thinking}。"
              f"预计 {len(recs) * (1.6 if args.thinking == 'off' else 6) / args.concurrency / 60:.1f} 分钟")

        def work(rec: dict) -> dict:
            msgs = render_messages(rec, args.shots, args.max_chars, args.truncation)
            out = call_once(msgs, args.thinking, api_key)
            pred, status = parse_label(out["content"], args.parse)
            if out["error"]:
                status = "fail"
            return {"id": rec["id"], "gold": rec["label"], "pred": pred,
                    "raw": (out["content"] or out["error"]).replace("\n", " "),
                    "status": status if pred else ("fail" if out["error"] else "abstain"),
                    "latency_ms": out["latency"] * 1000, "_usage": out["usage"]}

        preds: list[dict] = []
        done = 0
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futs = {ex.submit(work, r): r for r in recs}
            for fut in as_completed(futs):
                p = fut.result()
                for k in ("prompt_tokens", "completion_tokens"):
                    total_usage[k] += p["_usage"].get(k, 0)
                p.pop("_usage")
                preds.append(p)
                done += 1
                if done % 100 == 0 or done == len(recs):
                    acc = sum(1 for x in preds if x["pred"] == x["gold"]) / len(preds)
                    n_err = sum(1 for x in preds if x["status"] == "fail")
                    print(f"  {done}/{len(recs)} acc(含失败)={acc:.4f} 错误 {n_err} "
                          f"{done / max(1e-6, time.time() - t0):.1f} 条/s", flush=True)
                    # 护栏：错误率异常高说明是配额/网络故障，继续跑只会把错误写成"拒答"
                    if n_err > args.max_errors:
                        write_preds(run_id, split, preds)
                        raise SystemExit(
                            f"错误条数 {n_err} > --max_errors {args.max_errors}，已中止，"
                            f"本次结果**不可用于对比**（已保留 {run_id} 供排查）。")

        write_preds(run_id, split, preds)
        acc = sum(1 for p in preds if p["pred"] == p["gold"]) / max(1, len(preds))
        n_fail = sum(1 for p in preds if p["status"] != "ok")
        print(f"  {split}: acc={acc:.4f} 失败/拒答 {n_fail} ({n_fail / max(1, len(preds)) * 100:.1f}%)")

    cost = (total_usage["prompt_tokens"] / 1e6 * args.price_in
            + total_usage["completion_tokens"] / 1e6 * args.price_out)
    print(f"\n本次实际用量：输入 {total_usage['prompt_tokens'] / 1e6:.3f}M / "
          f"输出 {total_usage['completion_tokens'] / 1e6:.3f}M → 约 {cost:.2f} 元")

    write_meta(run_id, tier="t1" if args.shots == 0 else "t3", method="llm-api",
               backbone=MODEL_ID, n_train=0, seed=42,
               hyperparams={"shots": args.shots, "thinking": args.thinking,
                            "max_chars": args.max_chars, "truncation": args.truncation,
                            "parse": args.parse, "n_per_split": args.n,
                            "concurrency": args.concurrency},
               usage=total_usage, cost_usd=round(cost, 4),
               elapsed_s=round(time.time() - t_all, 1), device="api", peak_mem_gb=0.0)
    write_train_log(run_id, [])
    append_index(run_id, {"method": "llm-api", "cost_cny": round(cost, 3)})

    print(f"\nrun_id = {run_id}")
    print(f"下一步：python eval/evaluate.py --run {run_id}")


if __name__ == "__main__":
    main()
#endregion
