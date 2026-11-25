"""DeepSeek API 抽取：零样本 / 三样本提示，输出与本地模型完全同格式的 pred.jsonl。

设计要点
--------
1. 不让模型输出字符下标（生成式模型数下标极不可靠），只让它输出「原文片段」，
   程序用 text.find() 回填 offset；找不到的片段记为幻觉并丢弃。
2. 解析器容错：去 markdown 围栏 → 括号配平截取 → 去尾逗号 → JSON 修复；
   解析失败的样本保留空结果并计数（不删分母）。
3. 逐条落盘 raw_*.jsonl，可断点续跑；并发 8 + 指数退避重试。
4. key 从环境变量读（DEEPSEEK_KEY / DEEPSEEK_API_KEY），不写进任何文件。

用法：
    python -m src.llm_extract --dataset resume --mode zero
    python -m src.llm_extract --dataset cluener --mode few --n 300
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .common import OUTPUTS, Timer, env_key, get_logger, human_time, read_jsonl, write_json, write_jsonl
from .data import load_labels, load_splits
from .metrics import evaluate, format_table

LABEL_DEFS = {
    "resume": {
        "NAME": "姓名，只含姓名本身，不含“先生/女士”或括号性别",
        "TEL": "电话号码，含区号/连字符/+86/括号，不含“电话：”",
        "EMAIL": "完整邮箱地址，含 @ 与域名，大小写按原文",
        "EDU": "学历或学位词：大专/本科/硕士/博士/研究生",
        "SCHOOL": "教育经历中的学校名",
        "MAJOR": "专业名称",
        "ORG": "工作经历中的公司/单位名",
        "POS": "职位名称（工作经历与求职意向都算）",
        "TIME": "时间区间或时间点，起止整体作为一个实体，含连接符",
        "SALARY": "薪资，含单位与“/月”“/年”等后缀",
    },
    "cluener": {
        "address": "地址（省/市/区/街道等）",
        "book": "书名",
        "company": "公司名",
        "game": "游戏名",
        "government": "政府机构名",
        "movie": "电影名",
        "name": "人名",
        "organization": "组织机构名（非公司、非政府）",
        "position": "职位名",
        "scene": "景点名",
    },
    "msra": {
        "PER": "人名",
        "LOC": "地名（国家/城市/省份/山川等）",
        "ORG": "组织机构名（公司/学校/政府/媒体等）",
    },
}

SYSTEM_TMPL = """你是中文信息抽取专家。从用户给出的文本中抽取所有属于下列类型的实体。

## 实体类型
{label_lines}

## 抽取规则
1. 实体必须是原文中的**连续片段**，逐字照抄，不改写、不翻译、不补全。
2. 不要标字段名、冒号、竖线、空格等分隔符，只标实体本身。
3. 同一实体在文中出现多次就多次输出；实在没有就输出空列表。
4. 不要输出字符下标，程序会自己在原文里对齐。
5. 只输出一个 JSON 对象，不要解释、不要 markdown 围栏。

## 输出格式
{{"entities": [{{"text": "实体原文", "label": "类型"}}]}}
没有实体时输出：{{"entities": []}}"""


def build_system_prompt(dataset: str) -> str:
    defs = LABEL_DEFS[dataset]
    lines = "\n".join(f"- {lab}：{desc}" for lab, desc in defs.items())
    return SYSTEM_TMPL.format(label_lines=lines)


def pick_fewshot(rows: list[dict], k: int = 3, min_chars: int = 12) -> list[dict]:
    """few-shot 示例只从 train 里挑：先按「标签组合不同」取，再按长度补齐；绝不碰 dev/test。"""
    cand = sorted([r for r in rows if r["entities"] and len(r["text"]) >= min_chars],
                  key=lambda r: (len(r["text"]), r["id"]))
    picked, sigs = [], set()
    for r in cand:
        sig = tuple(sorted({e["label"] for e in r["entities"]}))
        if sig in sigs:
            continue
        sigs.add(sig)
        picked.append(r)
        if len(picked) >= k:
            return picked
    # 标签组合不够多（比如简历每条都含 10 类）时，按长度分布补到 k 条
    for r in cand:
        if len(picked) >= k:
            break
        if r["id"] not in {x["id"] for x in picked}:
            picked.append(r)
    return picked[:k]


def to_target(row: dict) -> str:
    return json.dumps({"entities": [{"text": row["text"][e["start"]:e["end"]], "label": e["label"]}
                                    for e in row["entities"]]}, ensure_ascii=False)


def _strip_fences(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    return s.strip()


def _extract_json(s: str) -> str | None:
    """括号配平截取第一个完整 JSON 对象（正则贪婪/非贪婪都会在嵌套时出错）。"""
    # @doc *** 括号配平要区分字符串内与转义，否则实体里的 } 会提前截断
    # 只数花括号会在引号里的 } 提前收工，于是整条解析失败、被记成 parse_fail
    start = s.find("{")
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start:i + 1]
    return None


def parse_output(raw: str, text: str, labels: list[str]) -> tuple[list[dict], dict]:
    """容错解析 → (entities, 诊断)。"""
    diag = {"json_ok": False, "repaired": False, "noise_stripped": False,
            "not_in_text": 0, "label_invalid": 0}
    if not raw or not raw.strip():
        return [], diag
    s = _strip_fences(raw)
    if s != raw.strip():
        diag["noise_stripped"] = True
    body = _extract_json(s)
    if body is None:
        return [], diag
    try:
        obj = json.loads(body)
    except json.JSONDecodeError:
        fixed = re.sub(r",\s*([}\]])", r"\1", body)          # 去尾逗号
        fixed = re.sub(r"'([^']*)'", r'"\1"', fixed)          # 单引号 → 双引号
        try:
            obj = json.loads(fixed)
            diag["repaired"] = True
        except json.JSONDecodeError:
            return [], diag
    diag["json_ok"] = True

    items = obj.get("entities", obj if isinstance(obj, list) else [])
    if isinstance(items, dict):
        items = [items]
    ents, cursor = [], 0
    for it in items:
        if not isinstance(it, dict):
            continue
        txt = str(it.get("text", it.get("entity", ""))).strip()
        lab = str(it.get("label", it.get("type", ""))).strip()
        if lab not in labels:
            diag["label_invalid"] += 1
            continue
        if not txt:
            continue
        # @doc ** 从 cursor 往后找，同一实体的第二处才不会被去重吃掉
        # 每次 find 都从头搜会把两处对齐到同一个位置，第二条被去重吃掉、看起来像漏标
        pos = text.find(txt, cursor)
        if pos < 0:
            pos = text.find(txt)
        if pos < 0:
            diag["not_in_text"] += 1
            continue
        ents.append({"start": pos, "end": pos + len(txt), "label": lab})
        cursor = pos + len(txt)
    # 去重 + 排序
    seen, uniq = set(), []
    for e in sorted(ents, key=lambda x: (x["start"], x["end"])):
        key = (e["start"], e["end"], e["label"])
        if key not in seen:
            seen.add(key)
            uniq.append(e)
    return uniq, diag


def call_api(base_url: str, key: str, model: str, messages: list[dict],
             max_tokens: int = 512, timeout: int = 90) -> tuple[str, dict]:
    payload = json.dumps({"model": model, "messages": messages, "temperature": 0,
                          "max_tokens": max_tokens, "stream": False}).encode()
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions", data=payload,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.load(r)
    latency = time.perf_counter() - t0
    content = data["choices"][0]["message"].get("content") or ""
    usage = data.get("usage", {}) or {}
    return content, {"latency": latency, "prompt_tokens": usage.get("prompt_tokens", 0),
                     "completion_tokens": usage.get("completion_tokens", 0)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["resume", "cluener", "msra"])
    ap.add_argument("--mode", required=True, choices=["zero", "few"])
    ap.add_argument("--n", type=int, default=0, help="抽样条数，0 表示全测试集")
    ap.add_argument("--model", default="deepseek-chat")
    ap.add_argument("--base_url", default="https://api.deepseek.com")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="冒烟：只跑前 N 条")
    ap.add_argument("--mock", action="store_true", help="不调 API，用空结果打通链路")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    tag = args.tag or f"{args.mode}_n{args.n or 'all'}"
    out_dir = OUTPUTS / args.dataset / "llm"
    out_dir.mkdir(parents=True, exist_ok=True)
    log = get_logger(f"llm.{args.dataset}.{args.mode}", out_dir / f"{tag}.log")

    splits = load_splits(args.dataset)
    labels = load_labels(args.dataset)
    test = splits["test"]

    # 统一抽样：先按 id 排序再 shuffle，和文件行序无关；ids 落盘供其他方法对齐
    if args.n and args.n < len(test):
        ordered = sorted(test, key=lambda r: r["id"])
        random.Random(args.seed).shuffle(ordered)
        test = ordered[:args.n]
    if args.limit:
        test = test[:args.limit]
    write_json(out_dir / f"eval_ids_{tag}.json", [r["id"] for r in test])
    log.info("评测样本 %d 条（dataset=%s, mode=%s）", len(test), args.dataset, args.mode)

    system = build_system_prompt(args.dataset)
    few = pick_fewshot(splits["train"]) if args.mode == "few" else []
    if few:
        log.info("few-shot 示例 %d 条: %s", len(few), [r["id"] for r in few])

    def make_messages(row: dict) -> list[dict]:
        msgs = [{"role": "system", "content": system}]
        for ex in few:
            msgs.append({"role": "user", "content": "文本：\n" + ex["text"]})
            msgs.append({"role": "assistant", "content": to_target(ex)})
        msgs.append({"role": "user", "content": "文本：\n" + row["text"]})
        return msgs

    raw_path = out_dir / f"raw_{tag}.jsonl"
    done = set()
    if raw_path.exists():
        done = {r["id"] for r in read_jsonl(raw_path)}
        log.info("断点续跑：已有 %d 条", len(done))

    todo = [r for r in test if r["id"] not in done]
    key = None if args.mock else env_key("DEEPSEEK_KEY", "DEEPSEEK_API_KEY")
    if not args.mock and not key:
        log.error("没有找到 DEEPSEEK_KEY / DEEPSEEK_API_KEY 环境变量")
        raise SystemExit(2)

    results: list[dict] = []
    lock_done = done.copy()

    def work(row: dict) -> dict:
        last_err = ""
        for attempt in range(4):
            try:
                if args.mock:
                    content, usage = '{"entities": []}', {"latency": 0.0, "prompt_tokens": 0,
                                                           "completion_tokens": 0}
                else:
                    content, usage = call_api(args.base_url, key, args.model, make_messages(row),
                                              max_tokens=args.max_tokens)
                ents, diag = parse_output(content, row["text"], labels)
                return {"id": row["id"], "raw": content, "entities": ents, **usage, **diag}
            except urllib.error.HTTPError as e:
                last_err = f"HTTP {e.code}: {e.read()[:200]!r}"
                time.sleep(2 ** attempt)
            except Exception as e:  # noqa: BLE001
                last_err = f"{type(e).__name__}: {str(e)[:200]}"
                time.sleep(2 ** attempt)
        log.warning("失败 %s: %s", row["id"], last_err)
        return {"id": row["id"], "raw": "", "entities": [], "latency": 0.0, "prompt_tokens": 0,
                "completion_tokens": 0, "json_ok": False, "repaired": False, "noise_stripped": False,
                "not_in_text": 0, "label_invalid": 0, "error": last_err}

    with Timer() as timer:
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futs = {ex.submit(work, r): r for r in todo}
            for i, fut in enumerate(as_completed(futs), 1):
                results.append(fut.result())
                if i % 20 == 0 or i == len(todo):
                    log.info("进度 %d/%d", i, len(todo))
    if results:
        merged = {r["id"]: r for r in (read_jsonl(raw_path) if raw_path.exists() else [])}
        merged.update({r["id"]: r for r in results})
        write_jsonl(raw_path, [merged[k] for k in sorted(merged)])
    log.info("完成 %d 条，用时 %s", len(results), human_time(timer.seconds))

    # 汇总：解析成功、幻觉、token、延迟
    all_raw = {r["id"]: r for r in read_jsonl(raw_path)} if raw_path.exists() else {}
    pred_rows, n_json_ok, n_rep, n_noise, n_oot, n_badlab, toks_in, toks_out, lat = [], 0, 0, 0, 0, 0, 0, 0, []
    for row in test:
        r = all_raw.get(row["id"], {})
        n_json_ok += int(r.get("json_ok", False))
        n_rep += int(r.get("repaired", False))
        n_noise += int(r.get("noise_stripped", False))
        n_oot += int(r.get("not_in_text", 0))
        n_badlab += int(r.get("label_invalid", 0))
        toks_in += int(r.get("prompt_tokens", 0))
        toks_out += int(r.get("completion_tokens", 0))
        if r.get("latency"):
            lat.append(float(r["latency"]))
        pred_rows.append({"id": row["id"], "text": row["text"], "entities": r.get("entities", [])})
    write_jsonl(out_dir / f"pred_{tag}.jsonl", pred_rows)

    n = len(test)
    price_in = float(os.environ.get("LLM_PRICE_IN_PER_M", "2"))
    price_out = float(os.environ.get("LLM_PRICE_OUT_PER_M", "8"))
    report = {
        "dataset": args.dataset, "mode": args.mode, "model": args.model, "tag": tag, "n": n,
        "json_ok_rate": n_json_ok / max(1, n),
        "json_ok": n_json_ok, "repaired": n_rep, "noise_stripped": n_noise,
        "hallucinated_spans": n_oot, "label_invalid": n_badlab,
        "prompt_tokens": toks_in, "completion_tokens": toks_out,
        "avg_latency_s": sum(lat) / max(1, len(lat)),
        "p50_latency_s": sorted(lat)[len(lat) // 2] if lat else 0.0,
        "elapsed_s": round(timer.seconds, 1),
        "cost_cny": toks_in / 1e6 * price_in + toks_out / 1e6 * price_out,
        "price_in_per_m": price_in, "price_out_per_m": price_out,
    }
    write_json(out_dir / f"report_{tag}.json", report)

    result = evaluate(splits["test"], pred_rows, labels, subset_ids={r["id"] for r in test})
    hard = [r for r in splits["test"] if r.get("meta", {}).get("hard")]
    if hard:
        hard_ids = {r["id"] for r in hard} & {r["id"] for r in test}
        if hard_ids:
            result["hard_subset"] = evaluate(splits["test"], pred_rows, labels, subset_ids=hard_ids)
    metrics = {"dataset": args.dataset, "method": f"llm_{args.mode}", "tag": tag,
               "test": result, "llm": report,
               "n_params": 0, "train_seconds": 0,
               "ms_per_sentence": round(1000 * report["avg_latency_s"], 2)}
    write_json(out_dir / f"metrics_{tag}.json", metrics)
    log.info("JSON 合法率 %.1f%% | 幻觉片段 %d | 平均延迟 %.2fs | 费用 ¥%.3f",
             100 * report["json_ok_rate"], n_oot, report["avg_latency_s"], report["cost_cny"])
    log.info("测试集：\n%s", format_table(result))


if __name__ == "__main__":
    main()
