# -*- coding: utf-8 -*-
"""
extract_api.py — C 组：DeepSeek API zero/few-shot 抽取 + 可复现性验证

模型标识：本账号 /models 返回 deepseek-flash 与 deepseek-v4-pro，作业要求 flash 档，
故用 deepseek-flash（课程曾用 deepseek-v4-flash，该标识已下线，flash 档现行为 deepseek-flash）。

工程点：
  - API key 读取顺序：环境变量 DEEPSEEK_API_KEY/LLM_API_KEY → Windows 用户级注册表
    （本机 key 设在用户环境变量里，但新开的 shell 不一定继承得到，所以做注册表兜底）；
  - temperature=0 + 同批重跑（--repeat 2）验证"服务端非确定性"（课程坑 #36：
    MoE 路由/并行归约会让重跑结果在 ±0.02 内波动）；
  - 并发 8 线程压测延迟；逐条记录 usage（prompt/completion tokens）用于成本折算。

用法：
  python llm/extract_api.py [--few_shot] [--repeat 2] [--n 200] [--model deepseek-flash]
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "common"))
sys.path.insert(0, str(ROOT / "llm"))

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-flash"


def get_api_key() -> str:
    key = os.getenv("LLM_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    if key:
        return key
    try:  # Windows 用户级环境变量兜底（新进程未继承时的救星）
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
            val, _ = winreg.QueryValueEx(k, "DEEPSEEK_API_KEY")
            return val
    except Exception as e:
        raise EnvironmentError(
            "找不到 DeepSeek API key：请 set DEEPSEEK_API_KEY=sk-xxx 或设置用户环境变量") from e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.getenv("LLM_MODEL", DEFAULT_MODEL))
    ap.add_argument("--base_url", default=os.getenv("LLM_BASE_URL", DEFAULT_BASE_URL))
    ap.add_argument("--few_shot", action="store_true")
    ap.add_argument("--repeat", type=int, default=1, help="同批重跑次数（可复现性验证）")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max_tokens", type=int, default=800, help="预算控制：多事件 JSON 一般 <600 token")
    ap.add_argument("--n_examples", type=int, default=9,
                    help="few-shot 用前 N 个示例（API token 预算紧时可减到 3；本地模型无此顾虑）")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--run_name", default="")
    ap.add_argument("--thinking", choices=["disabled", "enabled"], default="disabled",
                    help="disabled=关思考链（课程坑 #5 的 API 版：开着会把 max_tokens 烧在"
                         " reasoning_content 上、正文为空；实测关闭后延迟降 4 倍、空回复消失）")
    args = ap.parse_args()

    from openai import OpenAI
    from prompt import build_messages
    from extract_common import load_dev_sentences, load_schema, run_extract

    schema = load_schema()
    sentences = load_dev_sentences(args.n)
    client = OpenAI(api_key=get_api_key(), base_url=args.base_url, timeout=120)
    print(f"API：{args.base_url}  model={args.model}  few_shot={args.few_shot}  n={args.n}")

    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    lock_count = {"done": 0}

    def call_one(text: str) -> str:
        msgs = build_messages(text, schema, args.few_shot)
        if args.few_shot and args.n_examples < 9:
            # 只用前 N 个示例（token 预算控制）：system 之后替换为截断的示例对
            msgs = [msgs[0]] + msgs[1:1 + 2 * args.n_examples] + msgs[-1:]
        for attempt in range(3):  # 限流/网络抖动重试，尊重 backoff 信号
            try:
                r = client.chat.completions.create(
                    model=args.model, messages=msgs,
                    temperature=args.temperature, max_tokens=args.max_tokens,
                    extra_body={"thinking": {"type": args.thinking}})
                u = r.usage
                usage["prompt_tokens"] += getattr(u, "prompt_tokens", 0) or 0
                usage["completion_tokens"] += getattr(u, "completion_tokens", 0) or 0
                lock_count["done"] += 1
                if lock_count["done"] % 25 == 0:
                    print(f"  API 进度 {lock_count['done']}/{len(sentences)}", flush=True)
                return r.choices[0].message.content or ""
            except Exception as e:
                wait = 2 ** attempt * 5
                print(f"[warn] API 调用失败（{type(e).__name__}: {e}），{wait}s 后重试", flush=True)
                time.sleep(wait)
        return ""

    def generate_fn(batch_texts):
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            return list(ex.map(call_one, batch_texts))

    for rep in range(1, args.repeat + 1):
        lock_count["done"] = 0
        run_name = args.run_name or (
            f"api_{args.model}_{'few' if args.few_shot else 'zero'}"
            + ("_thinkon" if args.thinking == "enabled" else "")
            + (f"_rep{rep}" if args.repeat > 1 else ""))
        meta = {
            "display_name": f"API {args.model} {'few-shot' if args.few_shot else 'zero-shot'}"
                            + (f"（第 {rep} 次重跑）" if args.repeat > 1 else ""),
            "model": args.model, "base_url": args.base_url,
            "few_shot": bool(args.few_shot), "n_few_shot_examples": args.n_examples if args.few_shot else 0,
            "temperature": args.temperature, "max_tokens": args.max_tokens,
            "repeat_index": rep,
            "note": "temperature=0；重跑差异即服务端非确定性（课程坑 #36）",
        }
        run_extract(schema, sentences, generate_fn, run_name, meta, batch_size=args.workers)
        # 把 token 用量写回指标 JSON
        mf = ROOT / "outputs" / "metrics" / "disc" / f"{run_name}.json"
        data = json.loads(mf.read_text(encoding="utf-8"))
        data["meta"].update(dict(usage))
        mf.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[repeat {rep}] token 合计 prompt={usage['prompt_tokens']} "
              f"completion={usage['completion_tokens']}")
        usage = {"prompt_tokens": 0, "completion_tokens": 0}


if __name__ == "__main__":
    main()
