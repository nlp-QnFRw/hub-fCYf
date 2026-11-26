# -*- coding: utf-8 -*-
"""run_llm_with_env.py —— 用 .env 里的密钥跑课件的 LLM 抽取脚本（不改课件代码）

为什么需要这个壳：
    课件 `llm_zero_shot/llm_extract.py` 只从**环境变量**读密钥
    （`os.getenv("LLM_API_KEY")`），它自己不调用 `load_dotenv()`。
    所以你在项目根目录放的那个 `.env` 不会自动生效 —— 除非有人先把它的内容读进环境。
    这个脚本就干这一件事：找到 .env → 载入 → 再把命令行原样转发给课件脚本。

为什么用 dotenv 而不是手写 `os.environ[...] = ...`：
    你的 .env 里值带了引号（LLM_API_KEY="sk-..."），
    python-dotenv 会自动剥掉引号；手写解析会把引号一起塞进 header，请求直接 401。

用法（和课件脚本的参数完全一致）:
    python run_llm_with_env.py --dry_run --n 2          # 只验环境+拼 prompt，不花 token
    python run_llm_with_env.py --mode zero --n 200      # 真跑 zero-shot
    python run_llm_with_env.py --mode few  --n 200      # 真跑 few-shot
"""
import os
import sys
import runpy
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE = (HERE.parent / "code").resolve()
SCRIPT = CODE / "llm_zero_shot" / "llm_extract.py"


def find_dotenv():
    """从本文件所在目录开始向上找 .env（你的那份在 D:\\Dev\\Py\\AStady\\.env）。"""
    for p in [HERE, *HERE.parents]:
        f = p / ".env"
        if f.is_file():
            return f
    return None


def mask(v):
    """只露出前 3 位，避免密钥出现在终端/日志/截图里。"""
    if not v:
        return "（空）"
    return f"{v[:3]}{'*' * min(8, max(0, len(v) - 3))}  ({len(v)} 字符)"


def main():
    try:
        from dotenv import load_dotenv
    except ImportError:
        raise SystemExit("缺 python-dotenv：pip install python-dotenv")

    envf = find_dotenv()
    if envf is None:
        raise SystemExit(f"从 {HERE} 向上找不到 .env")
    load_dotenv(envf, override=True)        # override=True：.env 优先于系统里已有的同名变量
    print(f"[env] 已载入 {envf}")

    for k in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL", "LLM_DISABLE_THINKING"):
        print(f"[env] {k:<20} = {mask(os.getenv(k))}")

    if not os.getenv("LLM_API_KEY"):
        raise SystemExit("[env] LLM_API_KEY 是空的，检查 .env 里那一行")

    if not SCRIPT.is_file():
        raise SystemExit(f"找不到课件脚本：{SCRIPT}")

    # 把参数原样转发给课件脚本（sys.argv[0] 用脚本自己的路径，argparse 才对）
    sys.argv = [str(SCRIPT)] + sys.argv[1:]
    print(f"[run] {SCRIPT.name} {' '.join(sys.argv[1:])}\n")
    runpy.run_path(str(SCRIPT), run_name="__main__")


if __name__ == "__main__":
    main()
