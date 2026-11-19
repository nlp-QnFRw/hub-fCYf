"""
下载预训练模型到 work/pretrain_models/

为什么不用 snapshot_download：
  1. hf-mirror 对 HEAD 请求限流（429），hub 客户端会卡在重试上；
  2. 它会建 .cache/ 目录并做临时文件清理，在受限环境里清理动作会被拦，
     结果是「文件下完了但目录状态不对」。
  直接按文件清单 GET 下载，只留真正需要的权重与分词器文件，干净且可控。

用法：
  python scripts/fetch_models.py                 # macbert + rbt3
  python scripts/fetch_models.py --with-qwen     # 追加 Qwen3-0.6B（1.5GB）
"""

import argparse
import os
import time
import urllib.request
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
BASE = os.environ["HF_ENDPOINT"]

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "pretrain_models"

COMMON = ["config.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt",
          "tokenizer.json"]
SPECS = {
    # hfl 这两个仓库只放了 pytorch_model.bin（没有 safetensors），别照抄别处的文件名
    "macbert": ("hfl/chinese-macbert-base", "chinese-macbert-base", COMMON + ["pytorch_model.bin"]),
    "rbt3": ("hfl/rbt3", "rbt3", COMMON + ["pytorch_model.bin"]),
    "qwen": ("Qwen/Qwen3-0.6B", "Qwen3-0.6B",
             ["config.json", "generation_config.json", "tokenizer_config.json", "tokenizer.json",
              "vocab.json", "merges.txt", "model.safetensors"]),
}


# 镜像对同一 IP 有「每 5 分钟 N 次请求」的限流，超限时返回 200 + 一段英文提示文本。
# 不校验内容的话，那个提示页会被当成权重存下来，脚本还报成功——事后只会看到
# 「Unrecognized model ... should have a model_type key」，很难往下载上想。
RATE_LIMIT_MARK = b"rate limit"
MS_FALLBACK = {
    "qwen": "https://www.modelscope.cn/models/Qwen/Qwen3-0.6B/resolve/master/{}",
}


def _valid(target: Path) -> bool:
    if target.stat().st_size == 0:
        return False
    if target.suffix == ".json":
        import json as _json
        try:
            _json.loads(target.read_text(encoding="utf-8"))
        except Exception:
            return False
    return True


def fetch(repo: str, fname: str, target: Path, retries: int = 4,
          fallback: str | None = None) -> bool:
    tmp = target.with_suffix(target.suffix + ".part")
    urls = [f"{BASE}/{repo}/resolve/main/{fname}"]
    if fallback:
        urls.append(fallback.format(fname))

    for k in range(retries):
        url = urls[0] if k < retries - 1 or len(urls) == 1 else urls[1]
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
            with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
                total = int(r.headers.get("Content-Length") or 0)
                got = 0
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    got += len(chunk)
                    if total:
                        pct = got * 100 // total
                        print(f"\r  {fname:<24} {got/1e6:6.1f}/{total/1e6:.1f}MB {pct:3d}%",
                              end="", flush=True)
            print()
            if got < 1024 and RATE_LIMIT_MARK in tmp.read_bytes()[:4096].lower():
                print(f"  [rate-limit] {fname}：镜像限流，{60 if k < retries-1 else 0}s 后重试")
                tmp.unlink(missing_ok=True)
                time.sleep(60)
                continue
            if not _valid(tmp):
                print(f"  [invalid] {fname}：内容不是合法文件，重试")
                tmp.unlink(missing_ok=True)
                time.sleep(3 * (k + 1))
                continue
            tmp.replace(target)
            return True
        except Exception as e:
            print(f"\n  [retry {k+1}/{retries}] {fname}: {e}")
            time.sleep(3 * (k + 1))
    if tmp.exists():
        tmp.unlink(missing_ok=True)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-qwen", action="store_true")
    ap.add_argument("--only", choices=list(SPECS), default=None)
    args = ap.parse_args()

    keys = [args.only] if args.only else (["macbert", "rbt3", "qwen"] if args.with_qwen
                                          else ["macbert", "rbt3"])
    for k in keys:
        repo, local, files = SPECS[k]
        d = DEST / local
        d.mkdir(parents=True, exist_ok=True)
        print(f"[down] {repo} → {d}")
        ok = 0
        for fn in files:
            target = d / fn
            if target.exists() and target.stat().st_size > 0:
                print(f"  {fn:<24} 已存在，跳过")
                ok += 1
                continue
            if fetch(repo, fn, target, fallback=MS_FALLBACK.get(k)):
                ok += 1
        print(f"[{ 'ok' if ok == len(files) else 'fail'}] {local}：{ok}/{len(files)} 个文件")


if __name__ == "__main__":
    main()
