"""下载公开数据集原始文件到 data/raw/。

- CLUENER2020：CLUE 官方 zip（storage.googleapis.com，国内可直连）
- MSRA NER：GitHub 仓库的 BIO 文本，走 jsdelivr CDN（raw.githubusercontent 被墙）

用法：
    python scripts/download_data.py            # 全部
    python scripts/download_data.py --only cluener

下载失败不会中断整个流程：已存在的文件直接跳过，最后打印每个数据集的状态。
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import DATA_RAW, get_logger, human_time  # noqa: E402

import urllib.request
from src.common import Timer

SOURCES = {
    "cluener": {
        "url": "https://storage.googleapis.com/cluebenchmark/tasks/cluener_public.zip",
        "file": "cluener_public.zip",
        "unzip_to": "cluener",
    },
    "msra_train": {
        "url": "https://cdn.jsdelivr.net/gh/OYE93/Chinese-NLP-Corpus@master/NER/MSRA/msra_train_bio.txt",
        "file": "msra_train_bio.txt",
    },
    "msra_test": {
        "url": "https://cdn.jsdelivr.net/gh/OYE93/Chinese-NLP-Corpus@master/NER/MSRA/msra_test_bio.txt",
        "file": "msra_test_bio.txt",
    },
}

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) week07-homework"}


def download(url: str, dst: Path, retries: int = 3) -> bool:
    if dst.exists() and dst.stat().st_size > 1024:
        print(f"  [skip] {dst.name} 已存在 {dst.stat().st_size/1024:.0f} KB")
        return True
    for i in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=60) as r, open(dst, "wb") as f:
                while True:
                    chunk = r.read(1 << 16)
                    if not chunk:
                        break
                    f.write(chunk)
            print(f"  [ok] {dst.name} {dst.stat().st_size/1024:.0f} KB")
            return True
        except Exception as e:  # noqa: BLE001
            print(f"  [retry {i}/{retries}] {dst.name}: {type(e).__name__}: {str(e)[:120]}")
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="all", help="cluener|msra|all")
    args = ap.parse_args()

    want = set()
    if args.only in ("all", "cluener"):
        want.add("cluener")
    if args.only in ("all", "msra"):
        want.update({"msra_train", "msra_test"})

    log = get_logger("download")
    ok = True
    with Timer() as t:
        if "cluener" in want:
            src = SOURCES["cluener"]
            dst = DATA_RAW / src["file"]
            log.info("CLUENER2020: %s", src["url"])
            if download(src["url"], dst):
                out = DATA_RAW / src["unzip_to"]
                if not out.exists():
                    with zipfile.ZipFile(dst) as z:
                        z.extractall(out)
                log.info("  解压到 %s", out)
            else:
                ok = False
        if "msra_train" in want:
            for name in ("msra_train", "msra_test"):
                src = SOURCES[name]
                log.info("%s: %s", name, src["url"])
                ok &= download(src["url"], DATA_RAW / src["file"])

    log.info("耗时 %s", human_time(t.seconds))
    log.info("data/raw 内容:")
    for p in sorted(DATA_RAW.rglob("*")):
        if p.is_file():
            log.info("  %8.1f KB  %s", p.stat().st_size / 1024, p.relative_to(DATA_RAW))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
