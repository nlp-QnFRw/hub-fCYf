#region 0. 模块说明
"""
下载 THUCNews 数据集（THUCTC 官方发布的中文新闻分类语料）。

背景（重要）：
  thuctc.thunlp.org 的首页现在跳到了 THULAC 分词工具，`/message` 是一个要填信息的表单页。
  但**文件直链仍然有效**：`http://thuctc.thunlp.org/source/THUCNews.zip`（1,560,309,796 字节）。
  本脚本用 Range 断点续传，断了重跑即可，不会从头再来。

设计重点：
  1. 文件大小做**硬校验**，不是"下完就算成功"——网络中断留下的半截 zip 会让后续
     解析报 CRC 错误，那种错更难查；
  2. 已存在且大小正确则直接跳过，便于反复调用；
  3. 进度按 MB 打印，避免在 1.45GB 的下载里"看起来卡死"。

使用方式：
  python scripts/00_download.py
  python scripts/00_download.py --check     # 只校验，不下载
"""
#endregion

#region 1. 导入依赖
from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.paths import DATA_RAW, THUCNEWS_URL, THUCNEWS_ZIP_BYTES  # noqa: E402

#endregion

#region 2. 全局配置
ZIP_PATH: Path = DATA_RAW / "THUCNews.zip"
CHUNK: int = 1 << 20  # 1MB 一块
#endregion

#region 3. 函数：断点续传下载
def download(url: str = THUCNEWS_URL, dst: Path = ZIP_PATH, expected: int = THUCNEWS_ZIP_BYTES) -> Path:
    """下载到 dst，已存在的部分用 Range 续传。

    返回：
      dst 路径（保证大小 == expected）
    异常：
      RuntimeError，下载完成后大小不符（通常是源站换了文件或网络中断）
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    have = dst.stat().st_size if dst.exists() else 0

    if have == expected:
        print(f"[跳过] {dst.name} 已完整（{expected / 1024**3:.2f} GB）")
        return dst

    if have > expected:
        print(f"[警告] 本地文件比预期大（{have} > {expected}），删除重下")
        dst.unlink()
        have = 0

    if have:
        print(f"[续传] 已有 {have / 1048576:.0f} MB，从断点继续")

    req = urllib.request.Request(url)
    if have:
        req.add_header("Range", f"bytes={have}-")

    mode = "ab" if have else "wb"
    t0 = time.time()
    last_report = 0.0
    with urllib.request.urlopen(req, timeout=60) as resp, open(dst, mode) as f:
        total = expected
        got = have
        while True:
            chunk = resp.read(CHUNK)
            if not chunk:
                break
            f.write(chunk)
            got += len(chunk)
            now = time.time()
            if now - last_report > 5:  # 每 5 秒报一次，别刷屏
                last_report = now
                speed = (got - have) / max(1e-6, now - t0) / 1048576
                pct = got / total * 100
                eta = (total - got) / max(1e-6, (got - have) / max(1e-6, now - t0)) / 60
                print(f"  {got / 1048576:>7.0f}/{total / 1048576:.0f} MB  {pct:5.1f}%  "
                      f"{speed:5.2f} MB/s  剩余约 {eta:.0f} min", flush=True)

    size = dst.stat().st_size
    if size != expected:
        raise RuntimeError(f"下载完成但大小不符：{size} != {expected}（可能源站文件已变更）")
    print(f"[完成] {dst}  {size / 1024**3:.2f} GB，用时 {(time.time() - t0) / 60:.1f} min")
    return dst
#endregion

#region 4. 主流程
def main() -> None:
    ap = argparse.ArgumentParser(description="下载 THUCNews 数据集（断点续传 + 大小校验）")
    ap.add_argument("--check", action="store_true", help="只校验已下载文件，不联网")
    ap.add_argument("--url", default=THUCNEWS_URL)
    args = ap.parse_args()

    if args.check:
        ok = ZIP_PATH.exists() and ZIP_PATH.stat().st_size == THUCNEWS_ZIP_BYTES
        print(f"{ZIP_PATH} -> {'✓ 完整' if ok else '✗ 不存在或大小不符'}")
        return

    download(args.url)
    print("\n下一步：python scripts/01_to_jsonl.py")


if __name__ == "__main__":
    main()
#endregion
