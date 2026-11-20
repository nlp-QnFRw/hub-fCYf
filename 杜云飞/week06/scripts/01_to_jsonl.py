#region 0. 模块说明
"""
把 THUCNews.zip 里的 74 万个 txt 解析成**单个 jsonl**（data/processed/all.jsonl）。

为什么直接读 zip、不解压：
  1. 74 万个小文件在 NTFS 上遍历 + 逐个 open 要几十分钟，解压本身也要写 1.8GB；
     zip 内顺序读只要几分钟；
  2. 省 1.8GB 磁盘（本机 D 盘只剩 46GB）；
  3. 解压还会遇到中文文件名编码问题（Windows 上 zip 的老毛病），这里统一处理。

必须处理的三件事：
  - **文件名编码**：老 zip 常用 GBK 存中文名且不带 UTF-8 标志位，zipfile 会按 cp437 解出乱码。
    统一尝试 `name.encode('cp437').decode('gbk')`，失败则保留原样。
  - **title / content 拆分**：THUCNews 的 txt 有的首行是标题、有的直接是正文。
    这里用"首行短且不以句末标点结尾"的启发式，并统计命中率（报告里要写明它只是启发式）。
  - **全局唯一 id**：每条记录分配自增 id，这是所有方法预测对齐、McNemar 配对检验的唯一键。

使用方式：
  python scripts/01_to_jsonl.py                 # 全量（预计 10~20 分钟）
  python scripts/01_to_jsonl.py --limit 2000    # 冒烟：只解析 2000 条
"""
#endregion

#region 1. 导入依赖
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.labels import LABELS  # noqa: E402
from common.paths import ALL_JSONL, DATA_RAW  # noqa: E402

#endregion

#region 2. 全局配置
ZIP_PATH: Path = DATA_RAW / "THUCNews.zip"
# 句末标点：首行以它们结尾说明是正文的一句话，而不是标题
SENT_END: tuple[str, ...] = ("。", "！", "？", "…", "”", '"')
TITLE_MAX_LEN: int = 50
#endregion

#region 3. 函数：解码文件名与正文
def fix_name(info: zipfile.ZipInfo) -> str:
    """修复 zip 内中文文件名的编码。

    本数据集的实际情况（已实测）：文件名是 **UTF-8 字节但没有打 0x800 标志位**，
    Python 于是按 cp437 解码，得到形如 "Σ╜ôΦé▓"（= 体育）的乱码。
    因此正确做法是 `encode("cp437").decode("utf-8")`；只有解不出来时才试 gbk
    （老式 GBK 打包的 zip 会用这种情况）。

    带 0x800 标志位的则说明 Python 已正确解码，直接用。
    """
    name = info.filename
    if info.flag_bits & 0x800:
        return name
    try:
        raw = name.encode("cp437")
    except UnicodeEncodeError:
        return name  # 本身就是正常 Unicode，别动它
    for enc in ("utf-8", "gbk"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return name


def decode_bytes(b: bytes) -> tuple[str, str]:
    """正文解码，返回 (文本, 实际使用的编码)。uft-8 优先，失败退 gbk。"""
    for enc in ("utf-8", "gbk"):
        try:
            return b.decode(enc), enc
        except UnicodeDecodeError:
            continue
    return b.decode("utf-8", errors="ignore"), "utf-8(ignore)"


def split_title(text: str) -> tuple[str, str]:
    """把「标题\\n正文」拆成 (title, content)。

    启发式：首行长度在 5~50 之间、且不以句末标点结尾、且全文不止一行 → 视为标题。
    命中率会在结束时打印，**报告里必须说明这是启发式而非官方标注**。
    """
    lines = text.split("\n")
    if len(lines) >= 2:
        first = lines[0].strip()
        if 5 <= len(first) <= TITLE_MAX_LEN and not first.endswith(SENT_END):
            return first, "\n".join(lines[1:]).strip()
    return "", text.strip()
#endregion

#region 4. 主流程
def main() -> None:
    ap = argparse.ArgumentParser(description="THUCNews.zip -> data/processed/all.jsonl")
    ap.add_argument("--limit", type=int, default=None, help="只解析前 N 条（冒烟）")
    ap.add_argument("--out", default=str(ALL_JSONL))
    ap.add_argument("--keep_dup", action="store_true", help="保留重复文本（默认按正文 md5 去重）")
    args = ap.parse_args()

    if not ZIP_PATH.exists():
        sys.exit(f"缺少 {ZIP_PATH}，请先运行 python scripts/00_download.py")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    counts: Counter[str] = Counter()
    enc_counts: Counter[str] = Counter()
    seen_md5: set[str] = set()
    n_files = n_dup = n_title = n_chars_total = 0
    next_id = 0
    unknown_dirs: Counter[str] = Counter()

    with zipfile.ZipFile(ZIP_PATH) as zf, open(out_path, "w", encoding="utf-8") as fout:
        infos = zf.infolist()
        print(f"zip 内共 {len(infos)} 个成员，开始解析 ...")
        for i, info in enumerate(infos):
            if info.is_dir():
                continue
            name = fix_name(info).replace("\\", "/")
            # 跳过 macOS 打包残留：__MACOSX/ 下是 AppleDouble 资源叉（每个 223 字节的 "._xxx"），
            # 本数据集里这类垃圾成员有 83 万个，不筛掉会把条数搞乱
            if name.startswith("__MACOSX") or "/._" in name or name.split("/")[-1].startswith("._"):
                continue
            if not name.lower().endswith(".txt"):
                continue
            parts = name.split("/")
            if len(parts) < 2:
                continue
            label = parts[-2]  # 上一级目录名 = 类别名
            if label not in LABELS:
                unknown_dirs[label] += 1
                continue

            text, enc = decode_bytes(zf.read(info))
            enc_counts[enc] += 1
            if not text.strip():
                continue

            if not args.keep_dup:
                h = hashlib.md5(text.encode("utf-8")).hexdigest()
                if h in seen_md5:
                    n_dup += 1
                    continue
                seen_md5.add(h)

            title, content = split_title(text)
            if title:
                n_title += 1
            rec = {
                "id": next_id,
                "label": label,
                "title": title,
                "content": content,
                "n_chars": len(title) + len(content),
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            next_id += 1
            counts[label] += 1
            n_chars_total += rec["n_chars"]
            n_files += 1

            if args.limit and n_files >= args.limit:
                print(f"到达 --limit {args.limit}，提前结束")
                break
            if n_files % 50000 == 0:
                print(f"  已写入 {n_files} 条（{time.time() - t0:.0f}s）", flush=True)

    # ---- 结果摘要 ----
    print("\n" + "=" * 70)
    print(f"写入 {out_path}：{n_files} 条，耗时 {(time.time() - t0) / 60:.1f} min")
    print(f"文本编码分布：{dict(enc_counts)}")
    print(f"标题命中率：{n_title / max(1, n_files) * 100:.1f}%（启发式拆分，非官方标注）")
    print(f"重复正文剔除：{n_dup} 条（{n_dup / max(1, n_files + n_dup) * 100:.2f}%）")
    print(f"平均文本长度：{n_chars_total / max(1, n_files):.0f} 字")
    if unknown_dirs:
        print(f"[警告] 出现未知类别目录（已跳过）：{dict(unknown_dirs.most_common(5))}")
    print("\n类别分布：")
    head = max(counts.values()) if counts else 1
    for name in LABELS:
        c = counts.get(name, 0)
        print(f"  {name:<4} {c:>7}  {'█' * int(c / head * 40)}")
    if counts:
        print(f"\n长尾比（头部/尾部）= {head / max(1, min(counts.values())):.1f}×  "
              f"（规范预警线 10×）")
    print("\n下一步：python scripts/02_splits.py")


if __name__ == "__main__":
    main()
#endregion
