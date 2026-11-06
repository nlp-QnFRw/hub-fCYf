"""
prepare_novel.py —— 把下载到的中文小说清洗成训练语料（一行一篇文档）
================================================================================
为什么用小说，而不是维基百科：
    · 小说是唯一一种"续写本身就是正确任务"的语料 —— 不需要 SFT 教它回答，
      因为"接着往下写"就是它该干的事，也就不会出现"问北京、答新闻"的错位。
    · 叙事文本的人称、场景、因果在段内自洽，小模型也能学到"谁做了什么"。
    · 对话、心理、景物这些位置天然有多个都合理的续写 —— 这就是温度/top-p
      终于能被看出来的地方（维基百科每个位置几乎只有一种写法，分布太尖）。

清洗做四件事：
    1. 编码兜底：utf-8-sig / utf-8 / gb18030 依次试（网文 txt 一半是 GBK）
    2. 去噪：章节标题、"求月票/求收藏"、站点广告、作者注、纯符号分隔行
    3. 过滤：中文字符占比过低（译作、乱码）、过短或过长的段落
    4. 去重：段级哈希去重（网文互相抄袭严重，重复段落会让模型陷入复读）

输出格式：一行 = 一篇约 1500~3000 字的"文档"，段与段之间用两个全角空格（　　）
分隔 —— 保留段落边界的同时不破坏"一行一文档"的结构（train.py 按行读）。

用法：
    conda activate py312
    python prepare_novel.py                       # 默认目标 8000 万字
    python prepare_novel.py --target 3e7          # 只做 3000 万字，快速试
================================================================================
"""

#region 0. 导入库与路径   【模块:标准库 argparse/hashlib/os/random/re/sys/collections】
import argparse
import hashlib
import os
import random
import re
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))  # 脚本所在目录
NOVEL_DIR = os.path.join(HERE, "data", "novel")  # 原始 txt 存放目录，下分 lit100/ 与 webnovel/
OUT = os.path.join(HERE, "data", "corpus_novel.txt")  # 默认输出文件
#endregion

# ── 噪声规则 ──────────────────────────────────────────────────────────────
#region 1. 正则规则表   【集合:RE_CHAPTER/RE_AD/RE_AUTHOR/RE_SYMBOL/RE_SPACE/RE_HAN/RE_TRANS/RE_JUNK/RE_TOC】
# 章节标题：第X章/第X回/楔子/序章/尾声……（后面 30 字以内算标题行，不算正文）
RE_CHAPTER = re.compile(r"^\s*(第[0-9零一二三四五六七八九十百千两]{1,7}[章节回卷部集篇]|"
                        r"楔子|序章|序言|尾声|后记|番外)[^\n]{0,30}$")
# 站点广告与求票话术：命中任意一条，整行丢掉
RE_AD = re.compile(r"(https?://|www\.|\.com|\.net|\.cc|笔趣阁|顶点小说|"
                   r"求月票|求推荐票|求收藏|求订阅|求打赏|求评价票|"
                   r"本章未完|请点击下一页|一秒记住|手机用户请|txt下载|"
                   r"首发地址|最新网址|章节错误|内容举报)")
# 作者注：一般出现在段首
RE_AUTHOR = re.compile(r"^\s*(作者有话要说|作者有话说|PS[:：]|ps[:：]|注[:：])")
# 纯符号分隔行：全是 * - = _ ~ · — … 之类，没有任何内容
RE_SYMBOL = re.compile(r"^[\s\*\-=_~·—…\.。、#=+]+$")
# 连续 2 个以上空白（含全角空格）：压成一个全角空格做段落缩进标记
RE_SPACE = re.compile(r"[ \t\u3000]{2,}")
# 中文字符：用来算"汉字占比"，判断是不是中文正文
RE_HAN = re.compile(r"[\u4e00-\u9fff]")
# 译本一律剔除：翻译腔 + 音译人名会污染"中文叙事"的语感
RE_TRANS = re.compile(r"[（(]\s*(美|英|日|俄|法|德|加|澳|意|西|韩|印|爱尔兰|奥地利|比利时|"
                      r"瑞士|芬兰|瑞典|挪威|丹麦|荷兰|波兰|捷克|匈牙利|希腊|葡萄牙|土耳其|"
                      r"以色列|阿根廷|巴西|南非|新西兰|苏联|乌克兰)\s*[）)]")
# 电子书制作组的版权页、目录页
RE_JUNK = re.compile(r"(好读书柜|经典版|电子书|书库|校对|精校|整理制作|版权所有|"
                     r"本书由|下载于|更多精彩|目录$|^目录)")
# 目录行：一行里塞了三五个"第X回/第X章"，是回目表而不是正文
RE_TOC = re.compile(r"第[0-9零一二三四五六七八九十百千两]{1,7}[回章节]")
#endregion


#region 2. 编码兜底读取   【函数:read_text(文件路径),return 字符串（失败返回 ""）】
def read_text(path):
    """编码兜底：网文 txt 经常是 GBK，个别是 UTF-16 / BIG5。

    为什么不能一股脑用 gb18030：gb18030 几乎能解任何字节序列（不会抛异常），
    但它也会把 UTF-8 的中文解成乱码。所以顺序要"严→宽"，并且用汉字占比兜底校验。
    """
    with open(path, "rb") as f:
        raw = f.read()  # 先当二进制读，编码的事后面再定
    for enc in ("utf-8-sig", "utf-8", "gb18030", "utf-16", "big5"):
        try:
            t = raw.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue  # 这个编码解不了，换下一个
        # 解出来的中文占比太低，说明选错了编码，换下一个
        # gb18030 是最后兜底，它无论如何都能解，所以直接收下
        if enc == "gb18030" or len(t) < 50 or han_ratio(t[:2000]) > 0.4:
            return t
    return ""  # 全都失败，调用方会跳过这个文件
#endregion

#region 3. 工具：汉字占比   【函数:han_ratio(字符串),return float 0~1】
def han_ratio(s):
    """返回汉字字符占全长的比例。用来判断"这是不是中文正文"。"""
    return len(RE_HAN.findall(s)) / max(1, len(s))  # 兜底 1，防止空串除零
#endregion

#region 4. 清洗单个文件：逐行过滤   【函数:clean_file(文件路径),return (段落列表, 原始字数)】
def clean_file(path, cache_len=None):
    """返回 (段落列表, 原始字数)。段落已去掉噪声行，未去重。"""
    raw = read_text(path)
    if not raw:
        return [], 0  # 读不出来就返回空，调用方会计入"跳过"
    n_raw = len(raw)  # 原始字数（清洗前的，用来算保留率）
    # 统一换行与空白：Windows 的 \r\n、老 Mac 的 \r、不间断空格，全部归一
    raw = raw.replace("\r\n", "\n").replace("\r", "\n").replace("\u00a0", " ")
    paras = []  # 列表，装清洗后的段落
    for line in raw.split("\n"):
        line = line.strip(" \t\u3000")  # 去掉首尾空白（含全角空格）
        if len(line) < 12:
            continue  # 太短的行基本是标题残片、页眉，没信息量
        # 下面四个 continue 就是"去噪"的全部：符号行、章节名、广告、作者注
        if RE_SYMBOL.match(line) or RE_CHAPTER.match(line) or RE_AD.search(line):
            continue
        if RE_AUTHOR.match(line) or RE_JUNK.search(line):
            continue
        if len(RE_TOC.findall(line)) >= 3:     # 回目表
            continue
        if han_ratio(line) < 0.55:      # 译作里大段英文、乱码
            continue
        if len(line) > 2000:            # 整章没分段的，按句号硬切
            buf = ""  # 攒句子用
            # (?<=[。！？]) 是"后顾断言"：在句号/叹号/问号之后切开，但符号本身留在前一段
            for seg in re.split(r"(?<=[。！？])", line):
                buf += seg
                if len(buf) >= 800:  # 攒够 800 字就切成一段
                    paras.append(buf.strip())
                    buf = ""
            if len(buf.strip()) >= 12:  # 最后剩下的不够 12 字就丢掉
                paras.append(buf.strip())
        else:
            # 正常段落：把行内连续空白压成一个全角空格，模拟中文首行缩进
            paras.append(RE_SPACE.sub("　", line))
    return paras, n_raw
#endregion


#region 5. 主流程：遍历文件 → 清洗 → 拼文档 → 写出   【函数:main(),return None】
def main():
    #region 5.1 命令行参数   【集合:args】
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=float, default=8e7, help="目标总字数（默认 8000 万）")
    ap.add_argument("--doc_chars", type=int, default=2000, help="每篇文档目标字数")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--seed", type=int, default=42)  # 固定种子，保证每次打乱顺序一致
    args = ap.parse_args()
    #endregion

    #region 5.2 收集文件清单   【集合:files】
    sets = [("lit100", os.path.join(NOVEL_DIR, "lit100")),  # 100 本严肃文学
            ("webnovel", os.path.join(NOVEL_DIR, "webnovel"))]  # 900 多本通俗长篇
    files = {}  # 字典，tag → 文件路径列表
    for tag, root in sets:
        got = []  # 这一类的所有 txt
        for dp, _, fns in os.walk(root):  # os.walk 递归遍历子目录
            for fn in fns:
                if fn.lower().endswith(".txt"):
                    got.append(os.path.join(dp, fn))
        files[tag] = sorted(got)  # 排序保证顺序可复现
        print(f"[{tag}] {len(got)} 个 txt 文件")
    if not sum(len(v) for v in files.values()):
        sys.exit("没找到语料，先跑下载：data/novel/ 下应有 lit100/ 和 webnovel/")
    #endregion

    #region 5.3 清洗状态   【集合:rnd/seen/docs/kept/dropped/raw_total/dedup_hit】
    rnd = random.Random(args.seed)
    seen = set()  # 已经出现过的段落哈希，用于去重
    docs = []  # 列表，装最终拼好的文档
    kept = dropped = 0  # kept=已收字数，dropped=跳过的碎文件数
    raw_total = 0  # 所有源文件加起来的原始字数
    dedup_hit = 0  # 被去重丢掉的段落数
    #endregion

    #region 5.4 内层函数：把段落拼成文档   【函数:flush(段落列表, 字数预算),return 当前已收字数】
    def flush(paras, budget):
        """把段落拼成一篇篇文档；返回写入的字数，超出 budget 就停。

        每篇文档攒到 args.doc_chars 就切断，这样"一行一文档"既保留上下文，
        又不会让单行过长导致 train.py 切样本时浪费。
        """
        nonlocal kept, dedup_hit  # 要改外层的计数器
        out, buf = [], ""  # out=这篇文件产出的文档，buf=正在攒的文档
        for p in paras:
            # md5 做段级哈希：比存原文字符串省内存，碰撞概率可忽略
            key = hashlib.md5(p.encode("utf-8", "ignore")).digest()
            if len(p) >= 24:  # 太短的段落（"他说。"）本来就容易重复，不做去重
                if key in seen:
                    dedup_hit += 1
                    continue  # 这段以前见过，丢掉
                seen.add(key)
            if kept + len(p) > budget:
                break  # 到预算了，剩下的留到下次
            # 段与段之间用两个全角空格连起来：既有段落边界，又不破坏"一行一文档"
            buf = (buf + "　　" + p) if buf else p
            kept += len(p)
            if len(buf) >= args.doc_chars:  # 攒够一篇就切出去
                out.append(buf)
                buf = ""
        if buf:  # 最后不满一篇的也留下来，别浪费
            out.append(buf)
        docs.extend(out)
        return kept
    #endregion

    #region 5.5 第一轮：严肃文学（腔调来源）
    # 严肃文学语言最规范，是"文风"的来源，所以先收，并且限制在总预算的 25%、最多 3000 万字
    lit_budget = min(args.target * 0.25, 3e7)
    for path in files["lit100"]:
        paras, n_raw = clean_file(path)
        raw_total += n_raw
        if kept >= lit_budget:
            break  # 严肃文学收够了就停，剩下的额度给网文
        flush(paras, lit_budget)
    #endregion

    #region 5.6 第二轮：通俗长篇（数量来源）
    web = files["webnovel"]
    rnd.shuffle(web)  # 打乱顺序，保证题材分散（否则可能连收十本同一作者的）
    for path in web:
        if kept >= args.target:
            break
        if RE_TRANS.search(os.path.basename(path)):   # 译本：翻译腔会毁掉语感
            dropped += 1
            continue
        paras, n_raw = clean_file(path)
        raw_total += n_raw
        if len(paras) < 30:              # 太碎的文件直接跳过
            dropped += 1
            continue
        flush(paras, args.target)
    #endregion

    #region 5.7 写出文件 + 抽样自检
    print(f"原始 {raw_total/1e4:.0f} 万字 → 清洗后 {kept/1e4:.0f} 万字，"
          f"{len(docs)} 篇文档，去重丢弃 {dedup_hit} 段，跳过碎文件 {dropped} 个")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)  # 输出目录可能不存在
    with open(args.out, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(d + "\n")  # 一行一篇，train.py 就是按行读的

    # 抽样看一眼，确认没被广告或乱码污染
    print("\n—— 抽样（第 1 篇前 120 字）——")
    print(docs[0][:120])  # 打印第一篇的开头
    print("\n—— 抽样（随机一篇前 120 字）——")
    print(rnd.choice(docs)[:120])  # 打印随机一篇的开头
    print(f"\n已写出 {args.out}  ({os.path.getsize(args.out)/1048576:.1f} MB)")  # 打印输出路径与体积
    #endregion
#endregion


#region 6. 入口
if __name__ == "__main__":
    main()
#endregion
