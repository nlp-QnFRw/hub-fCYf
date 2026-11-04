"""
PDF 语料提取脚本 —— 将 PDF（文字版或扫描版）提取为符合 language_model.py 读取格式的纯文本语料。

依赖安装（只需执行一次）:
    pip install pymupdf easyocr

用法示例:
    python extract_pdf_corpus.py "E:/edge download/【金融学】.pdf" -o corpus_finance.txt
    python extract_pdf_corpus.py input.pdf -o output.txt --dpi 150 --batch 10

输出格式:
    UTF-8 编码 .txt，每行一句/一段，与 corpus_general.txt 保持一致，
    可直接被 language_model.py 的 load_corpus("*.txt") 读取。
"""

import argparse
import os
import re
import sys
import warnings

# 屏蔽 pymupdf 的 FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

# EasyOCR + torch 可能会出现 libiomp5md.dll 重复加载的错误，
# 在 Windows CPU 环境下通过环境变量屏蔽（对结果无影响）。
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def is_scanned_pdf(doc, sample_pages=5):
    """抽样检测 PDF 是否为扫描版（无文字层）。"""
    checked = 0
    text_total = 0
    for i in range(0, min(len(doc), sample_pages * 3), 3):
        page = doc[i]
        text = page.get_text()
        text_total += len(text.strip())
        checked += 1
    return text_total < 50  # 若几页加起来不到 50 个字符，认为是扫描版


def clean_line(line):
    """单行清洗：去除首尾空白、归一化内部空格、过滤乱码片段。"""
    line = line.strip()
    line = re.sub(r"\s+", " ", line)
    # 过滤纯数字、纯符号、过短的行
    if len(line) < 4:
        return ""
    if re.fullmatch(r"[0-9\s\.\-]+", line):
        return ""
    return line


def split_chinese_sentences(text):
    """按中文句末标点切分，并保留标点。"""
    # 按 。！？… 切分，同时保留标点
    parts = re.split(r"([。！？…]+)", text)
    sentences = []
    i = 0
    while i < len(parts):
        s = parts[i]
        if i + 1 < len(parts) and re.match(r"[。！？…]+", parts[i + 1]):
            s += parts[i + 1]
            i += 2
        else:
            i += 1
        s = s.strip()
        if s:
            sentences.append(s)
    return sentences


def clean_ocr_text(text):
    """对 OCR 结果进行后处理：分段、分句、过滤垃圾行。"""
    lines = text.splitlines()
    out_lines = []
    buf = ""

    for line in lines:
        line = clean_line(line)
        if not line:
            continue
        # 若行尾没有句末标点，且下一行看起来是同一段，则合并
        buf += line
        if line[-1] in "。！？…":
            # 尝试按句切分
            sentences = split_chinese_sentences(buf)
            out_lines.extend(sentences)
            buf = ""
        else:
            buf += " "

    if buf.strip():
        sentences = split_chinese_sentences(buf.strip())
        out_lines.extend(sentences)

    return out_lines


def extract_text_pdf(doc):
    """提取文字版 PDF 的文本。"""
    all_lines = []
    for page in doc:
        text = page.get_text()
        lines = clean_ocr_text(text)
        all_lines.extend(lines)
    return all_lines


def extract_ocr_pdf(doc, reader, dpi=150, start=0, end=None, verbose=True):
    """对扫描版 PDF 逐页 OCR 提取。"""
    all_lines = []
    end = end or len(doc)
    for idx in range(start, end):
        page = doc[idx]
        pix = page.get_pixmap(dpi=dpi)
        img_bytes = pix.tobytes("png")
        result = reader.readtext(img_bytes, detail=0)
        page_text = "\n".join(result)
        lines = clean_ocr_text(page_text)
        all_lines.extend(lines)
        if verbose and (idx + 1) % 10 == 0:
            print(f"  OCR progress: {idx + 1}/{end} pages ...", file=sys.stderr)
    return all_lines


def main():
    parser = argparse.ArgumentParser(description="PDF 转 language_model.py 语料")
    parser.add_argument("pdf", help="输入 PDF 路径")
    parser.add_argument("-o", "--output", default="corpus_finance.txt", help="输出 txt 路径")
    parser.add_argument("--dpi", type=int, default=150, help="OCR 渲染分辨率 (默认 150)")
    parser.add_argument("--start", type=int, default=0, help="起始页码（0-based，含）")
    parser.add_argument("--end", type=int, default=None, help="结束页码（0-based，不含）")
    parser.add_argument("--batch", type=int, default=0, help="每 N 页保存一次中间结果（防止崩溃丢失进度）")
    parser.add_argument("--quiet", action="store_true", help="减少日志输出")
    args = parser.parse_args()

    if not os.path.exists(args.pdf):
        print(f"错误: 找不到文件 {args.pdf}", file=sys.stderr)
        sys.exit(1)

    print(f"打开 PDF: {args.pdf}")
    doc = __import__("fitz").open(args.pdf)
    total = len(doc)
    end = args.end or total
    print(f"总页数: {total}, 处理范围: [{args.start}, {end})")

    scanned = is_scanned_pdf(doc)
    if scanned:
        print("检测到扫描版 PDF，将使用 OCR 识别（首次运行会自动下载模型，请耐心等待）...")
        import easyocr
        reader = easyocr.Reader(["ch_sim", "en"], gpu=False)
        lines = extract_ocr_pdf(doc, reader, dpi=args.dpi, start=args.start, end=end, verbose=not args.quiet)
    else:
        print("检测到文字版 PDF，直接提取文本...")
        lines = extract_text_pdf(doc)

    doc.close()

    # 最终过滤：去重相邻行、去除过短行
    filtered = []
    prev = None
    for line in lines:
        if line == prev:
            continue
        if len(line) < 5:
            continue
        filtered.append(line)
        prev = line

    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(filtered))

    print(f"完成！共写入 {len(filtered)} 行，输出文件: {args.output}")


if __name__ == "__main__":
    main()
