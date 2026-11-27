# -*- coding: utf-8 -*-
"""生成 week7 交付报告 DOCX。

数据来源分两类，不要混：
  1. 可自动读取的：数据体检 `outputs/data_audit.json`、sample 复评 `outputs/sample_subset.csv`、
     本地重训 `outputs/local_run/metrics_*.json`；
  2. **PPT 全量参考表**：`full` 字典里的 930 句指标/训练成本/延迟是按课程 PPT 复现的，
     刻意硬编码（含人工整理的成本列），不随 reference/metrics 自动漂移。
     改这些数字时必须同步 `f1_changes.py` 的一致性检查，否则报告与指标会打架。

运行（使用 Office 运行时自带的 python，需 python-docx）：
  <office-python> report/build_report.py
"""

import csv
import json
import sys
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
OUT = ROOT / "outputs"
FIG = OUT / "figures"

NAVY = RGBColor(0x1F, 0x38, 0x64)
DARK = RGBColor(0x26, 0x26, 0x26)
GRAY = RGBColor(0x59, 0x59, 0x59)
ACCENT = RGBColor(0xC0, 0x39, 0x2B)


# ─────────────────────────── 文档基础样式 ───────────────────────────

def set_run_font(run, size=10.5, bold=False, color=DARK, name_ascii="Times New Roman",
                 name_east="宋体"):
    run.font.name = name_ascii
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    rPr = run._element.get_or_add_rPr()
    rFonts = rPr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = OxmlElement("w:rFonts")
        rPr.append(rFonts)
    rFonts.set(qn("w:ascii"), name_ascii)
    rFonts.set(qn("w:hAnsi"), name_ascii)
    rFonts.set(qn("w:eastAsia"), name_east)


def style_setup(doc):
    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(10.5)
    normal.font.color.rgb = DARK
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    normal.paragraph_format.line_spacing = 1.35
    normal.paragraph_format.space_after = Pt(4)

    for lvl, size, color in ((1, 15, NAVY), (2, 12.5, NAVY), (3, 11, DARK)):
        st = doc.styles[f"Heading {lvl}"]
        st.font.name = "Arial"
        st.font.size = Pt(size)
        st.font.bold = True
        st.font.color.rgb = color
        st._element.rPr.rFonts.set(qn("w:eastAsia"), "PingFang SC")
        st.paragraph_format.space_before = Pt(12 if lvl == 1 else 8)
        st.paragraph_format.space_after = Pt(5)
        st.paragraph_format.keep_with_next = True


def page_setup(doc):
    sec = doc.sections[0]
    sec.page_width = Cm(21.0)
    sec.page_height = Cm(29.7)
    sec.top_margin = Cm(2.2)
    sec.bottom_margin = Cm(2.0)
    sec.left_margin = Cm(2.2)
    sec.right_margin = Cm(2.2)
    sec.header_distance = Cm(1.2)
    sec.footer_distance = Cm(1.1)


def add_page_number(section):
    p = section.footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run("第 ")
    set_run_font(run, 8.5, color=GRAY)
    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), "PAGE")
    r = OxmlElement("w:r")
    t = OxmlElement("w:t")
    t.text = "1"
    r.append(t)
    fld.append(r)
    p._p.append(fld)
    run = p.add_run(" 页 / 共 ")
    set_run_font(run, 8.5, color=GRAY)
    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), "NUMPAGES")
    r = OxmlElement("w:r")
    t = OxmlElement("w:t")
    t.text = "1"
    r.append(t)
    fld.append(r)
    p._p.append(fld)
    run = p.add_run(" 页")
    set_run_font(run, 8.5, color=GRAY)


def shade(cell, hex_color):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_color)
    tcPr.append(shd)


def cell_borders(cell, color="BFBFBF", sz=4):
    tcPr = cell._tc.get_or_add_tcPr()
    borders = OxmlElement("w:tcBorders")
    for edge in ("top", "left", "bottom", "right"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), str(sz))
        el.set(qn("w:color"), color)
        borders.append(el)
    tcPr.append(borders)


def cell_text(cell, text, bold=False, size=8.5, color=DARK, align=WD_ALIGN_PARAGRAPH.CENTER):
    cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
    p = cell.paragraphs[0]
    p.alignment = align
    p.paragraph_format.space_after = Pt(1)
    p.paragraph_format.space_before = Pt(1)
    p.paragraph_format.line_spacing = 1.05
    run = p.add_run(str(text))
    set_run_font(run, size, bold=bold, color=color)


def add_table(doc, headers, rows, widths=None, font_size=8.5, header_bg="1F3864",
              align_left_cols=(), highlight_rows=(), keep_together=True):
    table = doc.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    hdr = table.rows[0].cells
    for i, h in enumerate(headers):
        cell_text(hdr[i], h, bold=True, size=font_size, color=RGBColor(0xFF, 0xFF, 0xFF))
        shade(hdr[i], header_bg)
        cell_borders(hdr[i], "1F3864")
    for ridx, row in enumerate(rows):
        cells = table.add_row().cells
        for i, val in enumerate(row):
            al = WD_ALIGN_PARAGRAPH.LEFT if i in align_left_cols else WD_ALIGN_PARAGRAPH.CENTER
            cell_text(cells[i], val, size=font_size, align=al)
            cell_borders(cells[i])
            if ridx in highlight_rows:
                shade(cells[i], "FFF2CC")
            elif ridx % 2 == 1:
                shade(cells[i], "F2F2F2")
    if widths:
        for i, w in enumerate(widths):
            table.columns[i].width = Cm(w)
        for row in table.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Cm(w)
    # 行不跨页；整表尽量不拆开（小表用）
    for r_i, row in enumerate(table.rows):
        trPr = row._tr.get_or_add_trPr()
        cant = OxmlElement("w:cantSplit")
        cant.set(qn("w:val"), "true")
        trPr.append(cant)
        if keep_together and r_i < len(table.rows) - 1:
            for cell in row.cells:
                for para in cell.paragraphs:
                    para.paragraph_format.keep_with_next = True
    # 表头重复
    trPr = table.rows[0]._tr.get_or_add_trPr()
    tblHeader = OxmlElement("w:tblHeader")
    tblHeader.set(qn("w:val"), "true")
    trPr.append(tblHeader)
    return table


def add_caption(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(8)
    run = p.add_run(text)
    set_run_font(run, 8.5, color=GRAY)


def add_para(doc, text, bold_prefix="", size=10.5, space_after=4, color=DARK):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.first_line_indent = Cm(0.74)
    if bold_prefix:
        r = p.add_run(bold_prefix)
        set_run_font(r, size, bold=True, color=NAVY)
    r = p.add_run(text)
    set_run_font(r, size, color=color)
    return p


def add_bullet(doc, text, bold_prefix=""):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.left_indent = Cm(0.6)
    if bold_prefix:
        r = p.add_run(bold_prefix)
        set_run_font(r, 10.5, bold=True, color=NAVY)
    r = p.add_run(text)
    set_run_font(r, 10.5)
    return p


def add_figure(doc, path, width_cm=16.2, caption=""):
    if not Path(path).exists():
        return
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(0)
    p.add_run().add_picture(str(path), width=Cm(width_cm))
    if caption:
        add_caption(doc, caption)


def add_callout(doc, title, lines):
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = table.rows[0].cells[0]
    cell.width = Cm(16.2)
    shade(cell, "EAF1F8")
    cell_borders(cell, "9DC3E6", 8)
    p = cell.paragraphs[0]
    r = p.add_run(title)
    set_run_font(r, 10, bold=True, color=NAVY)
    for line in lines:
        p = cell.add_paragraph()
        p.paragraph_format.space_after = Pt(2)
        r = p.add_run("• " + line)
        set_run_font(r, 9.5)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


# ─────────────────────────── 数据 ───────────────────────────

def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read_csv(path):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def local_metric(name):
    p = OUT / "local_run" / f"metrics_{name}.json"
    return load_json(p) if p.exists() else None


def build():
    doc = Document()
    page_setup(doc)
    style_setup(doc)
    add_page_number(doc.sections[0])

    audit = load_json(OUT / "data_audit.json")
    sample = read_csv(OUT / "sample_subset.csv")
    # 课程 PPT 全量 dev 930 的参考表（表 4-1）。
    # 元组：显示名、trigger F1、argument F1、event F1、训练/成本、延迟。
    # 这些 F1 与 reference/metrics 一致；延迟列按 PPT 口径（CRF 40.92ms 与项目实测
    # 74.28ms 不一致，正文已注明）。不要在未更新出处的情况下改这里的数。
    full = {
        "rule": ("L0 规则（词典+正则）", 0.6168, 0.1740, 0.0161, "0（纯 CPU）", "0.72 ms"),
        "bilstm_crf": ("L1 字符级 BiLSTM-CRF", 0.8165, 0.4634, 0.0885, "1,540 s（CPU 3 epoch）", "21.17 ms"),
        "bert_linear_256": ("L2 BERT+Linear", 0.8293, 0.5630, 0.1564, "531 s（12 层/3 epoch）", "12.26 ms"),
        "bert_crf_256": ("L2 BERT+CRF", 0.8320, 0.5815, 0.1697, "3,112 s（52 分钟/5.9×）", "40.92 ms"),
        "bert_crf_256_hard": ("L2 + 解码期硬约束（不重训）", 0.8464, 0.6566, 0.2177, "0（复用 ckpt，decode 90 s）", "40.92 ms"),
        "pipeline": ("L2 两段式（检测→类型→论元）", 0.6769, 0.3690, 0.0957, "1,377.5 s / 16 模型（压缩）", "49.87 ms"),
        "local_zero_n60": ("L3 本地 Qwen3-0.6B zero-shot", 0.4068, 0.1167, 0.0000, "本地推理（8 GB 卡）", "5.32 s"),
        "api_zero_n200": ("L3 API zero-shot（prompt v2）", 0.7789, 0.5224, 0.0402, "按 token（191.6k/200 条）", "3.62 s"),
        "api_few_n200": ("L3 API few-shot（prompt v2）", 0.8321, 0.5395, 0.0852, "按 token（311k/200 条）", "3.53 s"),
        "sft_lora_n200": ("L4 LoRA SFT（3,499 条训练，eval n=200）", 0.9000, 0.6971, 0.1850, "2,327 s（1 epoch/3,499 条）", "3.13 s"),
    }

    # ── 封面 ─────────────────────────────────────────────
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(40)
    r = p.add_run("八斗学院 · week7 信息抽取")
    set_run_font(r, 11, bold=True, color=GRAY)

    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(20)
    p.paragraph_format.space_after = Pt(4)
    r = p.add_run("金融公告事件抽取")
    set_run_font(r, 28, bold=True, color=NAVY, name_east="PingFang SC")

    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(18)
    r = p.add_run("多方法同口径对比、数据体检与场景选型")
    set_run_font(r, 16, bold=False, color=ACCENT, name_east="PingFang SC")

    p = doc.add_paragraph()
    r = p.add_run("规则  →  BiLSTM-CRF  →  BERT  →  两段式  →  LLM 提示  →  LoRA 微调  →  系统组合")
    set_run_font(r, 9.5, color=GRAY)

    meta = doc.add_table(rows=4, cols=2)
    meta.alignment = WD_TABLE_ALIGNMENT.LEFT
    meta_rows = [
        ("任务", "句子级事件抽取（事件类型 + 触发词 span + 角色论元 span）"),
        ("数据", "DuEE-Fin 派生；本次 sample：train 800 / dev 300 / detect 1,500+600 / docs 50"),
        ("口径", "trigger / argument / event_exact 三级 F1，全部由同一评测函数计算"),
        ("交付", "范胜利 · week7（数据 + 代码 + 指标 + 预测 + 报告）"),
    ]
    for i, (k, v) in enumerate(meta_rows):
        c0, c1 = meta.rows[i].cells
        c0.width, c1.width = Cm(2.2), Cm(14.0)
        cell_text(c0, k, bold=True, size=9.5, color=NAVY, align=WD_ALIGN_PARAGRAPH.LEFT)
        cell_text(c1, v, size=9.5, align=WD_ALIGN_PARAGRAPH.LEFT)
        for c in (c0, c1):
            c._tc.get_or_add_tcPr().append(OxmlElement("w:tcBorders"))
    doc.add_paragraph().paragraph_format.space_before = Pt(60)

    add_callout(doc, "一页结论", [
        "LoRA SFT 精度最高：3,499 条 / 1 epoch / 9 MB adapter，触发 0.900 / 论元 0.697，超过 5,747 条训练的 BERT。",
        "BERT+CRF + 解码期硬约束是性价比最高的一格：不重训，论元 +7.5pp、非法序列 62%→7.4%、事件级 0.218。",
        "规则最快（<3 ms）但论元只有 0.174；两段式结构最清晰，却因误差逐段放大跌到 0.369。",
        "数据 7 个文件全部可用；主要风险是长尾类型样本少（被约谈 8 句）、13%~14% 截断、少量重复。",
        "sample 指标不能直接对 PPT 全量 930 的数字：类型分布 TV 距离 0.053，样本量只有 1/3。",
    ])

    doc.add_page_break()

    # ── 1 执行摘要 ────────────────────────────────────────
    doc.add_heading("1  执行摘要", level=1)
    add_para(doc, "本报告在同一批 dev 数据、同一套评测函数（trigger / argument / event_exact 三级 F1）上，"
                  "复现并对比了课程 PPT 中的 10 种事件抽取做法：规则、BiLSTM-CRF、BERT+Linear、BERT+CRF、"
                  "CRF 解码期硬约束、两段式流水线、LLM zero-shot / few-shot、LoRA SFT。")
    add_para(doc, "本次交付同时完成两件事：① 在提供的 sample 数据上跑通规则、BiLSTM-CRF、BERT+Linear 的"
                  "本地重训与推理；② 用项目已保存的全量模型预测，在 sample 的同一批 200 句上重新评分，"
                  "保证 10 种方法严格同批可比。")
    add_bullet(doc, "LoRA SFT 触发词 0.900 / 论元 0.697 / 事件级 0.185，全面领先；", "精度上限：")
    add_bullet(doc, "BERT+CRF 只改解码（硬约束）就把论元从 0.5815 提到 0.6566（+7.5pp），"
                    "事件级 0.1697→0.2177，且 0 额外训练；", "最高性价比：")
    add_bullet(doc, "规则 <3 ms、BERT+Linear 11 ms、CRF 74 ms、LLM 3.1~3.6 s——"
                    "延迟跨了三个数量级，选型必须先看延迟预算；", "速度差异：")
    add_bullet(doc, "两段式标签空间小了一个数量级，论元反而从 0.5815 跌到 0.3690，"
                    "根因是检测漏 42.7%、类型错 5.5% 的误差逐段放大；", "反直觉：")
    add_bullet(doc, "7 个文件没有无效文件；span 100% 可还原、类型/角色 100% 在 schema 内；"
                    "但存在长尾、截断、少量重复，详见第 2 章。", "数据结论：")

    # ── 2 数据 ────────────────────────────────────────────
    doc.add_heading("2  任务与数据体检", level=1)
    doc.add_heading("2.1  任务定义", level=2)
    add_para(doc, "输入一句金融公告原文，输出一个结构化事件：事件类型（13 类）、触发词 span、"
                  "以及若干「角色 → 论元 span」。一行 = 一句话 + 恰好一个事件；"
                  "论元允许同一角色多值（如两个「中标公司」），但 event_exact 要求集合完全一致才算对。")
    add_para(doc, "数据为 DuEE-Fin 派生的句子级样本。schema 共 13 类事件、92 个「类型-角色」槽位、"
                  "61 个去重角色名；另有 __enums__ 保存枚举型角色的取值范围。")

    doc.add_heading("2.2  七个文件分别有什么用", level=2)
    rows = []
    for name, info in audit["files"].items():
        if not info.get("exists"):
            continue
        rows.append((name, info["rows"], info["unique_ids"],
                     {"train_extract.jsonl": "句级抽取训练", "dev_extract.jsonl": "句级抽取主评测",
                      "train_detect.jsonl": "检测器训练（含负样本）", "dev_detect.jsonl": "检测器评测",
                      "dev_docs.jsonl": "篇章级事件流演示"}.get(name, "-")))
    rows.append(("event_schema.json", "-", "-", "标签体系（13 类 / 92 槽 / 61 角色）"))
    rows.append(("sample_stats.json", "-", "-", "抽样元数据（seed=42，规模一致）"))
    add_table(doc, ["文件", "行数", "唯一 id", "用途"], rows,
              widths=[5.2, 2.2, 2.2, 6.6], font_size=8.5, align_left_cols=(0, 3))
    add_caption(doc, "表 2-1  数据文件清单：没有无效文件，但用途不同，不能混用")

    add_para(doc, "结论是：7 个文件都“有用”，但“有用”不等于“都能拿来训句级抽取模型”。"
                  "detect 集负责“有没有事件”，docs 集负责“一篇多事件”，"
                  "只有 extract 集能直接做句级事件抽取的训练与评测。")

    doc.add_heading("2.3  体检发现的问题", level=2)
    tr = audit["files"]["train_extract.jsonl"]
    dv = audit["files"]["dev_extract.jsonl"]
    leak = audit.get("train_dev_leakage", {})
    coverage = audit["schema"].get("slot_coverage", 0)
    rows = [
        ("span 精确性", "触发词/论元 100% 能在原文还原", "通过", "无"),
        ("类型/角色合法性", "全部在 schema 内，无未知标签", "通过", "无"),
        ("schema 覆盖", f"训练集覆盖 {coverage:.1%} 角色", "注意", "「减持部分占所持比例」训练集 0 例"),
        ("长尾类型", f"被约谈 {audit['files']['dev_extract.jsonl']['type_counts'].get('被约谈', 0)} 句、"
                     f"企业破产 {audit['files']['dev_extract.jsonl']['type_counts'].get('企业破产', 0)} 句（dev）",
         "注意", "长尾类 F1 波动大，macro 指标更稳"),
        ("截断", f"train {tr['truncate_rate_254']:.1%} / dev {dv['truncate_rate_254']:.1%} 超过 254 字",
         "注意", "max_len=256 会丢尾部论元/事件"),
        ("同角色多值", f"train {tr['multi_value_role_events']} 个事件、dev {dv['multi_value_role_events']} 个事件",
         "合法但难", "event_exact 要求集合完全一致，多值会拉低事件级"),
        ("span 重叠", f"train {tr['overlap_arg_pairs']} 对、dev {dv['overlap_arg_pairs']} 对",
         "地板值", "BIO 投影必然产生约 1.9% 非法，无法归零"),
        ("重复", f"train 同文本多出 {tr['dup_text_extra_rows']} 行；dev 同文本 {dv['dup_text_extra_rows']} 行",
         "轻微", "同 id 多行是公告多句，不是错误"),
        ("train/dev 泄漏", f"完全相同文本 {leak.get('exact_text_overlap', 0)} 句"
                            f"（{leak.get('overlap_rate_in_dev', 0):.1%}）", "轻微", "同文不同类型 0 句"),
    ]
    add_table(doc, ["检查项", "实测", "级别", "说明"], rows,
              widths=[3.0, 6.4, 1.8, 5.0], font_size=8.3, align_left_cols=(0, 1, 3))
    add_caption(doc, "表 2-2  数据体检：没有阻断训练的硬伤，但有 5 类需要写进实验假设的偏差")

    add_para(doc, "对 sample 与全量 dev 的事件类型分布做 TV 距离，得到 0.053。"
                  "这说明 sample 不是完全随机的小全量：股份回购、亏损、中标占比略高，"
                  "因此 sample 上的绝对分数会比 PPT 全量 930 句略高或略低，"
                  "只能用于方法排序和链路验证，不能直接替换全量指标。")

    doc.add_heading("2.4  数据是否“都有用”", level=2)
    add_bullet(doc, "训练句级抽取、做阈值/早停、跑主评测——只应使用 train_extract / dev_extract。", "直接可用：")
    add_bullet(doc, "train_detect / dev_detect 用于两段式的检测器与系统层“先判有没有事件”，"
                    "正样本率 22.1%/22.2%，负样本充足，是两段式召回瓶颈的来源。", "间接可用：")
    add_bullet(doc, "dev_docs 有 50 篇、74 个事件，32% 的篇是多事件（最多 6 个），"
                    "适合演示篇章事件流，不适合算句级 F1。", "仅系统层：")
    add_bullet(doc, "event_schema 是构标签的必需文件；sample_stats 只记录抽样来源，不参与训练。", "元数据：")

    # ── 3 方法 ────────────────────────────────────────────
    doc.add_heading("3  十种方法与实现", level=1)
    add_para(doc, "所有方法共用 common/evaluate_events.py 的同一套评测：trigger 只看触发词文本，"
                  "argument 看（角色, 论元文本）二元组，event_exact 要求类型、触发词、论元集合完全一致。"
                  "训练方法都从 train 切 10% 做内层验证选 epoch，dev 只用于最终评测。")

    method_rows = [
        ("L0", "规则（词典+正则）", "触发词词典 + 金额/日期/比例/股数正则 + 公司名词典", "0", "<3 ms"),
        ("L1", "BiLSTM-CRF", "字向量 128 → BiLSTM 256×2 → Linear(149) → CRF", "1,540 s（CPU）", "21 ms"),
        ("L2", "BERT+Linear", "bert-base-chinese → 149 标签逐 token softmax", "531 s", "11 ms"),
        ("L2", "BERT+CRF", "同一编码器 + CRF 全局解码（手写前向/维特比）", "3,112 s", "74 ms"),
        ("L2", "CRF 解码期硬约束", "复用 CRF 权重，非法转移 mask 成 -1e4 后重解码", "0", "40 ms"),
        ("L2", "两段式流水线", "检测器 + 13 分类 + 13 个类型条件论元专家（16 个模型）", "1,377.5 s", "50 ms"),
        ("L3", "本地 Qwen3-0.6B zero-shot", "提示词直接生成事件 JSON，不训练", "推理", "5.3 s"),
        ("L3", "API zero/few-shot", "prompt v2，zero 与 few 两版对比", "按 token", "3.5~3.6 s"),
        ("L4", "LoRA SFT", "Qwen3-0.6B + 9 MB adapter，1 epoch / 3,499 条", "2,327 s", "3.1 s"),
        ("L5", "系统组合", "切句→检测→抽取→规则兜底→校验→篇章合并", "组合", "累加"),
    ]
    add_table(doc, ["层", "方法", "核心思路", "训练成本", "延迟/句"], method_rows,
              widths=[1.1, 3.7, 6.6, 2.8, 2.0], font_size=8.2, align_left_cols=(1, 2))
    add_caption(doc, "表 3-1  方法实现：从零训练到生成式微调，再到系统组合")

    # ── 4 结果 ────────────────────────────────────────────
    doc.add_heading("4  实验结果", level=1)
    doc.add_heading("4.1  课程 PPT 全量参考（dev 930 句）", level=2)
    order = ["rule", "bilstm_crf", "bert_linear_256", "bert_crf_256", "bert_crf_256_hard",
             "pipeline", "local_zero_n60", "api_zero_n200", "api_few_n200", "sft_lora_n200"]
    rows = []
    for k in order:
        name, t, a, e, cost, lat = full[k]
        rows.append((name, f"{t:.4f}", f"{a:.4f}", f"{e:.4f}", cost, lat))
    add_table(doc, ["方法", "触发词 F1", "论元 F1", "事件级 F1", "训练/成本", "延迟"], rows,
              widths=[4.9, 2.1, 2.1, 2.1, 4.0, 1.8], font_size=8.0,
              align_left_cols=(0, 4), highlight_rows=(4, 9))
    add_caption(doc, "表 4-1  PPT 主表复现（高亮：性价比最高的硬约束、精度最高的 LoRA SFT）")

    add_para(doc, "这张表就是用户提供的 PPT 图。需要注意两点：① LLM 行只跑了固定 n=200 子集（本地 0.6B 为 n=60），"
                  "不是全量 930；② 表里的 CRF 延迟 40.92 ms 与项目实测 74.28 ms 存在差异，"
                  "本报告在 4.2 的 sample 复评中以实测为准。")

    doc.add_heading("4.2  在本次 sample 上的同口径复评（n=200；本地 0.6B 为 n=60）", level=2)
    add_para(doc, "LLM 链路的固定 200 条抽样恰好全部落在本次 sample dev 300 句内。"
                  "因此把 10 种方法的已保存预测都投影到这 200 句上重新评分，"
                  "得到严格同批、同口径的对比。这张表与项目 metrics/*_n200.json 的 n200 指标一致；"
                  "早期 method_compare.csv 是另一轮结果（SFT 事件级 0.205 vs 本轮 0.185），以本轮 JSON 为准。")

    ref = [r for r in sample if r["source"] == "reference"]
    ref.sort(key=lambda r: float(r["argument_f1"]))
    rows = []
    for r in ref:
        rows.append((r["display_name"], r["n"], f"{float(r['trigger_f1']):.4f}",
                     f"{float(r['argument_f1']):.4f}", f"{float(r['event_f1']):.4f}"))
    add_table(doc, ["方法", "n", "触发词 F1", "论元 F1", "事件级 F1"], rows,
              widths=[8.6, 1.4, 2.1, 2.1, 2.1], font_size=8.3, align_left_cols=(0,),
              highlight_rows=(len(rows) - 1,))
    add_caption(doc, "表 4-2  sample n=200 同批复评（本地 0.6B 为 n=60；按论元 F1 升序）")

    add_para(doc, "补充：本次还在本机 CPU 上真实重跑了本地 Qwen3-0.6B zero-shot（n=60，独立抽样，"
                  "不是上表同一批句子）：触发词 0.4370 / 论元 0.1321 / 事件级 0.0000，"
                  "28.6 s/句，JSON 合法率 100%。与 GPU 参考（0.4068 / 0.1167 / 0）同一量级，"
                  "差异来自抽样不同与 fp32/bf16 精度，不改变“0.6B zero-shot 只能出结构、内容很弱”的结论。")

    add_figure(doc, FIG / "fig_compare_sample.png", 16.2,
               "图 4-1  sample n=200：触发词、论元、事件级三级 F1")

    add_para(doc, "排序与全量 PPT 一致：LoRA SFT > 硬约束 > BERT+CRF > BERT+Linear > "
                  "API few-shot > API zero-shot > BiLSTM-CRF > 两段式 > 规则 > 本地 0.6B zero-shot。"
                  "API few-shot 的论元（0.5395）高于从零训练的 BiLSTM（0.4633），"
                  "与全量结论一致；但它的绝对精度仍远低于 LoRA。")

    doc.add_heading("4.3  本地重训验证", level=2)
    add_para(doc, "为了确认“不是只复评旧预测”，本次在 sample 上重新训练了规则、BiLSTM-CRF、BERT+Linear。"
                  "受本机算力限制（无 NVIDIA GPU，CPU 训练），BERT+CRF 未重训，"
                  "其指标使用项目已保存的全量 checkpoint 预测复评，并明确标注来源。")
    local_rows = []
    skipped = []
    for name, display in (("rule_sample", "规则（sample 重训）"),
                          ("bilstm_crf_sample", "BiLSTM-CRF（sample 重训，6 epoch）"),
                          ("bert_linear_sample", "BERT+Linear（sample 重训，3 epoch）"),
                          ("bert_crf_sample", "BERT+CRF（sample 重训）"),
                          ("bert_crf_sample_hard", "BERT+CRF+硬约束（sample 重训）")):
        m = local_metric(name)
        if not m:
            continue
        if m["trigger"]["f1"] == 0 and m["argument"]["f1"] == 0:
            skipped.append(display)
            continue
        meta = m.get("meta") or {}
        local_rows.append((display, m["num_samples"], f"{m['trigger']['f1']:.4f}",
                           f"{m['argument']['f1']:.4f}", f"{m['event_exact']['f1']:.4f}",
                           f"{meta.get('train_s', 0):.0f} s" if meta.get("train_s") else "0",
                           f"{meta.get('ms_per_sentence', 0):.1f} ms"))
    if local_rows:
        add_table(doc, ["方法", "n", "触发词 F1", "论元 F1", "事件级 F1", "训练", "延迟"], local_rows,
                  widths=[5.2, 1.2, 2.0, 1.9, 1.9, 2.0, 1.9], font_size=8.0, align_left_cols=(0,))
        add_caption(doc, "表 4-3  sample dev 300 句上的本地重训结果（未收敛的 BERT 不列入）")
        add_para(doc, "本地重训只用于验证链路可跑通与相对排序：样本从 5,747 降到 800，"
                      "CPU 训练没有 cuDNN 调优，绝对分数低于全量参考是预期内的。"
                      "规则与 BiLSTM 的排序和全量参考一致，说明数据与代码没有接错。")
        if skipped:
            add_para(doc, "需要单独说明：" + "、".join(skipped) + " 在 800 句 / 135 步下内层验证 span F1 始终为 0"
                          "（模型退化为全 O）。这不是代码接错，而是低学习率（2e-5）+ 小样本 + 149 标签"
                          "导致步数不足；同批 BiLSTM 用 3e-3 学习率、6 epoch 就能到论元 0.363。"
                          "该结果不参与方法排名，只作为“小数据训练预算”的反例。")
        add_figure(doc, FIG / "fig_training_curves.png", 14.2,
                   "图 4-2  本地重训逐 epoch 的内层验证 span F1：BERT 在 800 句/135 步下未收敛")
    else:
        add_para(doc, "本地重训任务仍在运行，结果将在 outputs/local_run 中补齐。" )

    doc.add_heading("4.4  精度与延迟的权衡", level=2)
    add_figure(doc, FIG / "fig_latency_accuracy.png", 15.6,
               "图 4-3  论元 F1 vs 单句延迟（log 横轴）：规则最快，LoRA 最准，硬约束在拐点上")

    # ── 4.5 F1 变化 ─────────────────────────────────────
    doc.add_heading("4.5  各模型的 F1 变化：每一步到底涨多少", level=2)
    add_para(doc, "把方法按“规则 → BiLSTM → BERT → CRF → 硬约束”排成一条链，"
                  "每一步只改变一个变量，就能把总提升拆成可归因的增量。")
    trans_rows = [
        ("规则 → BiLSTM-CRF", "从零建模", "0.1740 → 0.4634", "+28.9pp"),
        ("BiLSTM → BERT+Linear", "预训练语言模型", "0.4634 → 0.5630", "+10.0pp"),
        ("BERT+Linear → BERT+CRF", "结构化解码", "0.5630 → 0.5815", "+1.9pp"),
        ("BERT+CRF → 解码期硬约束", "合法性约束（0 重训）", "0.5815 → 0.6566", "+7.5pp"),
        ("BERT+CRF → 两段式", "拆细（检测→类型→论元）", "0.5815 → 0.3690", "-21.3pp"),
        ("BERT+CRF → LoRA SFT", "生成式微调", "0.5815 → 0.6971", "+11.6pp"),
        ("规则 → LoRA SFT", "端到端最大跨度", "0.1740 → 0.6971", "+52.3pp"),
    ]
    add_table(doc, ["跃迁", "变化类型", "论元 F1", "增量"], trans_rows,
              widths=[5.4, 5.0, 3.4, 2.4], font_size=8.3, align_left_cols=(0, 1),
              highlight_rows=(3, 6))
    add_caption(doc, "表 4-4  论元 F1 的逐步归因（全量 dev 930；高亮为最划算与最大跨度）")
    add_para(doc, "三个要点：① 最大单步是规则→BiLSTM（+28.9pp），说明“有监督建模”本身比“换更强模型”更关键；"
                  "② CRF 只 +1.9pp，硬约束却 +7.5pp，说明解码约束比训练结构更划算；"
                  "③ 两段式是唯一负增长（-21.3pp），拆细的代价大于收益。")
    add_para(doc, "但指标并非单调：论元最高是 LoRA SFT（0.697），事件级最高却是硬约束（0.218）；"
                  "SFT 的论元高 4.1pp，事件级反而低 3.3pp。"
                  "如果业务按“整条事件入库”考核，应优先硬约束；如果按“论元抽取量”考核，才选 SFT。")

    add_figure(doc, FIG / "fig_f1_progression.png", 16.2,
               "图 4-4  判别式主线与生成式路线的 F1 递进")
    add_figure(doc, FIG / "fig_f1_delta.png", 14.8,
               "图 4-5  相对规则基线的论元/事件级提升（百分点）")

    # ── 5 选型 ────────────────────────────────────────────
    doc.add_heading("5  模型 × 场景选型指南", level=1)
    add_para(doc, "没有“最好的模型”，只有在标注量、算力、延迟、schema 稳定性、输出合法性五个约束下的最优解。"
                  "下表把本任务实测数字映射到具体业务场景。")
    sel_rows = [
        ("零标注冷启动，schema 已定", "API zero-shot（论元 0.522）", "few-shot（0.540，token 1.6×）", "本地 0.6B（0.044~0.117）"),
        ("3,000+ 条标注、单卡 8GB", "LoRA SFT（0.697）", "BERT+CRF+硬约束（0.657）", "两段式（0.369）"),
        ("只有 200~500 条标注", "API few-shot（0.540）", "API zero-shot（0.522）", "从零训 BERT/BiLSTM"),
        ("有 GPU、数千条、追事件级", "BERT+CRF+硬约束（0.218）", "LoRA SFT（0.185）", "Linear（0.156）"),
        ("延迟 <15 ms / 纯 CPU 在线", "BERT+Linear（11 ms）", "BiLSTM-CRF（21 ms）", "BERT+CRF（74 ms）"),
        ("完全无 GPU / 无预训练", "BiLSTM-CRF（0.463）", "规则（0.174）", "BERT（跑不动）"),
        ("可解释/可审计/零训练", "规则（<3 ms）", "规则+模型兜底", "端到端 LLM"),
        ("schema 频繁新增类型", "两段式（0.369）", "类型条件联合模型", "单一大 joint"),
        ("输出必须合法 BIO/JSON", "CRF+硬约束（非法 7.4%）", "受约束解码+校验", "Linear（62%~67% 非法）"),
        ("篇章级多事件", "切句+检测+抽取+合并", "长上下文 LLM", "单句 joint 直接喂整篇"),
        ("高并发、成本极敏感", "规则 / Linear", "蒸馏小模型", "API LLM"),
    ]
    add_table(doc, ["业务场景", "首选", "次选", "不建议"], sel_rows,
              widths=[4.4, 4.6, 4.2, 3.0], font_size=8.0, align_left_cols=(0, 1, 2, 3))
    add_caption(doc, "表 5-1  场景选型矩阵（括号内为全量 dev 论元 F1，特殊标注除外）")

    add_para(doc, "三条反直觉发现：")
    add_bullet(doc, "两段式把 149 标签拆到十几标签，论元反而从 0.5815 跌到 0.3690——"
                    "检测漏 42.7%、类型错 5.5%，误差逐段放大，拆细适合可维护而不是刷分。", "结构更复杂 ≠ 分更高：")
    add_bullet(doc, "Linear 62%~67% 的句子含非法 BIO，CRF 硬约束压到 7.4%；"
                    "论元只涨 1.8pp，但事件级从 0.156 涨到 0.218。", "CRF 的价值主要是合法性：")
    add_bullet(doc, "LoRA SFT 只训 1 epoch / 3,499 条、adapter 9 MB，就比 5,747 条训练的 BERT+CRF "
                    "高 11.7pp 论元——数据到三五千条时优先考虑 LoRA。", "轻量微调打赢全量 BERT：")

    add_callout(doc, "本任务最终建议", [
        "离线批量、追精度：LoRA SFT。",
        "在线服务、要合法结构化：BERT+CRF + 解码期硬约束（0 重训、40 ms）。",
        "极致延迟/降级兜底：规则负责金额/日期/比例，模型负责自由文本论元。",
        "新类型冷启动：API zero-shot 先出标注建议，人工改几百条、扩到千级后再转 LoRA SFT。",
        "篇章多事件：不要用单句模型硬扛，走切句→检测→抽取→合并→校验链路。",
    ])

    # ── 6 误差 ────────────────────────────────────────────
    doc.add_heading("6  误差分析与讨论", level=1)
    doc.add_heading("6.1  规则：触发词命中率高，论元几乎抽不对", level=2)
    add_para(doc, "规则在 dev 上的触发词命中率（整句至少中一个触发词）高达 99.8%，"
                  "但论元 F1 只有 0.174。失败类型中 arg_spurious 占 55.6%、wrong_trigger 23.0%、"
                  "wrong_type 15.2%、arg_missing 4.4%，完全正确仅 1.6%。"
                  "典型错误是把“公司名/标的物”这类自由文本硬切成了错误 span——"
                  "这正是需要模型的地方。")

    doc.add_heading("6.2  两段式：误差逐段放大", level=2)
    add_para(doc, "两段式的检测召回只有 57.25%，意味着 42.75% 的句子在第一步就被判成“无事件”直接丢掉；"
                  "类型分类在 dev_extract 上的准确率 94.52%，再乘上论元专家的错误，"
                  "最终论元 F1 0.3690、事件级 0.0957。拆细的收益是“可定位、可替换”，"
                  "代价是“每一段都要交过路费”。")

    doc.add_heading("6.3  event_exact 为什么这么低", level=2)
    add_para(doc, "事件级要求类型、触发词文本、论元集合完全一致。即使论元 F1 到 0.657，"
                  "事件级也只有 0.218：一句平均 4.4 个论元，只要错一个（多值、边界差一个字、"
                  "角色混淆）就整条判错；再加上 13%~14% 的样本超过 254 字会被截断，"
                  "尾部论元天然丢失。这也是为什么报告必须同时给 trigger / argument / event 三级指标，"
                  "只看 event 会严重低估模型能力。")

    doc.add_heading("6.4  长尾与 schema 覆盖", level=2)
    add_para(doc, "dev 中「被约谈」只有 2 句、「企业破产」8 句，单类 F1 的置信区间很宽；"
                  "「减持部分占所持比例」在训练集一次都没出现，模型只能靠角色名语义泛化。"
                  "如果业务真的关心这些类型，需要补标或做规则兜底，而不是只看 macro 平均值。")

    # ── 7 结论 ────────────────────────────────────────────
    doc.add_heading("7  结论", level=1)
    add_bullet(doc, "同一批数据上，LoRA SFT 精度最高（论元 0.697），BERT+CRF+硬约束性价比最高"
                    "（0 重训、+7.5pp、事件级 0.218），规则最快（<3 ms）但论元最弱（0.174）。", "方法层面：")
    add_bullet(doc, "预训练带来 +11.8pp 论元（BiLSTM 0.463 → BERT 0.581）；"
                    "CRF 的主要收益是合法性而非精度；两段式拆细反而掉 21.3pp。", "技术判断：")
    add_bullet(doc, "7 个文件全部可用，无 span/标签硬伤；需处理长尾、截断、少量重复与 2 句泄漏，"
                    "且不要把 sample 指标直接对 PPT 全量数字。", "数据层面：")
    add_bullet(doc, "离线追精度上 LoRA SFT；在线要合法结构化上 CRF 硬约束；"
                    "延迟敏感上 Linear/规则；篇章多事件走系统组合。", "落地建议：")

    # ── 8 复现 ────────────────────────────────────────────
    doc.add_heading("8  交付物与复现", level=1)
    tree = [
        "week7/",
        "├── data/sample/            用户提供的 7 个数据文件（原样）",
        "├── src/                    完整代码（common/rule/bert/llm/eval/service）",
        "├── reference/              已保存的全量指标与 gold（预测见 outputs/predictions）",
        "├── outputs/predictions/    所有预测 JSONL（disc/llm/local）",
        "├── outputs/                本次评测表、图、数据体检 JSON、sample 复评指标",
        "│   ├── sample_subset.csv   10 种方法的 sample n=200 同批结果",
        "│   ├── data_audit.json     数据体检机器可读结果",
        "│   ├── f1_changes.csv/md   F1 变化与逐步归因",
        "│   └── figures/            7 张报告图",
        "├── report/                 本报告 + 数据体检 + 选型指南",
        "└── README.md               复现命令与结论",
    ]
    for line in tree:
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.left_indent = Cm(0.4)
        r = p.add_run(line)
        set_run_font(r, 9, name_ascii="Menlo", name_east="Songti SC")

    add_para(doc, "核心复现命令（详见 README.md）：")
    for line in [
        "python tools/audit_data.py --data_dir ../data/sample --full_dir <全量目录>",
        "python rule/run_rule.py --data_dir ../data/sample --name rule_sample",
        "python bilstm_crf/train.py --data_dir ../data/sample --device cpu --epochs 6 --lr 3e-3 --warmup_steps 20 --run_name bilstm_crf_sample",
        "python bert/train.py --data_dir ../data/sample --device cpu --decoder linear --bert_path \"$BERT_PATH\" --bert_layers 12 --epochs 3 --run_name bert_linear_sample",
        "python eval/score_subset.py          # 同批复评 10 种方法",
        "python eval/make_report_figures.py   # 出图",
    ]:
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(1)
        p.paragraph_format.left_indent = Cm(0.4)
        r = p.add_run(line)
        set_run_font(r, 8.5, name_ascii="Menlo", name_east="Songti SC", color=RGBColor(0x33, 0x33, 0x33))

    # 结尾说明
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(14)
    r = p.add_run("说明：本报告所有指标均由 src/common/evaluate_events.py 计算；"
                  "本地重训与参考预测在表中已分列标注，未把“复评”写成“重训”。")
    set_run_font(r, 8.5, color=GRAY)

    doc.add_heading("附录 A  指标口径", level=1)
    add_para(doc, "同一套指标是横比的前提。本项目所有方法（含 LLM）都把自己的预测写成同一种 JSONL，"
                  "交给同一个函数计算，禁止各脚本各算一套。")
    metric_rows = [
        ("trigger P/R/F1", "预测触发词文本与金标触发词文本的集合匹配；不看事件类型", "衡量“找没找到触发词”"),
        ("argument P/R/F1", "（角色, 论元文本）二元组的集合匹配；不看边界偏移，只看文本", "最常用的论元级指标"),
        ("event_exact P/R/F1", "类型、触发词文本、论元集合全部一致才算 TP", "最严格，最容易低"),
        ("type_macro_f1", "先按事件类型分组算 event F1，再对类型取宏平均", "抗长尾，适合类型不均衡"),
        ("illegal BIO", "序列中出现 I- 开头、B-/I- 类型不一致等非法转移的比例", "衡量输出合法性"),
    ]
    add_table(doc, ["指标", "定义", "用途"], metric_rows,
              widths=[3.4, 9.2, 3.6], font_size=8.2, align_left_cols=(0, 1, 2))
    add_caption(doc, "表 A-1  三级指标与合法性指标的定义")
    add_para(doc, "两个容易误解的点：① 预测解析失败时保留空事件、不删分母，否则 F1 会虚高；"
                  "② 金标自身因多值角色/span 重叠，投影到 BIO 就有约 1.9% 非法序列，"
                  "所以“非法率 0%”不是合理目标，7.4% 已经接近地板。")

    out = ROOT / "report" / "week7_信息抽取实验报告.docx"
    doc.save(out)
    print("已生成", out)


if __name__ == "__main__":
    build()
