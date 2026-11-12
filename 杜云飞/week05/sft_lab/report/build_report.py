#region 0. 导入库
import base64
import html
import io
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                                   # 无窗口后端，纯出图不弹窗
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))     # 让本脚本能 import common
from common import ART_DIR, load_json, build_prompt
#endregion

#region 1. 常量   【常量:METRICS 指标口径，(键, 名称, 含义, 越大越好?)】
TPL_PATH = Path(__file__).resolve().parent / "template.html"
METRICS = [
    ("ppl",          "answer_ppl",      "只在回答段算的困惑度",           False),
    ("token_acc",    "answer_token_acc", "回答段逐字命中率",              True),
    ("stop_rate",    "stop_rate",       "主动生成 &lt;|im_end|&gt; 的比例", True),
    ("multi_q_rate", "multi_q_rate",    "输出 ≥2 个问号（继续列问题）",   False),
    ("len_in_range", "len_in_range",    "长度落在 6~80 字区间",           True),
    ("char_f1",      "char_f1",         "与参考答案的字符 F1",            True),
    ("rouge_l",      "rouge_l",         "字符级 LCS 的 F1（看语序）",     True),
    ("repeat_rate",  "repeat_rate",     "复读退化",                       False),
    ("echo_rate",    "echo_rate",       "照抄提问",                       False),
]
#endregion

#region 2. 函数：画 loss 曲线并转成 base64   【函数:loss_img(train_log),return str】
def loss_img(log):
    """train_log.json 的 log 字段 → 内嵌 PNG。返回：<img> 标签。"""
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]     # 不设中文会变方块
    plt.rcParams["axes.unicode_minus"] = False
    steps = [r["step"] for r in log["log"]]
    losses = [r["loss"] for r in log["log"]]
    fig, ax = plt.subplots(figsize=(8.2, 3.4), dpi=130)
    ax.plot(steps, losses, color="#2b6cb0", lw=1.8, marker="o", ms=3)             # 主曲线
    spe = log.get("steps_per_epoch", 0)
    for e in range(1, int(log.get("epochs", 0)) + 1):
        if spe:
            ax.axvline(spe * e, color="#c0392b", ls="--", lw=1, alpha=.6)         # epoch 分界
            ax.text(spe * e, max(losses) * .98, f" epoch {e}", color="#c0392b", fontsize=9)
    ax.set_xlabel("step")
    ax.set_ylabel("train loss")
    ax.grid(alpha=.25, ls=":")
    ax.set_title(f"训练 loss：{log['steps']} 步，{log['seconds']}s，{log['s_per_step']}s/步", fontsize=11)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return f'<img alt="loss curve" src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}">'
#endregion

#region 3. 函数：拼 HTML 片段（表格 / 卡片 / 列表）
def embed_norms():
    """读词嵌入表，返回三个关键 token 的行范数。返回：dict(im_end, normal, end)。"""
    from safetensors import safe_open
    from common import MODEL_DIR
    with safe_open(str(MODEL_DIR / "model.safetensors"), framework="pt") as f:
        key = [k for k in f.keys() if "embed_tokens" in k][0]
        w = f.get_tensor(key).float()
    n = w.norm(dim=1)
    return {"im_end": n[151645].item(), "start": n[151644].item(), "end": n[151643].item(),
            "normal": n.median().item()}

def eos_story():
    """终止符踩坑记录。返回：html 片段。"""
    v = embed_norms()
    return (f"<b>第一版把回答结尾写成 <code>&lt;|im_end|&gt;</code>，训练 2 个 epoch 后 stop_rate 仍然是 0</b>"
            f"——内容答对了，但停不下来，一直在复读。做了一次过拟合实验：只拿 32 条样本训 15 个 epoch，"
            f"其余 token 的 loss 全降到 0，唯独 <code>&lt;|im_end|&gt;</code> 卡在 7.8 不动。"
            f"查词嵌入才明白：它在 Base 词表里几乎是空行（行范数 <b>{v['im_end']:.2f}</b>，"
            f"普通汉字约 <b>{v['normal']:.2f}</b>，<code>&lt;|endoftext|&gt;</code> 是 <b>{v['end']:.2f}</b>），"
            f"而 Qwen3 的输出头与词嵌入共享权重（tied），这个 token 的 logit 几乎推不动。"
            f"换成 Base 本来就会在句末预测的 <code>&lt;|endoftext|&gt;</code> 之后，一个 epoch 就学会了。"
            f"<br>——这正是课件第 18 页那句话的另一面：<b>SFT 改的是「怎么答」，改不了模型压根没见过的符号。</b>")

def fmt(v, key):
    """按指标类型格式化数值。返回：str。"""
    return f"{v:.2f}" if key == "ppl" else f"{v * 100:.1f}%" if key != "avg_len" else f"{v:.0f}"

def kpi_cards(base, lora):
    """三张关键指标卡。返回：html 片段。"""
    cards, items = [], [("ppl", "回答段困惑度 PPL", "越低说明越会写这种短答案"),
                        ("char_f1", "与参考答案的字符 F1", "衡量内容重合，不是正确率"),
                        ("stop_rate", "主动收尾率", "会不会在说完时停下")]
    for key, name, hint in items:
        b, l = base[key], lora[key]
        if key == "ppl":                                     # 困惑度看相对变化
            delta = f"{'↓' if l < b else '↑'} {abs(l - b) / b * 100:.0f}%"
        else:                                                # 比例型指标看百分点，避免 1408% 这种噪声
            delta = f"{'↑' if l > b else '↓'} {abs(l - b) * 100:.0f} 个百分点"
        cards.append(f'<div class="kpi"><div class="name">{name}</div>'
                     f'<div class="val">{fmt(b, key)} <span style="color:#aaa;font-size:15px">→</span> {fmt(l, key)}</div>'
                     f'<div class="delta">{hint}；{delta}</div></div>')
    return "".join(cards)

def metrics_table(base, lora):
    """9 行指标对比表。返回：html 片段。"""
    rows = ['<table><tr><th style="width:150px">指标</th><th>含义</th>'
            '<th class="num" style="width:88px">base</th><th class="num" style="width:88px">LoRA</th>'
            '<th class="num" style="width:110px">变化</th></tr>']
    for key, name, desc, high_better in METRICS:
        b, l = base[key], lora[key]
        good = (l > b) if high_better else (l < b)
        cls = "up" if good else "down"                    # 红=变好，绿=变坏（A 股习惯）
        sign = "+" if l >= b else ""
        rows.append(f'<tr><td><code>{name}</code></td><td>{desc}</td>'
                    f'<td class="num">{fmt(b, key)}</td><td class="num">{fmt(l, key)}</td>'
                    f'<td class="num {cls}">{sign}{fmt(l - b, key) if key != "ppl" else f"{l - b:+.2f}"}</td></tr>')
    rows.append("</table>")
    return "".join(rows)

def samples_table(samples):
    """10 组问答对照。返回：html 片段。"""
    rows = ['<table><tr><th style="width:32%">提问</th><th style="width:22%">参考答案</th>'
            '<th style="width:23%">微调前 base</th><th style="width:23%">微调后 LoRA</th></tr>']
    for s in samples:
        rows.append(f'<tr><td>{html.escape(s["instruction"])}</td>'
                    f'<td>{html.escape(s["reference"])}</td>'
                    f'<td class="base">{html.escape(s["base"]) or "（空）"}</td>'
                    f'<td class="lora">{html.escape(s["lora"]) or "（空）"}</td></tr>')
    rows.append("</table>")
    return "".join(rows)

def kv_list(pairs):
    """键值对渲染成 <ul>。返回：html 片段。"""
    return "<ul>" + "".join(f"<li>{k}：<b>{v}</b></li>" for k, v in pairs) + "</ul>"
#endregion

#region 4. 函数：生成 markdown 版（方便直接贴作业）
def to_markdown(d):
    """把同一份数据写成 md。返回：str。"""
    b, l = d["eval"]["base"], d["eval"]["lora"]
    lines = [f"# 把「续写器」调成「问答助手」——Qwen3-0.6B-Base LoRA SFT 实验报告", "",
             f"- 数据：课程 sft_train.jsonl 3000 条训练 / {d['eval']['n']} 条验证",
             f"- 训练：{d['train']['steps']} 步，{d['train']['seconds']}s，可训练参数 "
             f"{d['train']['trainable_params']:,}（{d['train']['trainable_params'] / d['train']['all_params']:.2%}）", "",
             "## 指标对比", "", "| 指标 | 含义 | base | LoRA | 变化 |", "|---|---|---|---|---|"]
    for key, name, desc, high_better in METRICS:
        lines.append(f"| `{name}` | {desc.replace('&lt;', '<').replace('&gt;', '>')} | {fmt(b[key], key)} | "
                     f"{fmt(l[key], key)} | {fmt(l[key] - b[key], key)} |")
    lines += ["", f"> 随机配对基线 char_f1 = {d['eval']['random_baseline']:.3f}，任何低于它的相似度都没有意义。", "",
              "## 抽样对照", ""]
    for i, s in enumerate(d["eval"]["samples"], 1):
        lines += [f"**{i}. {s['instruction']}**", f"- 参考：{s['reference']}",
                  f"- base：{s['base'] or '（空）'}", f"- LoRA：{s['lora'] or '（空）'}", ""]
    return "\n".join(lines)
#endregion

#region 5. 主流程：读产物 → 渲染 → 落盘
def main():
    train = load_json(ART_DIR / "train_log.json")
    stats = load_json(ART_DIR / "data_stats.json")
    ev = load_json(ART_DIR / "eval.json")
    mask = (ART_DIR / "mask_check.txt").read_text(encoding="utf-8")[:900]
    base, lora = ev["base"], ev["lora"]

    import torch
    env = kv_list([("GPU", torch.cuda.get_device_name(0)),
                   ("CUDA / torch", f"{torch.version.cuda} / {torch.__version__}"),
                   ("显存", f"{torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB"),
                   ("精度", "bf16")])
    data = kv_list([("训练条数", f"{stats['train']:,}"),
                    ("验证条数", stats["val"]),
                    ("token 长度 P50/P95/max", f"{stats['token_len']['p50']}/{stats['token_len']['p95']}/{stats['token_len']['max']}"),
                    ("被截断条数", stats["truncated"])])
    cfg = kv_list([("epoch", train["epochs"]), ("batch × 累积", f"{train['batch']} × {train['accum']}"),
                   ("学习率", train["lr"]), ("max_len", train["max_len"]),
                   ("可训练参数", f"{train['trainable_params']:,} ({train['trainable_params'] / train['all_params']:.2%})"),
                   ("总耗时", f"{train['seconds']}s（{train['s_per_step']}s/步）")])
    lora_rows = kv_list([("r / alpha", "8 / 16"), ("可训练参数", f"{train['trainable_params']:,}"),
                         ("占全模型", f"{train['trainable_params'] / train['all_params']:.2%}"),
                         ("adapter 体积", "约 9 MB")])

    tpl = TPL_PATH.read_text(encoding="utf-8")
    mapping = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M"),
        "kpi_cards": kpi_cards(base, lora),
        "chatml_evidence": html.escape(build_prompt("什么是人工智能？")),
        "mask_snippet": html.escape(mask),
        "lora_rows": lora_rows,
        "loss_img": loss_img(train),
        "eos_story": eos_story(),
        "train_cfg": cfg,
        "baseline_note": (f"随机配对基线 char_f1 = <b>{ev['random_baseline']:.3f}</b>："
                          "把预测和参考答案打乱重配，得到的平均相似度。低于它说明「纯噪声」，高于它才算学到了内容。"),
        "metrics_table": metrics_table(base, lora),
        "samples_table": samples_table(ev["samples"]),
        "env_rows": env, "data_rows": data, "epochs": train["epochs"],
    }
    out = tpl
    for k, v in mapping.items():
        out = out.replace("{{" + k + "}}", str(v))
    (ART_DIR / "report.html").write_text(out, encoding="utf-8")
    (ART_DIR / "report.md").write_text(to_markdown({"train": train, "eval": ev}), encoding="utf-8")
    print(f"报告已生成: {ART_DIR / 'report.html'} / report.md")
    return 0
#endregion

if __name__ == "__main__":
    sys.exit(main())
