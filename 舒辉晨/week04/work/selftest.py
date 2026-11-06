"""
selftest.py -- 不训练也能验证实现对不对

    python selftest.py

课件里 export_ime_model.py 也导出了自检数据，但没有任何代码去用，
Python 和 JS 是否算得一样其实从来没验证过。这里把「可验证」这件事补上：
每个测试都对应一个容易写错、写错又不容易被发现的地方。

覆盖的点：
    · 因果掩码是否真的挡住了未来（改后面的字，前面的输出必须一字不差）
    · 注意力每行是否归一化、上三角是否真的为 0
    · loss 是否等于手算的交叉熵
    · 权重绑定是否生效
    · 优化器分组是否把 bias/LayerNorm 排除在 weight decay 之外
    · 学习率曲线两端是否符合预期
    · 数据管线：OOV 是否落到 [UNK]、评估窗口是否不重叠、y 是否为 x 右移一位
    · checkpoint 存读往返是否一致
    · 小模型能否过拟合一个 batch（反向传播链路是否通）
"""

from __future__ import annotations

import math
import os
import sys
import tempfile

import torch
import torch.nn.functional as F

import data as data_mod
from model import GPT, GPTConfig

RESULTS: list[tuple[str, str, str]] = []


class Skip(Exception):
    """环境里没有需要的资源（比如没找到 bert-base-chinese）时，标记为跳过而不是失败。"""


def case(name):
    def deco(fn):
        try:
            detail = fn() or ""
            RESULTS.append((name, "PASS", str(detail)))
        except Skip as exc:
            RESULTS.append((name, "SKIP", str(exc)))
        except Exception as exc:                       # noqa: BLE001
            RESULTS.append((name, "FAIL", f"{type(exc).__name__}: {exc}"))
        return fn
    return deco


def tiny(**over) -> GPTConfig:
    kw = dict(vocab_size=64, block_size=32, n_layer=2, n_head=2, n_embd=32, dropout=0.0)
    kw.update(over)
    return GPTConfig(**kw)


# ─────────────────────────── 模型 ───────────────────────────

@case("配置校验：n_embd 不能被 n_head 整除时报错")
def _():
    try:
        GPTConfig(vocab_size=10, n_embd=30, n_head=4)
    except ValueError:
        return "已拒绝"
    raise AssertionError("应该抛 ValueError 却通过了")


@case("权重绑定：输出层与词嵌入是同一份参数")
def _():
    m = GPT(tiny())
    assert m.head.weight is m.tok_emb.weight, "head 与 tok_emb 不是同一个张量"
    n_head_only = m.head.weight.numel()
    assert m.num_params < sum(p.numel() for p in m.parameters()) + n_head_only + 1
    return f"{m.num_params:,} 参数（已去重）"


@case("因果性：改动最后一个 token，前面位置的输出必须完全不变")
def _():
    torch.manual_seed(0)
    m = GPT(tiny()).eval()
    a = torch.randint(0, 64, (1, 12))
    b = a.clone()
    b[0, -1] = (a[0, -1] + 7) % 64
    with torch.no_grad():
        la, _ = m(a)
        lb, _ = m(b)
    diff = (la[:, :-1] - lb[:, :-1]).abs().max().item()
    assert diff == 0.0, f"前面的位置受到了未来 token 影响，最大差值 {diff:g}"
    assert (la[:, -1] - lb[:, -1]).abs().max().item() > 0, "最后一个位置居然没变，测试无效"
    return "掩码正确"


@case("注意力：上三角为 0，每行和为 1")
def _():
    m = GPT(tiny()).eval()
    x = torch.randint(0, 64, (2, 16))
    with torch.no_grad():
        _, _, attns = m(x, return_attn=True)
    w = attns[0]
    assert w.shape == (2, 2, 16, 16), w.shape
    upper = torch.triu(w, diagonal=1).abs().max().item()
    assert upper < 1e-6, f"上三角不为 0（{upper:g}）"
    row_sums = w.sum(-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), \
        f"行和不等 1：{row_sums.min():.6f}~{row_sums.max():.6f}"
    return f"每行和 {row_sums.mean():.6f}"


@case("损失：与手算交叉熵一致")
def _():
    torch.manual_seed(1)
    m = GPT(tiny()).eval()
    x = torch.randint(0, 64, (3, 10))
    y = torch.randint(0, 64, (3, 10))
    with torch.no_grad():
        logits, loss = m(x, y)
        manual = F.cross_entropy(logits.reshape(-1, 64), y.reshape(-1))
    assert abs(loss.item() - manual.item()) < 1e-6, (loss.item(), manual.item())
    return f"loss={loss.item():.4f}"


@case("位置编码：输入超过 block_size 时给出明确报错")
def _():
    m = GPT(tiny(block_size=16)).eval()
    try:
        with torch.no_grad():
            m(torch.zeros(1, 17, dtype=torch.long))
    except ValueError as e:
        assert "block_size" in str(e)
        return "已拒绝"
    raise AssertionError("超长输入没有报错")


@case("训练/推理模式：dropout 关闭后两次前向结果一致")
def _():
    m = GPT(tiny(dropout=0.5)).eval()
    x = torch.randint(0, 64, (1, 8))
    with torch.no_grad():
        a, _ = m(x)
        b, _ = m(x)
    assert torch.equal(a, b), "eval 模式下仍有随机性"
    m.train()
    with torch.no_grad():
        c, _ = m(x)
        d, _ = m(x)
    assert not torch.equal(c, d), "train 模式下 dropout 没起作用"
    return "eval 确定 / train 随机"


# ─────────────────────────── 优化器与调度 ───────────────────────────

@case("优化器：bias 与 LayerNorm 不做 weight decay，且无重复参数")
def _():
    m = GPT(tiny())
    opt = m.configure_optimizers(lr=1e-3, weight_decay=0.1)
    decay_ids = [id(p) for p in opt.param_groups[0]["params"]]
    no_decay_ids = [id(p) for p in opt.param_groups[1]["params"]]
    assert not set(decay_ids) & set(no_decay_ids), "参数同时出现在两个组里"
    assert all(p.dim() >= 2 for p in opt.param_groups[0]["params"]), "衰减组里混进了一维参数"
    assert all(p.dim() < 2 for p in opt.param_groups[1]["params"]), "非衰减组里混进了二维参数"
    n = len(decay_ids) + len(no_decay_ids)
    assert n == sum(1 for _ in m.parameters()), f"覆盖了 {n} 个参数，模型共有 {sum(1 for _ in m.parameters())}"
    return f"衰减 {len(decay_ids)} 个 / 不衰减 {len(no_decay_ids)} 个"


@case("学习率：warmup 线性升到峰值，结束后退到 min_lr")
def _():
    import train as train_mod
    peak = train_mod.lr_at(99, 1e-3, 1e-4, 100, 1000)
    assert abs(peak - 1e-3) < 1e-9, peak
    assert train_mod.lr_at(0, 1e-3, 1e-4, 100, 1000) < train_mod.lr_at(50, 1e-3, 1e-4, 100, 1000)
    assert abs(train_mod.lr_at(1000, 1e-3, 1e-4, 100, 1000) - 1e-4) < 1e-12
    mid = train_mod.lr_at(550, 1e-3, 1e-4, 100, 1000)
    assert 1e-4 < mid < 1e-3, mid
    return f"峰值 {peak:.0e} / 中点 {mid:.2e} / 末尾 1e-4"


# ─────────────────────────── 数据 ───────────────────────────

@case("数据：未登录字映射到 [UNK]，编码解码可逆")
def _():
    stoi = {"[UNK]": 0, "你": 1, "好": 2}
    ids = data_mod.encode("你X好", stoi)
    assert ids.tolist() == [1, 0, 2], ids.tolist()
    assert data_mod.decode([1, 0, 2], ["[UNK]", "你", "好"]) == "你[UNK]好"
    return "OOV → 0"


@case("数据：y 是 x 右移一位")
def _():
    toks = torch.arange(500, dtype=torch.long)
    g = torch.Generator().manual_seed(0)
    x, y = data_mod.get_batch(toks, batch_size=4, block_size=16, generator=g)
    assert x.shape == y.shape == (4, 16), (x.shape, y.shape)
    assert torch.equal(y[:, :-1], x[:, 1:]), "y 不是 x 的左移"
    assert torch.all(y[:, -1] > x[:, -1]), "最后一个目标不是右移来的"
    return "形状与错位正确"


@case("数据：评估窗口不重叠且覆盖完整")
def _():
    toks = torch.arange(1000, dtype=torch.long)
    seen = []
    for x, _ in data_mod.eval_windows(toks, block_size=64, batch_size=8):
        seen.extend(x[:, 0].tolist())
    assert seen == list(range(0, 960, 64)), f"窗口起点不对: {seen[:5]}..."
    return f"{len(seen)} 个窗口，步长 64，无重叠"


# ─────────────────────────── 其他 ───────────────────────────

@case("checkpoint：存读往返后参数完全一致")
def _():
    cfg = tiny()
    m1 = GPT(cfg)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "ck.pt")
        torch.save({"model": m1.state_dict(), "config": cfg.to_dict()}, p)
        ck = torch.load(p, map_location="cpu")
        m2 = GPT(GPTConfig.from_dict(ck["config"]))
        m2.load_state_dict(ck["model"])
    for (n1, p1), (n2, p2) in zip(m1.state_dict().items(), m2.state_dict().items()):
        assert n1 == n2
        assert torch.equal(p1, p2), f"{n1} 不一致"
    return "state_dict 一致"


@case("生成：输出长度正确且 token id 不越界")
def _():
    m = GPT(tiny()).eval()
    idx = torch.randint(0, 64, (1, 5))
    g = torch.Generator().manual_seed(0)
    out = m.generate(idx, 12, temperature=0.9, top_k=10, generator=g)
    assert out.shape == (1, 17), out.shape
    assert torch.equal(out[:, :5], idx), "提示词被改动了"
    assert out.max().item() < 64 and out.min().item() >= 0
    return "长度 5→17"


@case("端到端：小模型能过拟合一个固定 batch（反向传播链路通）")
def _():
    torch.manual_seed(0)
    cfg = tiny(vocab_size=48, block_size=16, n_embd=64, n_layer=2, n_head=2)
    m = GPT(cfg)
    x = torch.randint(0, 48, (8, 16))
    y = torch.randint(0, 48, (8, 16))
    opt = m.configure_optimizers(lr=3e-3, weight_decay=0.0)
    first = last = None
    m.train()
    for i in range(80):
        _, loss = m(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if i == 0:
            first = loss.item()
        last = loss.item()
    assert first is not None and last < first * 0.35, f"loss 只从 {first:.3f} 降到 {last:.3f}"
    return f"loss {first:.3f} → {last:.3f}"


# ─────────────────────────── BERT ───────────────────────────
# 下面这组测试需要本机有 bert-base-chinese（约 420MB）。找不到就整体标记 SKIP，
# 不影响前面那些纯实现的测试。

_BERT_CACHE: dict = {}


def get_bert():
    """只加载一次：119M 参数的模型没必要每个测试都读一遍盘。"""
    if not _BERT_CACHE:
        from bert_data import BertTok
        from bert_model import load_pretrained, resolve_bert_dir

        try:
            d = resolve_bert_dir()
        except FileNotFoundError as exc:
            raise Skip(str(exc).splitlines()[0]) from exc
        model, cfg = load_pretrained(d, device="cpu")
        model.eval()
        _BERT_CACHE.update(dir=d, model=model, cfg=cfg, tok=BertTok(d))
    return _BERT_CACHE


@case("BERT：预训练权重与自建结构逐项对上")
def _():
    c = get_bert()
    sd = torch.load(os.path.join(c["dir"], "pytorch_model.bin"),
                    map_location="cpu", weights_only=True)
    mine = c["model"].state_dict()
    assert len(sd) == len(mine), f"张量个数不同：{len(sd)} vs {len(mine)}"
    missing = [k for k in sd if k not in mine]
    assert not missing, f"这些权重没接上：{missing[:5]}"
    return f"{len(sd)} 个张量全部命中 / {c['model'].num_params:,} 参数"


@case("BERT：注意力是双向的（改最后一个字，第一个字也会变）")
def _():
    c = get_bert()
    tok = c["tok"]
    a = torch.tensor([[tok.cls_id] + tok.encode("中国的首都是北京") + [tok.sep_id]])
    b = a.clone()
    b[0, -2] = tok.encode("上海")[0]        # 把「京」换成「上」
    with torch.no_grad():
        la, _ = c["model"](a)
        lb, _ = c["model"](b)
    d0 = (la[0, 0] - lb[0, 0]).abs().max().item()
    assert d0 > 1e-4, "第一个位置没受影响，说明掩码没去掉、根本不是双向的"
    return f"[CLS] 位置差值 {d0:.4f}（同样的测试在 GPT 上必须是 0）"


@case("BERT：MLM 遮字比例约 15%，且不碰 [CLS]/[SEP]")
def _():
    from bert_data import mask_batch
    c = get_bert()
    tok = c["tok"]
    ids = torch.tensor([[tok.cls_id]
                        + tok.encode("我今天要去北京看一个很重要的会议和展览")
                        + [tok.sep_id]])
    am = torch.ones_like(ids)
    gen = torch.Generator().manual_seed(0)
    total = masked = 0
    for _ in range(200):
        inputs, labels = mask_batch(ids, am, tok, 0.15, gen)
        total += inputs.numel()
        masked += int((labels != -100).sum())
        assert inputs[0, 0] == tok.cls_id and inputs[0, -1] == tok.sep_id, "特殊符号被遮了"
    ratio = masked / total
    assert 0.10 < ratio < 0.20, f"遮字比例 {ratio:.3f} 偏离 0.15 太多"
    return f"实测 {ratio * 100:.1f}%"


@case("BERT：遮字预测能力（前向数值是否算对的功能性证明）")
def _():
    c = get_bert()
    tok, model = c["tok"], c["model"]
    probes = [("中国的首都是北京", "北"), ("我昨天吃了一个很甜的苹果", "苹"),
              ("他把书放在桌子上", "桌"), ("他们在公园里散步", "散")]
    hit = 0
    detail = []
    for text, target in probes:
        toks = tok.tokens(text)
        ids = tok.encode(text)
        pos = toks.index(target)
        ids[pos] = tok.mask_id
        with torch.no_grad():
            logits, _ = model(torch.tensor([[tok.cls_id] + ids + [tok.sep_id]]))
        pred = tok.token(logits[0, pos + 1].argmax().item())
        hit += int(pred == target)
        detail.append(f"{target}->{pred}")
    assert hit >= 3, f"4 句只猜对 {hit} 句（{' '.join(detail)}），前向大概率算错了"
    return f"top-1 命中 {hit}/4：" + " ".join(detail)


@case("BERT：伪困惑度是有限值且量级合理")
def _():
    from bert_model import pseudo_perplexity
    c = get_bert()
    tok = c["tok"]
    seqs = [[tok.cls_id] + tok.encode(s) + [tok.sep_id]
            for s in ("中国的首都是北京", "我昨天吃了一个很甜的苹果")]
    L = max(len(s) for s in seqs)
    ids = torch.zeros(len(seqs), L, dtype=torch.long)
    am = torch.zeros(len(seqs), L, dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s)
        am[i, :len(s)] = 1
    ppl, n = pseudo_perplexity(c["model"], ids, am, tok.mask_id)
    assert ppl == ppl and 1.0 < ppl < 500, f"伪困惑度异常：{ppl}"
    return f"伪PPL {ppl:.2f}（{n} 个位置）"


def main():
    print("=" * 62)
    print(f"selftest  torch {torch.__version__}")
    print("=" * 62)
    passed = sum(1 for _, s, _ in RESULTS if s == "PASS")
    skipped = sum(1 for _, s, _ in RESULTS if s == "SKIP")
    failed = sum(1 for _, s, _ in RESULTS if s == "FAIL")
    for name, status, detail in RESULTS:
        print(f"[{status}] {name}")
        if detail:
            print(f"       {detail}")
    print("-" * 62)
    print(f"{passed}/{len(RESULTS)} 通过" +
          (f"，{skipped} 跳过" if skipped else "") +
          (f"，{failed} 失败" if failed else ""))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
