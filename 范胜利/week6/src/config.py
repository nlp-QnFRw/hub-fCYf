# -*- coding: utf-8 -*-
"""
config.py —— 两个基座共用的**唯一一份**训练配置

【这次作业的核心要求是「同样的配置」】
    对比实验最容易犯的错不是模型选错，而是隐性配置不一致：
    一个跑 64 长度、另一个跑 128；一个开了类别加权、另一个忘了开；
    一个用 seed 42、另一个用默认随机种子。
    这些差异每个都能带来 1~3 个点的波动，足以盖过两个基座之间的真实差距。

    做法：把配置写成 dataclass，训练脚本只接受「基座名字」这一个必填参数，
    其余全部来自这里；每个 run 结束时把 config 原样写进 metrics json，
    对比脚本会先断言两份 config 完全相等，不相等直接报错退出。
    → 结论里的差距可以放心归因到基座本身。

【两个基座为什么可比】
    bert-base-chinese：12 层 / 768 隐层 / 12 头 / 21128 词表，109M 参数，中文维基预训练
    chinese-roberta-wwm-ext：同样的 12 层 768 结构、同样 21128 词表，105M 参数，
        但预训练语料更大（约 5.4B tokens）、用 RoBERTa 训练策略 + 全词掩码（WWM）
    结构和词表几乎一样 → 差异主要来自「预训练数据与目标」，不是模型容量，
    这正是做基座对比想要的干净条件。
"""

import os
import platform
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

# ── 国内直连 huggingface.co 常常超时，统一走镜像；已在别处设过就尊重原值 ──
# 必须在 import huggingface_hub / transformers 之前设置才生效，所以放在 config 顶部。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

SRC_DIR = Path(__file__).resolve().parent
PROJ_DIR = SRC_DIR.parent
DATA_DIR = PROJ_DIR / "data"
# 允许用环境变量指向别处的模型目录（离线拷贝、或用小模型跑通链路时用）
MODEL_DIR = Path(os.environ.get("WEEK6_MODEL_DIR", str(PROJ_DIR / "models")))
OUT_DIR = PROJ_DIR / "outputs"
FIG_DIR = OUT_DIR / "figures"

# ── 两个待对比的基座 ──
BACKBONES = {
    "bert-base-chinese": {
        "repo": "bert-base-chinese",
        "dir": "bert-base-chinese",
        "display": "BERT-base-chinese",
        "note": "谷歌官方中文 BERT，12 层 768 隐层，中文维基百科预训练，MLM + NSP",
    },
    "chinese-roberta-wwm-ext": {
        "repo": "hfl/chinese-roberta-wwm-ext",
        "dir": "chinese-roberta-wwm-ext",
        "display": "RoBERTa-wwm-ext",
        "note": "哈工大讯飞，同结构；RoBERTa 训练策略 + 全词掩码，语料规模约为前者 10 倍",
    },
}


@dataclass
class TrainConfig:
    """一次训练的全部超参。两个基座逐字段完全相同，只有基座目录不同。"""

    # —— 输入 ——
    max_length: int = 64          # 训练集最长 145 字，64 覆盖绝大多数标题；两个基座同一截断口径
    # —— 规模 ——（本机只有 CPU，见 README「实验规模」；GPU 上把 train_size 设 0 即用全量 53,360 条）
    train_size: int = 6000        # 按 15 类比例分层抽样；0 = 用全部训练数据
    # —— 优化 ——
    batch_size: int = 64
    epochs: int = 2
    lr: float = 2e-5              # 预训练权重用小 lr，避免冲掉语言知识
    head_lr_mult: float = 5.0     # 分类头随机初始化，需要比底座学得快
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1     # 前 10% 步数线性升温，缓解初期大梯度
    # —— 结构 ——
    pool: str = "cls"             # [CLS] 池化；mean/max 作为可切换项保留
    dropout: float = 0.1
    # —— 损失 ——
    class_weight: bool = True     # 23 倍不均衡，不加权尾部类 recall 会趋近 0
    label_smoothing: float = 0.0
    # —— 复现 ——
    seed: int = 42
    device: str = "auto"          # auto → mps > cuda > cpu

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TrainConfig":
        """只取认识的字段，避免旧 json 里多出来的键把构造搞崩"""
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


# 对比脚本要检查的字段：这些不一致 → 实验不可比
FAIRNESS_KEYS = [
    "max_length", "train_size", "batch_size", "epochs", "lr", "head_lr_mult", "weight_decay",
    "warmup_ratio", "pool", "dropout", "class_weight", "label_smoothing", "seed",
]


def backbone_spec(name: str) -> dict:
    if name not in BACKBONES:
        raise KeyError(f"未知基座 {name}，可选：{list(BACKBONES)}")
    spec = BACKBONES[name]
    return {"name": name, **spec, "local_dir": MODEL_DIR / spec["dir"]}


def ensure_backbone(name: str) -> Path:
    """
    返回本地模型目录，缺文件时自动下载（走 HF_ENDPOINT 镜像）。

    为什么先查本地：课堂/离线环境经常下到一半断掉，留下一个没有权重的空目录；
    直接 from_pretrained 会抛一个很长的 config 错误，看不出真正原因。
    这里显式检查「config.json + 权重」两个条件，缺了就重下。
    """
    spec = backbone_spec(name)
    d: Path = spec["local_dir"]
    has_cfg = (d / "config.json").exists()
    has_w = (d / "pytorch_model.bin").exists() or (d / "model.safetensors").exists()
    if has_cfg and has_w:
        _ensure_safetensors(d)
        return d

    print(f"[模型] 本地缺少 {name}，尝试从 {os.environ['HF_ENDPOINT']} 下载 …", flush=True)
    d.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=spec["repo"],
            local_dir=str(d),
            allow_patterns=["*.json", "*.txt", "*.bin", "*.safetensors", "*.model"],
        )
    except Exception as e:  # noqa: BLE001 —— 镜像/代理的各种异常都收敛到同一句提示
        # huggingface_hub 走的是「先 HEAD 拿大小再下」的流程，镜像在 xet/LFS 重定向上
        # 偶尔会失败（本地实测过：同样的 URL 用 curl 能下完，hub 却报 LocalEntryNotFound）。
        # 所以兜底直接按文件名逐个拉，逻辑最简单也最抗折腾。
        print(f"[模型] huggingface_hub 失败({type(e).__name__})，改用直连兜底 …", flush=True)
        _curl_fetch_repo(d, spec["repo"])

    if not ((d / "pytorch_model.bin").exists() or (d / "model.safetensors").exists()):
        raise FileNotFoundError(
            f"{name} 下载后仍没有权重文件。手动下载见 models/README.md，或检查网络/镜像。"
        )
    _ensure_safetensors(d)
    return d


def _curl_fetch_repo(d: Path, repo: str) -> None:
    """
    兜底下载：按固定文件名清单，从 HF_ENDPOINT 镜像逐个取（urllib，支持断点续传）。

    为什么不用 requests：torch/transformers 已经依赖 urllib，少一个依赖少一个坑。
    为什么允许 Range 续传：镜像在 411MB 的权重上偶尔会断，重下整只是几分钟的浪费。
    """
    import urllib.request

    base = os.environ["HF_ENDPOINT"].rstrip("/")
    # required=False 的几个是可选文件，缺了 transformers 也能跑
    plan = [
        ("config.json", True),
        ("vocab.txt", True),
        ("tokenizer_config.json", True),
        ("tokenizer.json", False),
        ("special_tokens_map.json", False),
        ("added_tokens.json", False),
        ("pytorch_model.bin", True),
    ]
    for fname, required in plan:
        ok = _fetch_one(f"{base}/{repo}/resolve/main/{fname}", d / fname, required)
        if ok:
            size_mb = (d / fname).stat().st_size / 1e6
            print(f"  ✓ {fname} ({size_mb:.0f} MB)", flush=True)
        elif required:
            print(f"  ✗ {fname} 下载失败", flush=True)


def _fetch_one(url: str, dst: Path, required: bool = True, max_try: int = 4) -> bool:
    """下单个文件；已存在的部分用 Range 续传，直到大小与服务器声明一致"""
    import urllib.request

    # 先拿总大小（有些镜像不返回 Content-Length，那就只能靠续传后的大小判断）
    total = 0
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as r:
            total = int(r.headers.get("Content-Length") or 0)
    except Exception:  # noqa: BLE001
        pass

    for _ in range(max_try):
        have = dst.stat().st_size if dst.exists() else 0
        if total and have >= total:
            return True
        if total and have > total:      # 上次写坏了，重来
            dst.unlink()
            have = 0
        req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                mode = "ab" if have and resp.status == 206 else "wb"
                with open(dst, mode) as out:
                    while True:
                        chunk = resp.read(1 << 20)
                        if not chunk:
                            break
                        out.write(chunk)
        except Exception:  # noqa: BLE001
            if not required:
                break
            continue
    return dst.exists() and dst.stat().st_size > 0 and (not total or dst.stat().st_size >= total)


def _ensure_safetensors(d: Path) -> None:
    """
    把 pytorch_model.bin 转成 model.safetensors（只转一次）。

    【为什么必须转】
      transformers >= 4.48 出于 CVE-2025-32434 的考虑，要求 torch>=2.6 才允许
      torch.load 解析 .bin。本机 torch 2.2（macOS x86_64 的最后一个版本）直接报
      「require users to upgrade torch to at least v2.6」，训练根本起不来。
      而 .bin 本身能正常 load——限制是 transformers 在加载前做的检查。
      所以这里自己做一次无损格式转换：torch.load 读 .bin → safetensors 落盘，
      transformers 之后只会走 safetensors 分支，既绕开版本限制，又顺带拿到
      safetensors 的安全（不执行反序列化代码）与更快的 mmap 加载。
    """
    st, bd = d / "model.safetensors", d / "pytorch_model.bin"
    if st.exists() or not bd.exists():
        return
    print(f"[模型] 检测到只有 pytorch_model.bin，转换为 safetensors（torch<2.6 的限制）…", flush=True)
    import torch
    from safetensors.torch import save_file

    state = torch.load(bd, map_location="cpu")

    # 预训练权重里存在「共享存储」的张量：MLM 头的 decoder 与 word_embeddings 是同一块内存
    # （tied embedding）。safetensors 不允许两个 key 指向同一块存储，会直接报错。
    # 这里按 data_ptr 去重，只保留第一个出现的 key——被丢掉的都是任务头里的副本，
    # AutoModel 只加载编码器，不需要它们。
    uniq, seen, dropped = {}, set(), []
    for k, v in state.items():
        if not isinstance(v, torch.Tensor):
            continue
        fp = (v.data_ptr(), v.dtype, tuple(v.shape))
        if fp in seen:
            dropped.append(k)
            continue
        seen.add(fp)
        uniq[k] = v.contiguous()

    save_file(uniq, str(st), metadata={"format": "pt", "converted_from": "pytorch_model.bin"})
    print(f"[模型] 已生成 {st}" + (f"（去重共享张量：{dropped}）" if dropped else ""), flush=True)


def resolve_device(pref: str = "auto") -> "torch.device":
    """
    设备选择。

    auto 的规则：CUDA > MPS（仅 Apple Silicon）> CPU。
    为什么 Intel Mac 上主动跳过 MPS：
      本机实测（torch 2.2.2 + transformers 4.55 + macOS 15）在 MPS 上做
      BERT 的反向传播会直接 segfault（Placeholder storage has not been allocated），
      连异常都抛不出来。torch 2.2 是 macOS x86_64 的最后一个官方版本，
      没有升级路径，所以按架构判断直接走 CPU，避免跑到一半崩掉。
    """
    import torch

    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and not (sys.platform == "darwin" and platform.machine() == "x86_64"):
        return torch.device("mps")
    return torch.device("cpu")



