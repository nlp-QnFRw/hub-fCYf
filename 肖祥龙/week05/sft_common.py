"""SFT 共用的数据编码、模型加载和生成逻辑，复用 week04 的 LM。"""

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "week04"))
from language_model import LM


SPECIAL_TOKENS = {
    "user": "<|user|>",
    "assistant": "<|assistant|>",
    "end": "<|end|>",
    "pad": "<|pad|>",
    "unk": "<|unk|>",
}
IGNORE_INDEX = -100


@dataclass(frozen=True)
class Sample:
    instruction: str
    output: str


def make_tagged_text(instruction, output):
    """将非空指令和回答转换为带角色标识的文本，返回字符串。"""
    for name, value in (("instruction", instruction), ("output", output)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} 必须是非空字符串")
        if any(token in value for token in SPECIAL_TOKENS.values()):
            raise ValueError(f"{name} 中包含保留的特殊标识")
    return (
        SPECIAL_TOKENS["user"] + instruction
        + SPECIAL_TOKENS["assistant"] + output + SPECIAL_TOKENS["end"]
    )


def parse_tagged_text(text):
    """解析一条带标识的文本；格式错误时抛出 ValueError，成功返回 Sample。"""
    if not isinstance(text, str):
        raise ValueError("text 必须是字符串")
    user, assistant, end = (SPECIAL_TOKENS[k] for k in ("user", "assistant", "end"))
    if not text.startswith(user) or not text.endswith(end):
        raise ValueError("text 必须以 <|user|> 开始、以 <|end|> 结束")
    if any(text.count(token) != 1 for token in (user, assistant, end)):
        raise ValueError("三个角色标识必须各出现一次")
    body = text[len(user):-len(end)]
    instruction, output = body.split(assistant)
    if make_tagged_text(instruction, output) != text:
        raise ValueError("角色标识顺序不正确")
    return Sample(instruction, output)


def read_jsonl(path):
    """读取 JSONL；错误信息包含文件名和物理行号，返回记录列表。"""
    rows = []
    with Path(path).open(encoding="utf-8-sig") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: JSON 格式错误") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: 每行必须是 JSON 对象")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: 数据为空")
    return rows


def load_tagged(path):
    """只读取 tagged 文件的 text 字段，返回解析后的样本列表。"""
    samples = []
    for row_no, row in enumerate(read_jsonl(path), 1):
        try:
            samples.append(parse_tagged_text(row.get("text")))
        except ValueError as exc:
            raise ValueError(f"{path}: 第 {row_no} 条记录: {exc}") from exc
    return samples


def write_json(path, value):
    """将可序列化对象写入 JSON，并原子替换目标文件。"""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def extend_vocab(base_vocab, train_samples):
    """保留原字符编号，仅依据训练集追加新字符和独立的特殊 token。"""
    vocab = dict(base_vocab)
    if sorted(vocab.values()) != list(range(len(vocab))):
        raise ValueError("原词表编号必须从 0 连续递增")
    chars = set("".join(s.instruction + s.output for s in train_samples))
    added_chars = sorted(chars.difference(vocab))
    for token in added_chars + list(SPECIAL_TOKENS.values()):
        if token not in vocab:
            vocab[token] = len(vocab)
    return vocab, added_chars


def make_model(cfg, vocab_size):
    """根据检查点配置创建 week04 的 LM，返回尚未加载权重的模型。"""
    return LM(
        vocab_size=vocab_size, embed_dim=cfg["embed_dim"],
        hidden_dim=cfg["hidden_dim"], num_layers=cfg["num_layers"],
        model_type=cfg["model"], dropout=cfg["dropout"],
        nhead=cfg.get("nhead", 8), max_seq_len=cfg["seq_len"],
    )


def initialize_sft_model(checkpoint, vocab, max_seq_len, dropout):
    """扩展词表和位置编码；继承原参数，新位置循环复制旧位置，返回模型及配置。"""
    cfg = dict(checkpoint["args"])
    if cfg["model"] != "transformer":
        raise ValueError("本次微调要求 Transformer 检查点")
    old_positions = checkpoint["model_state"]["position_embed.weight"].size(0)
    if max_seq_len < old_positions:
        raise ValueError("上下文长度不能小于原检查点的位置编码长度")
    cfg.update(seq_len=max_seq_len, dropout=dropout, max_vocab=len(vocab))
    model = make_model(cfg, len(vocab))
    old_state, new_state = checkpoint["model_state"], model.state_dict()
    if old_state.keys() != new_state.keys():
        raise ValueError("原检查点参数名称与 LM 结构不一致")
    with torch.no_grad():
        for name, old in old_state.items():
            target = new_state[name]
            if name in {"embed.weight", "fc.weight", "fc.bias"}:
                target[:old.size(0)].copy_(old)
            elif name == "position_embed.weight":
                positions = torch.arange(max_seq_len) % old_positions
                target.copy_(old[positions])
            else:
                target.copy_(old)
    return model, cfg


def encode_chars(text, vocab):
    """逐字符编码正文；未登录字符映射到独立的未知 token。"""
    return [vocab.get(char, vocab[SPECIAL_TOKENS["unk"]]) for char in text]


class SFTDataset(Dataset):
    """按样本构造移位后的输入和标签，只监督回答与结束 token。"""

    def __init__(self, samples, vocab, max_seq_len):
        self.items = []
        self.unknown_instruction = 0
        self.unknown_output = 0
        self.max_input_tokens = 0
        self.supervised_tokens = 0
        unk = vocab[SPECIAL_TOKENS["unk"]]
        for row_no, sample in enumerate(samples, 1):
            instruction = encode_chars(sample.instruction, vocab)
            output = encode_chars(sample.output, vocab)
            self.unknown_instruction += instruction.count(unk)
            self.unknown_output += output.count(unk)
            prompt = [vocab[SPECIAL_TOKENS["user"]]] + instruction
            prompt.append(vocab[SPECIAL_TOKENS["assistant"]])
            answer = output + [vocab[SPECIAL_TOKENS["end"]]]
            ids = prompt + answer
            if len(ids) - 1 > max_seq_len:
                raise ValueError(
                    f"第 {row_no} 条样本输入长 {len(ids) - 1}，超过 {max_seq_len}；未做静默截断"
                )
            # assistant 所在位置预测回答首字；回答最后一字所在位置预测结束 token。
            labels = [IGNORE_INDEX] * (len(prompt) - 1) + answer
            self.items.append((torch.tensor(ids[:-1]), torch.tensor(labels)))
            self.max_input_tokens = max(self.max_input_tokens, len(ids) - 1)
            self.supervised_tokens += len(answer)
        if not self.items:
            raise ValueError("SFT 数据集不能为空")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]

    def stats(self):
        return {
            "samples": len(self), "max_input_tokens": self.max_input_tokens,
            "supervised_tokens": self.supervised_tokens,
            "unknown_instruction_tokens": self.unknown_instruction,
            "unknown_output_tokens": self.unknown_output,
        }


def collate_batch(batch, pad_id):
    """按当前批次最大长度右侧补齐；因果注意力确保补齐不影响正文位置。"""
    inputs, labels = zip(*batch)
    return (
        pad_sequence(inputs, batch_first=True, padding_value=pad_id),
        pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX),
    )


def perplexity(loss):
    """将平均损失转换为困惑度，对浮点溢出抛出异常。"""
    return math.exp(loss)


def save_checkpoint(path, model, vocab, cfg, sft_config, metrics):
    """保存兼容 week04 字段的检查点，附带特殊 token 配置和训练指标。"""
    path = Path(path)
    payload = {
        "model_state": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "char2idx": vocab,
        "idx2char": {index: token for token, index in vocab.items()},
        "args": cfg,
        "special_tokens": dict(SPECIAL_TOKENS),
        "special_token_ids": {name: vocab[token] for name, token in SPECIAL_TOKENS.items()},
        "sft_config": sft_config,
        "metrics": metrics,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_sft_checkpoint(path, device):
    """严格加载 SFT 检查点，返回评估模式模型和原始检查点。"""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("special_tokens") != SPECIAL_TOKENS:
        raise ValueError("该检查点缺少兼容的 SFT 特殊 token 配置")
    vocab = checkpoint["char2idx"]
    if checkpoint.get("special_token_ids") != {k: vocab[t] for k, t in SPECIAL_TOKENS.items()}:
        raise ValueError("特殊 token 编号与检查点词表不一致")
    model = make_model(checkpoint["args"], len(vocab))
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model.to(device).eval(), checkpoint


@torch.inference_mode()
def generate(model, vocab, instruction, device, max_new_tokens=64):
    """按统一角色模板贪心生成，返回正文、是否正常结束及生成 token 数。"""
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("问题不能为空")
    if any(token in instruction for token in SPECIAL_TOKENS.values()):
        raise ValueError("问题不能包含保留的特殊标识")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens 必须大于 0")
    ids = [vocab[SPECIAL_TOKENS["user"]]] + encode_chars(instruction, vocab)
    ids.append(vocab[SPECIAL_TOKENS["assistant"]])
    if len(ids) > model.max_seq_len:
        raise ValueError(f"问题包含角色标识后有 {len(ids)} 个 token，超过 {model.max_seq_len}")
    reverse = {index: token for token, index in vocab.items()}
    forbidden = [vocab[SPECIAL_TOKENS[k]] for k in ("user", "assistant", "pad", "unk")]
    result = []
    model.eval()
    for step in range(1, max_new_tokens + 1):
        context = torch.tensor([ids[-model.max_seq_len:]], dtype=torch.long, device=device)
        logits = model(context)[0, -1].clone()
        logits[forbidden] = -torch.inf
        token_id = int(logits.argmax().item())
        if token_id == vocab[SPECIAL_TOKENS["end"]]:
            return {"text": "".join(result), "ended": True, "generated_tokens": step}
        result.append(reverse[token_id])
        ids.append(token_id)
    return {"text": "".join(result), "ended": False, "generated_tokens": max_new_tokens}
