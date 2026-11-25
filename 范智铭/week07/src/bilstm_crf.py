"""BiLSTM + CRF（字符级，无预训练）。

为什么要 CRF：
    BiLSTM 每个位置独立 softmax，会输出 B-PER 后面跟 I-LOC 这类非法转移；
    CRF 用一个转移矩阵把所有合法路径的分数做归一化，解码时直接取全局最优路径。
    在只有 500 条训练样本时，这个约束比多一层网络更值钱。
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CRF(nn.Module):
    def __init__(self, num_tags: int):
        super().__init__()
        self.num_tags = num_tags
        self.START = num_tags
        self.END = num_tags + 1
        # transitions[i, j] = 从 i 转移到 j 的分数
        self.transitions = nn.Parameter(torch.empty(num_tags + 2, num_tags + 2))
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        # @doc ** 禁用转移用 -1e4 而非 -inf：logsumexp 会出 nan
        # 禁止转移到 START、禁止从 END 出发
        with torch.no_grad():
            self.transitions[:, self.START] = -1e4
            self.transitions[self.END, :] = -1e4

    def _score(self, feats: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """整条路径的分子分数（发射 + 转移 + 起止）。feats: [B, L, T]"""
        B, L, _ = feats.shape
        mf = mask.float()
        score = feats.new_zeros(B)
        # 起始转移
        score = score + self.transitions[self.START, tags[:, 0]] + feats[:, 0].gather(
            1, tags[:, 0].unsqueeze(1)).squeeze(1)
        for t in range(1, L):
            m = mf[:, t]
            emit = feats[:, t].gather(1, tags[:, t].unsqueeze(1)).squeeze(1)
            trans = self.transitions[tags[:, t - 1], tags[:, t]]
            score = score + (emit + trans) * m
        # 结束转移：取每条序列最后一个有效位置
        lengths = mask.sum(1).long().clamp(min=1)
        last_tags = tags.gather(1, (lengths - 1).unsqueeze(1)).squeeze(1)
        score = score + self.transitions[last_tags, self.END]
        return score

    def _partition(self, feats: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """log Z：前向算法，逐位置 logsumexp 累积。"""
        # @doc *** 掩码位不更新 alpha，循环结束后 alpha 天然停在最后有效位置
        # 早期版本用 lengths-1 去 gather，结果 index 越界（T vs L 混淆），对拍才抓出来
        B, L, T = feats.shape
        mf = mask.float()
        alpha = self.transitions[self.START, :T].unsqueeze(0) + feats[:, 0]  # [B, T]
        for t in range(1, L):
            m = mf[:, t].unsqueeze(1)  # [B,1]
            # [B, T, 1] + [T, T] -> [B, T, T] -> logsumexp over 前一个状态
            nxt = alpha.unsqueeze(2) + self.transitions[:T, :T].unsqueeze(0)
            new_alpha = torch.logsumexp(nxt, dim=1) + feats[:, t]
            alpha = alpha * (1 - m) + new_alpha * m
        # 掩码位置不更新，所以循环结束后 alpha 就是每条序列「最后一个有效位置」的前向值
        return torch.logsumexp(alpha + self.transitions[:T, self.END].unsqueeze(0), dim=1)

    def neg_log_likelihood(self, feats: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return (self._partition(feats, mask) - self._score(feats, tags, mask)).mean()

    @torch.no_grad()
    def viterbi(self, feats: torch.Tensor, mask: torch.Tensor) -> list[list[int]]:
        B, L, T = feats.shape
        mf = mask.float()
        score = self.transitions[self.START, :T].unsqueeze(0) + feats[:, 0]
        history = []
        for t in range(1, L):
            m = mf[:, t].unsqueeze(1)
            cand = score.unsqueeze(2) + self.transitions[:T, :T].unsqueeze(0)  # [B,T,T]
            best, idx = cand.max(dim=1)                                        # [B,T]
            new_score = best + feats[:, t]
            score = score * (1 - m) + new_score * m
            history.append(idx)
        lengths = mask.sum(1).long().clamp(min=1)
        score = score + self.transitions[:T, self.END].unsqueeze(0)
        best_last = score.argmax(dim=1)
        paths = []
        for b in range(B):
            n = int(lengths[b].item())
            path = [int(best_last[b])]
            for t in range(n - 1, 0, -1):
                path.append(int(history[t - 1][b, path[-1]]))
            paths.append(path[::-1])
        return paths


class BiLSTMCRF(nn.Module):
    def __init__(self, vocab_size: int, num_tags: int, emb_dim: int = 128, hidden: int = 256,
                 num_layers: int = 1, dropout: float = 0.3, use_crf: bool = True):
        super().__init__()
        self.use_crf = use_crf
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden, num_layers=num_layers, bidirectional=True,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden * 2, num_tags)
        self.crf = CRF(num_tags) if use_crf else None

    def emissions(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.dropout(self.embed(input_ids))
        out, _ = self.lstm(x)
        return self.fc(self.dropout(out))

    def loss(self, input_ids, attention_mask, labels) -> torch.Tensor:
        feats = self.emissions(input_ids)
        mask = attention_mask.bool()
        if self.crf is not None:
            safe_tags = labels.clamp(min=0)  # -100 → 0，靠 mask 屏蔽
            return self.crf.neg_log_likelihood(feats, safe_tags, mask)
        return nn.functional.cross_entropy(feats.view(-1, feats.size(-1)), labels.view(-1),
                                           ignore_index=-100)

    @torch.no_grad()
    def decode(self, input_ids, attention_mask) -> list[list[int]]:
        feats = self.emissions(input_ids)
        mask = attention_mask.bool()
        if self.crf is not None:
            return self.crf.viterbi(feats, mask)
        preds = feats.argmax(-1)
        out = []
        for i in range(preds.size(0)):
            n = int(mask[i].sum().item())
            out.append([int(x) for x in preds[i, :n].tolist()])
        return out

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
