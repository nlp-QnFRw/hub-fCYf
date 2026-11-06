import torch
import torch.nn as nn
from tokenizer import CharTokenizer
from transformer_block import TransformerBlock

class MyModule(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.tok = CharTokenizer.from_file(cfg)
        emb_len = len(self.tok)
        self.embedding = nn.Embedding(emb_len, cfg.transformer.model_size, padding_idx=self.tok.pad_id)
        self.pos_embedding = nn.Embedding(cfg.hparams.max_len, cfg.transformer.model_size)
        self.emb_dropout = nn.Dropout(cfg.transformer.dropout)
        #调用transformer_block并按配置叠加
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg.hparams.num_layers)]
        )
        self.output = nn.Linear(cfg.transformer.model_size, emb_len)

    def forward(self, x):
        T = x.shape[1]
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        x = self.embedding(x) + self.pos_embedding(pos)
        x = self.emb_dropout(x)
        for block in self.transformer_blocks:
            x = block(x)
        x = self.output(x)
        return x
    