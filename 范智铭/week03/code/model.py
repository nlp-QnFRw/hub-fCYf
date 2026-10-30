# 模型定义
import torch
import torch.nn as nn
from config import (MODEL_TYPE, EMBED_DIM, HIDDEN_SIZE, DROPOUT, NUM_LAYERS,NUM_CLASSES, VOCAB_SIZE,PAD_ID)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # 双向的 2 倍是输出维度
        out_size = HIDDEN_SIZE * (2 if MODEL_TYPE == 'bilstm' else 1)
        self.embedding = nn.Embedding(VOCAB_SIZE, EMBED_DIM,padding_idx=PAD_ID)
        # 根据开关选择RNN/LSTM/BiSTM
        if MODEL_TYPE == 'rnn':
            self.rnn = nn.RNN(EMBED_DIM, HIDDEN_SIZE, num_layers=NUM_LAYERS,dropout=DROPOUT, batch_first=True)
        elif MODEL_TYPE == 'lstm':
            self.rnn = nn.LSTM(EMBED_DIM, HIDDEN_SIZE, num_layers=NUM_LAYERS,dropout=DROPOUT, batch_first=True)
        elif MODEL_TYPE == 'bilstm':
            self.rnn = nn.LSTM(EMBED_DIM, HIDDEN_SIZE, num_layers=NUM_LAYERS,dropout=DROPOUT, batch_first=True,bidirectional=True)
        # 定义归一化层LayerNorm
        self.batch_norm = nn.LayerNorm(out_size)
        # 定义dropout层
        self.dropout = nn.Dropout(DROPOUT)
        # 定义分类头
        self.fc = nn.Linear(out_size, NUM_CLASSES)
    
    def forward(self, x):
        mask = (x != PAD_ID).unsqueeze(-1).float()       # [B, T, 1]
        # 嵌入层
        x = self.embedding(x)
        # RNN/LSTM/BiRNN
        x, _ = self.rnn(x)
        # 归一化层
        x = self.batch_norm(x)
        x = (x * mask).sum(1) / mask.sum(1).clamp(min=1)  # [B, H*] 压掉时间维
        # dropout层
        x = self.dropout(x)
        # 分类头
        return self.fc(x)
