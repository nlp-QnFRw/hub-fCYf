# 超参数配置
# 数据：12 种语言 / train 9600、dev 1200 / 字符级词表 3441 项 / 最长句 62 字符
import torch
from pathlib import Path

# ---------------------------------------------------------------- 模型
MODEL_TYPE = 'bilstm'      # 'rnn' | 'lstm' | 'bilstm'
EMBED_DIM = 128
HIDDEN_SIZE = 128          # 双向时每侧 128，拼成 256 进分类头
DROPOUT = 0.3
NUM_LAYERS = 2            # 双向时 2 层
# ---------------------------------------------------------------- 路径
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / 'data'
TRAIN_FILE = DATA_DIR / 'train.csv'
DEV_FILE = DATA_DIR / 'dev.csv'
TEST_FILE = DATA_DIR / 'test.csv'
VOCAB_FILE = DATA_DIR / 'vocab.txt'
CKPT_FILE = BASE_DIR / 'checkpoints' / f'{MODEL_TYPE}_best.pt'
# ---------------------------------------------------------------- 数据 / 词表
VOCAB_SIZE = sum(1 for _ in open(VOCAB_FILE, 'r', encoding='utf-8'))  # 动态读取 vocab.txt 行数
NUM_CLASSES = 12           # label 按 sorted(set(label)) 编号
MAX_LEN = 64               # 最长句 62 字符，64 一条都不截断
PAD_ID, UNK_ID = 0, 1
# ---------------------------------------------------------------- 训练
BATCH_SIZE = 64            # 9600 / 64 = 150 step/epoch
EPOCHS = 100               
LR = 1e-3                  # Adam
PATIENCE = 4               # dev 连续 4 轮不升就停
GRAD_CLIP = 5.0            # RNN 必需，防梯度爆炸
LABELS = ['ar', 'en', 'es', 'hi', 'ja', 'ko', 'pt', 'ru', 'th', 'tr', 'vi', 'zh']
# ---------------------------------------------------------------- 运行
SEED = 42
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
