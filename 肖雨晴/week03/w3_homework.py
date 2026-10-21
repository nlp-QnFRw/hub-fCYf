"""
w3_homework.py
中文五字句子位置分类 —— RNN 与无 RNN 模型对比

任务：每句恰好 5 个汉字，且“你”只出现一次；在第几位，就属于第几类。
训练标签为 0～4，对外显示为第 1～5 类。
模型 A：Embedding → RNN → MaxPool → Linear(5)
模型 B：Embedding → MaxPool → Linear(5)
优化：Adam (lr=1e-3)   损失：CrossEntropyLoss   无需 GPU，CPU 即可运行

依赖：torch >= 2.0、matplotlib   (pip install torch matplotlib)
运行：python week3/w3_homework.py
输出：终端对比表、预测示例，以及同目录下的 w3_model_comparison.png。
"""

import random
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ─── 超参数 ────────────────────────────────────────────────
SEED        = 42
N_SAMPLES   = 4000
MAXLEN      = 5
NUM_CLASSES = 5
EMBED_DIM   = 64
HIDDEN_DIM  = 64
LR          = 1e-3
BATCH_SIZE  = 64
EPOCHS      = 20
TRAIN_RATIO = 0.8
CURVE_PATH  = Path(__file__).with_name('w3_model_comparison.png')

# ─── 1. 数据生成 ────────────────────────────────────────────
# 其余汉字不包含“你”，保证每句只有一个目标字。
# 随机组字用于学习位置规律，不要求句子语义通顺。
OTHER_CHARS = '我他她们今天真好有开心棒想请来大家都爱学习中文一起去看书吃饭'


def build_dataset(n=N_SAMPLES):
    data = []
    for i in range(n):
        label = i % NUM_CLASSES  # 0～4，五类样本数量尽量均衡
        chars = random.choices(OTHER_CHARS, k=MAXLEN - 1)
        chars.insert(label, '你')
        data.append((''.join(chars), label))
    random.shuffle(data)
    return data


# ─── 2. 词表构建与编码 ──────────────────────────────────────
def build_vocab(data):
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for sent, _ in data:
        for ch in sent:
            if ch not in vocab:
                vocab[ch] = len(vocab)
    return vocab


def encode(sent, vocab):
    if len(sent) != MAXLEN or sent.count('你') != 1:
        raise ValueError('输入必须恰好 5 个字，且只包含一个“你”')
    return [vocab.get(ch, vocab['<UNK>']) for ch in sent]


# ─── 3. Dataset / DataLoader ────────────────────────────────
class TextDataset(Dataset):
    def __init__(self, data, vocab):
        self.X = [encode(s, vocab) for s, _ in data]
        self.y = [lb for _, lb in data]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (
            torch.tensor(self.X[i], dtype=torch.long),
            torch.tensor(self.y[i], dtype=torch.long),
        )


# ─── 4. 模型定义 ────────────────────────────────────────────
class ModelA(nn.Module):
    """
    模型 A：Embedding → RNN → MaxPool → Linear(5)。
    对 RNN 所有时间步的输出做最大池化，输出五个类别的原始分数。
    """
    def __init__(self, vocab_size, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn       = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.fc        = nn.Linear(hidden_dim, NUM_CLASSES)

    def forward(self, x):
        # x: (batch, seq_len)
        outputs, _ = self.rnn(self.embedding(x))  # (B, seq_len, hidden_dim)
        pooled = outputs.max(dim=1).values  # 沿时间维度池化，得到 (B, hidden_dim)
        # CrossEntropyLoss 内部包含 log_softmax，训练时直接返回 logits。
        return self.fc(pooled)  # (B, 5)


class ModelB(nn.Module):
    """模型 B：Embedding → MaxPool → Linear(5)，不包含位置信息。"""
    def __init__(self, vocab_size, embed_dim=EMBED_DIM):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.fc = nn.Linear(embed_dim, NUM_CLASSES)

    def forward(self, x):
        embeddings = self.embedding(x)  # (B, seq_len, embed_dim)
        pooled = embeddings.max(dim=1).values  # (B, embed_dim)
        return self.fc(pooled)  # (B, 5)，直接返回 logits


# ─── 5. 训练与评估 ──────────────────────────────────────────
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X, y in loader:
            logits  = model(X)
            pred    = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total   += len(y)
    return correct / total


def train_one_model(name, model, train_dataset, val_loader):
    # 每个模型都使用独立且同种子的采样器，确保每轮的训练样本顺序一致。
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    total_params = sum(p.numel() for p in model.parameters())
    history = {'loss': [], 'val_acc': []}
    training_seconds = 0.0
    print(f"\n--- {name}，参数量：{total_params:,} ---")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        started = perf_counter()
        for X, y in train_loader:
            logits = model(X)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
        training_seconds += perf_counter() - started

        avg_loss = total_loss / len(train_dataset)
        val_acc  = evaluate(model, val_loader)
        history['loss'].append(avg_loss)
        history['val_acc'].append(val_acc)
        print(f"{name} | Epoch {epoch:2d}/{EPOCHS}  loss={avg_loss:.4f}  val_acc={val_acc:.2%}")

    return {
        'history': history,
        'params': total_params,
        'train_acc': evaluate(model, DataLoader(train_dataset, batch_size=BATCH_SIZE)),
        'val_acc': history['val_acc'][-1],
        'seconds': training_seconds,
    }


# ─── 6. 对比展示 ────────────────────────────────────────────
def show_predictions(models, vocab, sentences):
    X = torch.tensor([encode(sent, vocab) for sent in sentences], dtype=torch.long)
    probabilities = {}
    with torch.no_grad():
        for name, model in models.items():
            model.eval()
            probabilities[name] = torch.softmax(model(X), dim=1)

    print('句子 | 真实类别 | 模型 A 预测（概率） | 模型 B 预测（概率）')
    for i, sent in enumerate(sentences):
        predictions = []
        for probs in probabilities.values():
            confidence, index = probs[i].max(dim=0)
            predictions.append(f'第 {index.item() + 1} 类（{confidence.item():.2%}）')
        print(f'{sent} | 第 {sent.index("你") + 1} 类 | ' + ' | '.join(predictions))
    return probabilities


def plot_comparison(results):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, result in results.items():
        history = result['history']
        epochs = range(1, len(history['loss']) + 1)
        label = 'Model A (with RNN)' if name == '模型 A' else 'Model B (without RNN)'
        axes[0].plot(epochs, history['loss'], label=label)
        axes[1].plot(epochs, history['val_acc'], label=label)
    axes[0].set(title='Training loss', xlabel='Epoch', ylabel='Cross-entropy loss')
    axes[1].set(title='Validation accuracy', xlabel='Epoch', ylabel='Accuracy', ylim=(0, 1.05))
    axes[1].axhline(1 / NUM_CLASSES, color='gray', linestyle='--', label='Random guess (20%)')
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(CURVE_PATH, dpi=150)
    plt.close(fig)
    print(f'\n对比曲线已保存：{CURVE_PATH}')


def show_comparison(models, results, vocab):
    print('\n--- 两个模型的训练结果 ---')
    print('模型 A：Embedding → RNN → MaxPool → Linear(5)')
    print('模型 B：Embedding → MaxPool → Linear(5)')
    print('模型 | 参数量 | 最后一轮训练损失 | 训练准确率 | 验证准确率 | 训练耗时(s)')
    for name, result in results.items():
        print(
            f'{name} | {result["params"]:,} | {result["history"]["loss"][-1]:.4f}'
            f' | {result["train_acc"]:.2%} | {result["val_acc"]:.2%} | {result["seconds"]:.2f}'
        )
    gap = (results['模型 A']['val_acc'] - results['模型 B']['val_acc']) * 100
    print(f'验证准确率差值（A - B）：{gap:+.2f} 个百分点；均匀随机猜测的期望准确率为 20%。')
    print('耗时仅统计训练循环，受本机负载影响；模型 A 因增加 RNN 而有更多参数。')

    print('\n--- 普通句子的预测对比 ---')
    show_predictions(models, vocab, [
        '你今天真好', '有你真开心', '今天你真棒', '我想请你来', '大家都想你',
    ])

    print('\n--- 固定汉字，只改变“你”的位置 ---')
    base = '今天真好'
    position_sentences = [
        base[:position] + '你' + base[position:] for position in range(NUM_CLASSES)
    ]
    probabilities = show_predictions(models, vocab, position_sentences)
    for name, probs in probabilities.items():
        difference = (probs - probs[0]).abs().max().item()
        print(f'{name}：上述排列相对于第一句的最大类别概率差为 {difference:.8f}')

    print('\n--- 为什么 RNN 在这个任务中有效？ ---')
    print('1. 模型 B 对每个字独立做 Embedding，再取最大值。重排同一组字不会改变池化结果。')
    print('   理论上五句的输出相同（实际计算可能有微小浮点误差），无法据此区分五种位置。')
    print('2. 模型 A 的 RNN 隐藏状态同时依赖当前字与前文；字序变化会影响各时间步的表示。')
    print('   再做 MaxPool 时，池化的已经是包含顺序信息的表示，因此能够学习“你”的位置。')
    print('3. 这个实验说明位置敏感任务需要顺序信息，RNN 是一种实现方式。')
    print('   它不表示所有 NLP 任务都需要 RNN；带位置编码的 Transformer 等也能建模顺序。')
    plot_comparison(results)


def train():
    random.seed(SEED)
    print('生成两个模型共用的数据集...')
    data = build_dataset(N_SAMPLES)
    split = int(len(data) * TRAIN_RATIO)
    train_data, val_data = data[:split], data[split:]
    vocab = build_vocab(train_data)
    train_dataset = TextDataset(train_data, vocab)
    val_loader = DataLoader(TextDataset(val_data, vocab), batch_size=BATCH_SIZE)
    print(f'训练样本：{len(train_data)}，验证样本：{len(val_data)}，词表大小：{len(vocab)}')
    print(f'相同配置：Adam，lr={LR}，batch_size={BATCH_SIZE}，epochs={EPOCHS}')
    print('共享数据划分和词表，使用相同的 Embedding 初始权重与每轮训练样本顺序。')

    models, results = {}, {}
    for name, model_class in [('模型 A', ModelA), ('模型 B', ModelB)]:
        # 两个类都先构造相同形状的 Embedding；重置种子使其初始权重相同。
        torch.manual_seed(SEED)
        model = model_class(vocab_size=len(vocab))
        models[name] = model
        results[name] = train_one_model(name, model, train_dataset, val_loader)
    show_comparison(models, results, vocab)
    return models, results, vocab


if __name__ == '__main__':
    train()
