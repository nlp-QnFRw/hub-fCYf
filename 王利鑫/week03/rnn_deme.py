"""

分类器，RNN应该就够用了，LSTM相比与RNN，对文本的语义的记忆有改进，甚至RNN也不需要双向，因为这里不用考虑语义问题。

"""
import unicodedata
from collections import Counter, OrderedDict

import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, classification_report

# 批处理数量
# 时间步长
step_size = 1
# 嵌入层维度
embedding_dim = 64
# 学习率
lr = 0.001
# 轮次
epochs = 100


class Classifier12RNN(nn.Module):
    """ rnn 模型 计算文本分类"""

    def __init__(self, vocabulary_size):
        super().__init__()
        # 经过embedding层，转换成向量,同时指定填充符号在词表中的位置
        self.embedding = nn.Embedding(num_embeddings=vocabulary_size, embedding_dim=embedding_dim, padding_idx=0)
        # rnn 定义
        self.rnn = nn.RNN(batch_first=True, input_size=embedding_dim, hidden_size=embedding_dim)
        # 试试能否提升准确率：差别不大
        # self.norm = nn.LayerNorm(embedding_dim)
        # self.dropout = nn.Dropout(0.3)
        # 线性层
        self.linear = nn.Linear(embedding_dim, 12)
        # 交叉熵自动使用softmax输出概率
        self.active = nn.CrossEntropyLoss()

    # 前向传播，这里返回预测样本，方便后续计算准确率
    def forward(self, x):
        mask = (x != 0).unsqueeze(-1)
        embedding_x = self.embedding(x)
        output, _ = self.rnn(embedding_x)
        output = output.masked_fill(~mask, -1e9)
        pooled = output.max(dim=1)[0]
        # pooled = self.dropout(self.norm(pooled))
        return self.linear(pooled)


# 延用老师的方法
def normalize(text):
    """
    NFKC 归一化 + 转小写 + 只保留「字母(L)」与「组合符号(M)」。

    为什么保留 M：印地语、泰语的元音是组合标记（कि 里的 ि），
    只留字母会把印地语砍成裸辅音，模型就没法区分。
    """
    t = unicodedata.normalize("NFKC", text).lower()
    return "".join(c for c in t if unicodedata.category(c)[0] in ("L", "M"))


def build_vocabulary(data_train):
    """
    从训练集中读取数据
    :return: 词表
    """

    # 计数器
    seq = Counter()
    for row in data_train.iloc[:, 0]:
        for char in row:
            seq.update(normalize(char))
    # 填充字符、未知字符
    vocabulary = OrderedDict([("<PAD>", 0), ("<UNK>", 1)])
    for ch, _ in seq.items():
        vocabulary[ch] = len(vocabulary)
    return vocabulary


def encode(text, vocab, maxlen=40):
    ids = [vocab.get(ch, 1) for ch in normalize(text)][:maxlen]
    ids += [0] * (maxlen - len(ids))
    return ids


class TextDataset(Dataset):
    def __init__(self, rows, vocab, label2id):
        self.X = [encode(t, vocab) for t, _ in rows]
        self.y = [label2id[l] for _, l in rows]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return torch.tensor(self.X[i], dtype=torch.long), self.y[i]


def data_transform(data, vocabulary, label2id, batch_size=128, shuffle=False):
    rows = list(zip(
        data.iloc[:, 0],
        data.iloc[:, 1]
    ))

    return DataLoader(
        TextDataset(rows, vocabulary, label2id),
        batch_size=batch_size,
        shuffle=shuffle
    )


def predict(model, loader):
    model.eval()
    preds, golds = [], []
    with torch.no_grad():
        for X, y in loader:
            preds += model(X).argmax(dim=1).tolist()
            golds += y.tolist()
    return preds, golds


if __name__ == '__main__':

    # 开启交互模式
    plt.ion()
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()
    # Loss 曲线
    loss_line, = ax1.plot([], [], 'b-', label='Train Loss')
    # Accuracy 曲线
    acc_line, = ax2.plot([], [], 'r-', label='Dev Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    fig.suptitle('RNN Training Monitor')
    epochs_history = []
    loss_history = []
    acc_history = []
    plt.show(block=False)

    # 1.获取数据
    data_test = pd.read_csv('data/test.csv')
    data_dev = pd.read_csv('data/dev.csv')
    data_train = pd.read_csv('data/train.csv')
    # 2.构建词表
    vocabulary = build_vocabulary(data_train)
    # 3.数据转换
    label2id = {
        label: i
        for i, label in enumerate(
            sorted(data_train.iloc[:, 1].unique())
        )
    }

    train_loader = data_transform(
        data_train, vocabulary, label2id,
        shuffle=True
    )

    dev_loader = data_transform(
        data_dev, vocabulary, label2id
    )

    test_loader = data_transform(
        data_test, vocabulary, label2id
    )

    # 4.创建模型
    model = Classifier12RNN(len(vocabulary))

    # 5.定义优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print('begin_train')
    # 6.开始训练
    for epoch in range(epochs):
        model.train()
        loss_sum = 0
        for x, y in train_loader:
            logits = model(x)
            loss = model.active(logits, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_sum += loss.item()
        # 验证，计算准确率
        pred, real = predict(model, dev_loader)
        acc = sum(p == g for p, g in zip(pred, real)) / len(real)
        avg_loss = loss_sum / len(train_loader)

        # 保存历史指标
        epochs_history.append(epoch + 1)
        loss_history.append(avg_loss)
        acc_history.append(acc)
        # 更新曲线
        loss_line.set_data(epochs_history, loss_history)
        acc_line.set_data(epochs_history, acc_history)
        # 自动调整 Loss 轴范围
        ax1.relim()
        ax1.autoscale_view()
        # 更新 X 轴
        ax1.set_xlim(0, max(epochs, 1))
        # 刷新图表
        fig.canvas.draw_idle()
        plt.pause(0.01)

    print('done')

    preds, golds = predict(model, test_loader)
    acc = sum(p == g for p, g in zip(preds, golds)) / len(golds)
    print(f"\n测试集：accuracy = {acc:.4f}")

    accuracy = sum(
        p == g for p, g in zip(preds, golds)
    ) / len(golds)

    macro_f1 = f1_score(
        golds,
        preds,
        average="macro"
    )

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")

    # 训练结束后保持窗口显示
    plt.ioff()
    plt.show()

