# coding: utf-8
"""五分类练习：输入为 5 维向量，最大元素的位置决定类别。

训练标签使用 0～4，输出给用户的类别使用 1～5。
若最大值并列，标签取第一个最大值的位置。
运行：python Week2Homework.py（需要安装 torch 和 matplotlib）。
"""

from pathlib import Path
from itertools import combinations

import matplotlib.pyplot as plt
import torch
from torch import nn


INPUT_SIZE = 5
MODEL_PATH = Path(__file__).with_name("week2_model.bin")
CURVE_PATH = Path(__file__).with_name("week2_training_curves.png")
# 包含所有 2～5 个位置并列的组合，避免只学习前两个位置并列。
TIE_PATTERNS = [
    positions
    for count in range(2, INPUT_SIZE + 1)
    for positions in combinations(range(INPUT_SIZE), count)
]


class TorchModel(nn.Module):
    def __init__(self, input_size=INPUT_SIZE):
        super().__init__()
        # 隐藏层学习不同最大值位置组合下的优先关系。
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, INPUT_SIZE),
        )
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        logits = self.network(x)  # (batch_size, 5)
        if y is not None:
            # CrossEntropyLoss 内部包含 log_softmax，传入原始分数即可。
            return self.loss(logits, y)
        return logits


def build_sample(tie_ratio=0.3):
    x, y = build_dataset(1, tie_ratio)
    return x[0], y[0].item()


def build_dataset(total_sample_num, tie_ratio=0.3):
    """混合普通随机样本和并列样本；tie_ratio 是每个样本生成并列的概率。"""
    if not 0 <= tie_ratio <= 1:
        raise ValueError("tie_ratio 必须在 0～1 之间")
    x = torch.rand(total_sample_num, INPUT_SIZE)
    tie_rows = torch.where(torch.rand(total_sample_num) < tie_ratio)[0]
    pattern_indices = torch.randint(len(TIE_PATTERNS), (len(tie_rows),))
    for row, pattern_index in zip(tie_rows.tolist(), pattern_indices.tolist()):
        # 将选中的位置设为相同最大值，其余位置严格小于该值。
        max_value = 0.1 + 0.9 * torch.rand(())
        x[row] *= max_value * 0.99
        x[row, list(TIE_PATTERNS[pattern_index])] = max_value
    # 标签形状为 (样本数,)，类型为 long，值域为 0～4。
    # argmax 对并列最大值返回第一个位置；这个规则只用于制作标签。
    y = x.argmax(dim=1)
    return x, y


def evaluate(model, test_x, test_y):
    model.eval()
    with torch.no_grad():
        predictions = model(test_x).argmax(dim=1)
        accuracy = (predictions == test_y).float().mean().item()
    return accuracy


def plot_training_curves(acc_history, loss_history):
    epochs = range(1, len(acc_history) + 1)
    # 分开绘制，避免准确率和损失的不同量纲影响阅读。
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, acc_history, label="Validation acc")
    axes[0].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy")
    axes[0].set_ylim(0, 1)
    axes[1].plot(epochs, loss_history, color="tab:orange", label="Training loss")
    axes[1].set(title="Loss", xlabel="Epoch", ylabel="Cross-entropy loss")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(CURVE_PATH, dpi=150)
    print(f"训练曲线已保存至：{CURVE_PATH}")
    plt.show()
    plt.close(fig)


def main():
    torch.manual_seed(42)
    epoch_num = 60
    batch_size = 64
    learning_rate = 0.003
    model = TorchModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.3)

    # 单独生成验证集，评估数据不参与参数更新。
    validation_x, validation_y = build_dataset(2000)
    tie_mask = (validation_x == validation_x.max(dim=1, keepdim=True).values).sum(dim=1) > 1
    acc_history, loss_history = [], []
    for epoch in range(epoch_num):
        # 每轮重新采样，增加并列数值及位置组合的多样性。
        train_x, train_y = build_dataset(10000)
        model.train()
        indices = torch.randperm(len(train_x))
        total_loss = 0.0
        for start in range(0, len(train_x), batch_size):
            batch_indices = indices[start:start + batch_size]
            x, y = train_x[batch_indices], train_y[batch_indices]
            optimizer.zero_grad()
            loss = model(x, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(x)

        scheduler.step()
        accuracy = evaluate(model, validation_x, validation_y)
        ordinary_acc = evaluate(model, validation_x[~tie_mask], validation_y[~tie_mask])
        tie_acc = evaluate(model, validation_x[tie_mask], validation_y[tie_mask])
        average_loss = total_loss / len(train_x)
        acc_history.append(accuracy)
        loss_history.append(average_loss)
        print(
            f"第 {epoch + 1:02d} 轮 | 平均损失：{average_loss:.4f}"
            f" | 验证准确率：{accuracy:.2%}"
            f" | 普通样本：{ordinary_acc:.2%} | 并列样本：{tie_acc:.2%}"
        )

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"模型已保存至：{MODEL_PATH}")
    plot_training_curves(acc_history, loss_history)
    return model


def predict(model_path, input_vec):
    x = torch.as_tensor(input_vec, dtype=torch.float32)
    if x.ndim == 1:
        x = x.unsqueeze(0)
    if x.ndim != 2 or x.shape[1] != INPUT_SIZE:
        raise ValueError("每个输入样本必须是 5 维向量")
    if not torch.isfinite(x).all():
        raise ValueError("输入向量不能包含 NaN 或无穷大")

    model = TorchModel()
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()
    with torch.no_grad():
        probabilities = torch.softmax(model(x), dim=1)
        confidence, indices = probabilities.max(dim=1)
    classes = (indices + 1).tolist()
    for vec, category, probability in zip(x.tolist(), classes, confidence.tolist()):
        print(f"输入：{vec}，预测类别：第 {category} 类，概率：{probability:.4f}")
    return classes


if __name__ == "__main__":
    #main()  # 加入并列样本后需要重新训练，旧模型不会自动学到新规则。
    predict(MODEL_PATH, [
        [0.9, 0.1, 0.2, 0.3, 0.4],
        [0.9, 0.9, 0.2, 0.3, 0.4],
        [0.1, 0.2, 0.2, 0.3, 0.4],
        [0.1, 0.2, 0.9, 0.9, 0.4],
        [0.1, 0.2, 0.3, 0.4, 0.9],
    ])
