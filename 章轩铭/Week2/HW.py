# coding:utf8
"""
基于 PyTorch 框架编写多分类模型训练

实现一个自行构造的找规律任务：
输入是一个五维向量，最大数字出现在哪一维，就属于哪一类

类别编号使用 0 到 4：
0 表示第 1 维最大，1 表示第 2 维最大，依次类推
"""

import os
import random
import sys
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import torch
import torch.nn as nn


SEED = 2026
INPUT_SIZE = 5
CLASS_NUM = 5
MODEL_PATH = Path(__file__).resolve().parent / "model.bin"


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TorchModel(nn.Module):
    """输入五维向量，输出五个类别的 logits。"""

    def __init__(self, input_size=INPUT_SIZE, class_num=CLASS_NUM):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, class_num)

    def forward(self, x):
        return self.linear(x)


def build_sample():
    """随机生成一条样本，最大值所在位置就是标签。"""
    x = np.random.random(INPUT_SIZE).astype(np.float32)
    y = int(np.argmax(x))
    return x, y


def build_dataset(total_sample_num):
    """随机生成一批样本和标签。"""
    x_list = []
    y_list = []
    for _ in range(total_sample_num):
        x, y = build_sample()
        x_list.append(x)
        y_list.append(y)
    return torch.tensor(np.array(x_list), dtype=torch.float32), torch.tensor(y_list, dtype=torch.long)


def evaluate(model, device, test_sample_num=100):
    """生成一批新测试数据，检查模型分类准确率。"""
    model.eval()
    x, y = build_dataset(test_sample_num)
    x, y = x.to(device), y.to(device)

    with torch.no_grad():
        logits = model(x)
        predictions = logits.argmax(dim=1)

    correct = int((predictions == y).sum().item())
    accuracy = correct / test_sample_num
    print(f"本次测试集样本数：{test_sample_num}，正确预测：{correct}，准确率：{accuracy:.2%}")
    return accuracy


def predict(model_path, input_vec):
    """加载训练好的模型，对新的五维向量进行预测。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TorchModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    x = torch.tensor(input_vec, dtype=torch.float32).to(device)
    if x.ndim == 1:
        x = x.unsqueeze(0)

    with torch.no_grad():
        logits = model(x)
        predictions = logits.argmax(dim=1)

    for vector, prediction in zip(input_vec, predictions.cpu().numpy()):
        print(f"输入：{vector}")
        print(f"预测类别：第 {int(prediction) + 1} 类")


def main():
    import matplotlib.pyplot as plt

    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备：", device)

    # 配置参数
    epoch_num = 100  # 训练轮数
    batch_size = 32  # 每次训练样本个数
    train_sample_num = 5000  # 每轮训练总共训练的样本总数
    learning_rate = 0.005  # 学习率

    model = TorchModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []

    train_x, train_y = build_dataset(train_sample_num)
    train_x, train_y = train_x.to(device), train_y.to(device)

    for epoch in range(epoch_num):
        model.train()
        watch_loss = []

        for batch_index in range(train_sample_num // batch_size):
            start = batch_index * batch_size
            end = (batch_index + 1) * batch_size
            x = train_x[start:end]
            y = train_y[start:end]

            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            watch_loss.append(loss.item())

        average_loss = np.mean(watch_loss)
        print(f"=========\n第 {epoch + 1} 轮平均 loss：{average_loss:.6f}")
        accuracy = evaluate(model, device)
        log.append((accuracy, average_loss))

    torch.save(model.state_dict(), MODEL_PATH)
    print("\n模型参数已保存到：", MODEL_PATH)
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.xlabel("epoch")
    plt.ylabel("value")
    plt.legend()
    plt.show()
    return


if __name__ == "__main__":
    main()
    test_vec = [
        [0.2, 0.8, 0.3, 0.1, 0.4],
        [0.9, 0.2, 0.1, 0.3, 0.4],
        [0.1, 0.2, 0.3, 0.7, 0.4],
    ]
    predict(MODEL_PATH, test_vec)
