# coding:utf8

import numpy as np
import torch
import torch.nn as nn

"""
week2作业：
    对于任意五维向量，它的最大数字在第几维就属于第几类
"""


class TorchModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.hidden = torch.relu
        self.activation = torch.softmax
        self.loss = nn.functional.cross_entropy

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.linear(x)
        x = self.hidden(x)
        y_pred = self.activation(x, dim=1)
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred


def build_sample():
    x = np.random.random(5)
    y = np.argmax(x)  # 取第一个最大值的索引作为类别
    return x, y


def build_dataset(train_sample):
    X = []
    Y = []
    for _ in range(train_sample):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    X = np.array(X)
    Y = np.array(Y)
    return torch.FloatTensor(X), torch.LongTensor(Y)


def evaluate(model, X, Y):
    model.eval()
    correct = 0
    with torch.no_grad():
        y_pred = model(X)
        pred_classes = torch.argmax(y_pred, dim=1)
        correct = (pred_classes == Y).sum().item()
    accuracy = correct / len(Y)
    print(f'正确预测个数: {correct}, 正确率: {accuracy * 100:.2f}%')
    return accuracy


def train_model(model, optimizer, X_train, Y_train, epochs=20, batch_size=20):
    for _ in range(epochs):
        model.train()
        for i in range(0, len(X_train), batch_size):
            X_batch = X_train[i:i + batch_size]
            Y_batch = Y_train[i:i + batch_size]
            loss = model(X_batch, Y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def main(train_sample, test_sample, epochs=50, batch_size=25):
    input_size = 5
    output_size = 5
    X_train, Y_train = build_dataset(train_sample)
    X_test, Y_test = build_dataset(test_sample)

    model = TorchModel(input_size, output_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    train_model(model, optimizer, X_train, Y_train, epochs, batch_size)

    accuracy = evaluate(model, X_test, Y_test)
    print(f'训练样本数量: {train_sample}, 测试样本数量: {test_sample}, 测试集正确率: {accuracy * 100:.2f}%')


if __name__ == "__main__":
    train_sample = 5000
    test_sample = 500
    main(train_sample, test_sample)
