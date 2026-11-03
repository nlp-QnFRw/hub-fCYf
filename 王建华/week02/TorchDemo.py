# coding:utf8
# 解决 OpenMP 库冲突问题
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""
基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个5维向量，它最大的数字在第几维就属于第几类
如： [0.122,0.5622,0.85,0.874232,0.43523]
最大数字是0.874232，下标是3（从0数），属于第4类
"""


class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        # 输入5维，输出5个分数（5分类logits）
        self.linear = nn.Linear(input_size, 5)
        # 多分类使用CrossEntropyLoss，内部自带激活方法softmax
        self.loss = nn.CrossEntropyLoss()

    # 当输入真实标签，返回loss值；无真实标签，返回预测logits
    def forward(self, x, y=None):
        y_pred = self.linear(x)  # (batch_size, 5)，5个类别的原始分数logits
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred


# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个5维向量，它最大的数字在第几维就属于第几类（标签0~4）
def build_sample():
    x = torch.rand(5)   # 5个0~1随机数
    label = torch.argmax(x).item()  # 0,1,2,3,4
    return x, label


# 随机生成一批样本
def build_dataset(n):
    xs = []
    ys = []
    for _ in range(n):
        x, y = build_sample()
        xs.append(x)
        ys.append(y)
    x = torch.stack(xs)
    y = torch.tensor(ys)
    return x, y


# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    correct, wrong = 0, 0
    # torch.no_grad() 取消梯度计算
    with torch.no_grad():
        # 模型计算后,得到预测值, 100个数据对应100行,每行5个得分数,得分最高的维数就是预测的第几类别
        y_pred = model(x)
        for i in range(test_sample_num):
            # 拿到当前这一个样本的5个分数
            logits = y_pred[i]
            # 找到5个分数里，最大值在第几号位置（0~4）
            pred_idx = torch.argmax(logits).item()
            predict_class = pred_idx + 1
            # 拿到这个样本真实的类别
            true_idx = y[i].item()
            true_class = true_idx + 1
            # 判断预测和真实是不是一样
            if predict_class == true_class:
                correct = correct + 1
            else:
                wrong = wrong + 1

    print("正确数量：", correct)
    print("错误数量：", wrong)
    print("正确率：", correct / test_sample_num)
    return correct / test_sample_num


def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 500  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    learning_rate = 0.01  # 学习率
    # 建立模型
    model = TorchModel(input_size)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练集，正常任务是读取训练集
    train_x, train_y = build_dataset(train_sample)
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        # 每轮训练样本数量 = 样本总数/训练轮数
        for batch_index in range(train_sample // batch_size):
            #取出一个batch数据作为输入
            x = train_x[batch_index * batch_size: (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]
            loss = model(x, y)  # 计算loss  model.forward(x,y)
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "model.bin")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path, weights_only=True))  # 加载训练好的权重

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测，shape [N,5]
    for vec, res in zip(input_vec, result):
        pred_idx = torch.argmax(res).item()
        pred_class = pred_idx + 1
        print("输入：%s, 预测类别：%d" % (vec, pred_class))


if __name__ == "__main__":
    main()
    test_vec = [[0.81889086,0.15229675,0.31082123,0.03504317,0.88920843],
                [0.94963533,0.5524256,0.95758807,0.95520434,0.84890681],
                [0.90797868,0.67482528,0.13625847,0.34675372,0.19871392],
                [0.39349776,0.59416669,0.92579291,0.41567412,0.8358894]]
    predict("model.bin", test_vec)
