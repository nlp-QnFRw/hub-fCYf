# coding:utf8

# 解决 OpenMP 库冲突问题
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
import random
import json

"""

基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个5维向量，最大的数在第几维，就属于第几类
场景：5个维度分别代表语文、数学、英语、物理、化学五门课的归一化成绩，
      哪门课成绩最高，就预测为该学生的优势学科。
      即：0-语文，1-数学，2-英语，3-物理，4-化学

"""


class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 5)  # 线性层：5维输入 -> 5类输出
        self.activation = nn.Softmax(dim=1)  # softmax归一化函数
        self.loss = nn.functional.cross_entropy  # loss函数采用交叉熵损失

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.linear(x)  # (batch_size, input_size) -> (batch_size, 5)
        if y is not None:
            # 交叉熵内部自带log_softmax，因此这里直接传入未归一化的结果
            return self.loss(x, y)  # 预测值和真实值计算损失
        else:
            return self.activation(x)  # 输出预测概率


# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个5维向量，最大值在第几个位置，就属于第几类
def build_sample():
    x = np.random.random(5)
    max_index = 0
    for i in range(1, 5):
        if x[i] > x[max_index]:
            max_index = i
    y = max_index  # 最大值下标0~4，正好对应5个类别
    return x, y


# 随机生成一批样本
# 每个位置成为最大值的概率相同，所以各类别大致均匀
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    # print(X)
    # print(Y)
    return torch.FloatTensor(X), torch.LongTensor(Y)  # 交叉熵要求标签为long类型


# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    print("本次预测集中各类别样本数：%s" % torch.bincount(y, minlength=5).tolist())
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测 model.forward(x)
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            pred_class = 0
            for i in range(1, 5):
                if y_p[i] > y_p[pred_class]:
                    pred_class = i
            if pred_class == int(y_t):
                correct += 1  # 预测类别正确
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


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
        for batch_index in range(train_sample // batch_size):
            # 取出一个batch数据作为输入
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
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
    # 打印每轮结果；这里直接结束训练，不再调用Matplotlib绘图
    print("每轮结果 [准确率, 平均loss]：", log)
    print("训练完成，开始输出预测结果……")
    return


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
    for vec, res in zip(input_vec, result):
        # 先用for+if找出输入向量最大值在第几维
        true_class = 0
        for i in range(1, 5):
            if vec[i] > vec[true_class]:
                true_class = i

        # 再用for+if找出模型概率最大的类别
        pred_class = 0
        for i in range(1, 5):
            if res[i] > res[pred_class]:
                pred_class = i

        print()
        print("输入：%s" % (vec,))
        print("最大值在第%d维 -> 真实类别%d；模型预测第%d类；概率值：%s"
              % (true_class + 1, true_class + 1,
                 pred_class + 1, np.round(res.numpy(), 4)))  # 打印结果


if __name__ == "__main__":
    main()
    test_vec = [[0.81889086,0.15229675,0.31082123,0.03504317,0.88920843],
                [0.94963533,0.5524256,0.95758807,0.95520434,0.84890681],
                [0.90797868,0.67482528,0.13625847,0.34675372,0.19871392],
                [0.39349776,0.59416669,0.92579291,0.41567412,0.8358894]]
    predict("model.bin", test_vec)
