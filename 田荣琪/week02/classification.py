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
规律：对于任意五维向量，它最大数字在第几维就属于第几类

"""

class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 5)  # 线性层
        self.loss = nn.CrossEntropyLoss()  # loss函数采用交叉熵损失

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        logits = self.linear(x)  # (batch_size, input_size) -> (batch_size, 5)
        if y is not None:
            return self.loss(logits, y)  # 预测值和真实值计算损失
        else:
            return logits  # 输出预测结果
        

def build_sample():
    # 随机生成一个五维向量
    x = np.random.random(5)

    # 找到最大元素的位置，作为类别标签
    y = np.argmax(x)

    # 返回输入向量和真实标签
    return x, y

# 随机生成一批样本
def build_dataset(total_sample_num):
    X = []
    Y = []

    for i in range(total_sample_num):
        # 生成一个样本及其真实类别
        x, y = build_sample()

        # 将输入向量添加到 X
        X.append(x)

        # 将类别标签添加到 Y
        Y.append(y) # CrossEntropyLoss类别索引标签是一维

    # 输入使用浮点张量，标签使用整数张量
    return torch.FloatTensor(np.array(X)), torch.LongTensor(Y)

def main():
    # 1. 配置训练参数
    epoch_num = 30
    batch_size = 20
    train_sample = 5000
    input_size = 5
    learning_rate = 0.01

    # 2. 创建五分类模型
    model = TorchModel(input_size)

    # 3. 定义优化器
    optim = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    # 保存每轮的准确率和损失
    log = []

    # 4. 生成训练数据
    train_x, train_y = build_dataset(train_sample)

    # 5. 开始训练
    for epoch in range(epoch_num):

        # 切换到训练模式
        model.train()

        watch_loss = []

        # 每轮打乱训练样本顺序
        indices = torch.randperm(train_sample)
        train_x = train_x[indices]
        train_y = train_y[indices]

        # 按 Batch 训练
        for batch_index in range(train_sample // batch_size):

            # 取出当前 Batch
            start = batch_index * batch_size
            end = (batch_index + 1) * batch_size

            x = train_x[start:end]
            y = train_y[start:end]

            # 梯度清零
            optim.zero_grad()

            # 前向传播，计算交叉熵损失
            loss = model(x, y)

            # 反向传播，计算梯度
            loss.backward()

            # 使用 Adam 更新模型参数
            optim.step()

            # 保存当前 Batch 的损失
            watch_loss.append(loss.item())

        # 6. 每轮训练结束后评估
        avg_loss = float(np.mean(watch_loss))

        print("=========")
        print("第%d轮平均loss：%.6f" % (
            epoch + 1, avg_loss
        ))

        acc = evaluate(model)

        # 保存本轮训练记录
        log.append([acc, avg_loss])

    # 7. 保存训练好的模型
    torch.save(model.state_dict(), "model.bin")

    # 8. 绘制训练结果曲线
    epoch_list = range(1, len(log) + 1)

    plt.figure(figsize=(10, 5))

    plt.plot(
        epoch_list,
        [item[0] for item in log],
        label="Accuracy"
    )

    plt.plot(
        epoch_list,
        [item[1] for item in log],
        label="Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Five-Class Classification Training")
    plt.legend()
    plt.grid(True)

    plt.savefig("training_curve.png", dpi=150)

    return model

# 测试：计算五分类模型的准确率
def evaluate(model):
    # 1. 切换到评估模式
    model.eval()

    # 2. 生成一批新的测试数据
    test_sample_num = 1000
    x, y = build_dataset(test_sample_num)

    with torch.no_grad():

        # 3. 得到模型输出的5个类别得分
        logits = model(x)

        # 4. 选择得分最高的类别
        y_pred = torch.argmax(logits, dim=1)

        # 5. 比较预测标签与真实标签
        correct = (y_pred == y).sum().item()

    # 7. 计算准确率
    accuracy = correct / test_sample_num

    print("测试样本数：%d" % test_sample_num)
    print("正确预测个数：%d" % correct)
    print("测试准确率：%.4f" % accuracy)

    return accuracy

# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))  # 加载训练好的权重
    # print(model.state_dict())

    model.eval()  # 测试模式
    x = torch.FloatTensor(input_vec)  # 转换为张量
    with torch.no_grad():  # 不计算梯度
        logits = model(x)
        prob = torch.softmax(logits, dim=1)  # 计算每个类别的概率
        y_pred = torch.argmax(logits, dim=1)  # 选择概率最大的类别
    for vec, pred, prob in zip(
        input_vec, y_pred, prob
    ):
        
        print("输入：%s, 真实类别：%d, 预测类别：%d, 类别概率：%f" % (vec, int(np.argmax(vec)), int(pred), prob.tolist()[int(pred)]))
        print("预测结果：", "正确" if int(np.argmax(vec)) == int(pred) else "错误")

if __name__ == "__main__":
    # main()
    test_vec = [[0.81889086,0.15229675,0.31082123,0.03504317,0.88920843],
                [0.94963533,0.5524256,0.95758807,0.95520434,0.84890681],
                [0.90797868,0.67482528,0.13625847,0.34675372,0.19871392],
                [0.39349776,0.59416669,0.92579291,0.41567412,0.8358894]]
    predict("model.bin", test_vec)
