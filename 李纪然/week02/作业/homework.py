# coding:utf8

# 解决 OpenMP 库冲突问题
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt




class TorchModel(nn.Module):
    def __init__(self, input_size, class_num):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, class_num)  # 线性层
        # 多分类每个类别输出一个分数(logits)，不再需要sigmoid
        # cross_entropy内部会先做softmax再算交叉熵，所以这里输出原始logits
        self.loss = nn.functional.cross_entropy

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.linear(x)  # (batch_size, input_size) -> (batch_size, class_num)
        if y is not None:
            return self.loss(x, y)  # 预测值和真实值计算损失
        else:
            return x  # 输出每个类别的得分


# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个5维向量，最大值在第几维(从0开始数)，就是第几类
def build_sample():
    x = np.random.random(5)
    y = int(np.argmax(x))  # 最大值的下标，就是类别
    return x, y


# 随机生成一批样本
# 因为5个数都是均匀分布的随机数，每个维度成为最大值的概率相同，5类样本自然均匀
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)


# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model, class_num):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    class_count = [0] * class_num
    for t in y.tolist():
        class_count[t] += 1
    print("本次预测集中各类别样本个数：%s" % class_count)
    correct = 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测，得到 (test_sample_num, class_num) 的得分
        pred_class = torch.argmax(y_pred, dim=1)  # 得分最高的那一维就是预测类别
        correct = int((pred_class == y).sum())  # 与真实标签进行对比
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / test_sample_num))
    return correct / test_sample_num


def main():
    # 固定随机种子，保证每次运行结果一致，方便复现
    torch.manual_seed(0)
    np.random.seed(0)
    # 配置参数
    epoch_num = 100  # 训练轮数（线性模型训练不充分时准确率只有90%左右，所以多训一些）
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    class_num = 5  # 类别数：最大值可能在5个维度中的任意一个
    learning_rate = 0.01  # 学习率
    # 建立模型
    model = TorchModel(input_size, class_num)
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
            #取出一个batch数据作为输入   train_x[0:20]  train_y[0:20] train_x[20:40]  train_y[20:40]
            x = train_x[batch_index * batch_size: (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]
            loss = model(x, y)  # 计算loss  model.forward(x,y)
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, class_num)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "model_multiclass.bin")
    # 训练完立刻用 test_vec 检验模型有没有学到规律
    print("训练结束，用测试向量检验：")
    predict("model_multiclass.bin", test_vec)
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return


# 测试向量：用来检验模型有没有学到"最大值在第几维就是第几类"这个规律
test_vec = [[0.81889086, 0.15229675, 0.31082123, 0.03504317, 0.88920843],
            [0.94963533, 0.5524256, 0.95758807, 0.95520434, 0.84890681],
            [0.90797868, 0.67482528, 0.13625847, 0.34675372, 0.19871392],
            [0.39349776, 0.59416669, 0.92579291, 0.41567412, 0.8358894]]


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    class_num = 5
    model = TorchModel(input_size, class_num)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        logits = model.forward(torch.FloatTensor(input_vec))  # 模型预测，得到各类别得分
        probs = torch.softmax(logits, dim=1)  # 得分转成概率，方便观察把握程度
    correct = 0
    for vec, prob in zip(input_vec, probs):
        pred = int(torch.argmax(prob))  # 概率最大的类别
        true_class = int(np.argmax(vec))  # 真实类别：向量最大值所在的下标
        correct += int(pred == true_class)
        flag = "正确" if pred == true_class else "错误"
        print("输入：%s, 预测类别：%d (即第%d维数字最大), 概率值：%f, 真实类别：%d, 预测%s"
              % (vec, pred, pred + 1, prob[pred], true_class, flag))  # 打印结果
    print("共%d条测试向量，预测正确%d条" % (len(input_vec), correct))
    return correct / len(input_vec)


if __name__ == "__main__":
    predict("model_multiclass.bin", test_vec)
