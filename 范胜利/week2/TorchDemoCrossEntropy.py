# coding:utf8

# ============ 环境变量必须在 import torch / matplotlib 之前设置 ============
# 本机 conda 环境同时存在 Intel OpenMP(libiomp5) 和 LLVM OpenMP(libomp) 两套运行时，
# 多线程训练时会在 libomp 的 __kmp_suspend_initialize_thread 里段错误(Segmentation fault: 11)，
# 这里强制单线程来规避；如果把 os.environ 写在 import 之后就不生效了。
import matplotlib.pyplot as plt
import json
import random
import numpy as np
import torch.nn as nn
import torch
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # 允许重复 OpenMP 共存，避免直接 abort
os.environ['OMP_NUM_THREADS'] = '1'  # 关键：只用一个 OpenMP 线程，绕开两套运行时冲突
os.environ['MKL_NUM_THREADS'] = '1'


torch.set_num_threads(1)  # 双保险：即使环境变量没生效，也不会多线程跑

"""

基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个5维向量，5个数中最大值出现在第几维(下标0~4)，该样本就属于第几类
      即 y = argmax(x)，一共5个类别，采用交叉熵(cross entropy)损失函数

"""


class TorchModel(nn.Module):
    def __init__(self, input_size, class_num):
        super(TorchModel, self).__init__()
        # 线性层 (batch_size, 5) -> (batch_size, 5) 输出logits
        self.linear = nn.Linear(input_size, class_num)
        self.loss = nn.functional.cross_entropy  # 交叉熵损失，内部自带softmax，因此输入用logits

    # 当输入真实标签，返回loss值；无真实标签，返回预测概率
    def forward(self, x, y=None):
        # (batch_size, input_size) -> (batch_size, class_num) 未归一化的得分
        logits = self.linear(x)
        if y is not None:
            return self.loss(logits, y)  # 预测得分和真实类别计算交叉熵损失
        else:
            # 推理阶段用softmax把logits转换成概率，(batch_size, class_num)
            return torch.softmax(logits, dim=-1)


# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个5维向量，最大值在第几维(下标0~4)就属于第几类
def build_sample():
    x = np.random.random(5)
    y = int(np.argmax(x))  # 最大值所在维度就是类别下标
    return x, y


# 随机生成一批样本
# 每个类别(哪一维最大)理论上各占1/5的概率
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)  # 直接把类别下标作为标签，交叉熵要求标签为LongTensor
    # 一次性堆成 (n, 5) 的矩阵再转 tensor；
    # 若把 list[np.ndarray] 直接丢给 FloatTensor，会走逐元素拷贝的慢路径并报 UserWarning
    X = np.stack(X).astype('float32')
    return torch.FloatTensor(X), torch.LongTensor(Y)

# 测试代码
# 用来测试每轮模型的准确率


def evaluate(model, class_num=5):
    model.eval()
    test_sample_num = 200
    x, y = build_dataset(test_sample_num)
    class_count = [0] * class_num
    for y_t in y:
        class_count[int(y_t)] += 1
    print("本次预测集中各类别(哪一维最大)的样本数：%s" % class_count)
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测 model.forward(x)，得到(样本数, 类别数)的概率
        for y_p, y_t in zip(y_pred, y):
            if int(torch.argmax(y_p)) == int(y_t):  # 概率最大的那一类就是预测类别
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    class_num = 5  # 类别个数：5维向量中最大值出现在第几维，就属于第几类
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
            # 取出一个batch数据作为输入   train_x[0:20]  train_y[0:20] train_x[20:40]  train_y[20:40]
            x = train_x[batch_index *
                        batch_size: (batch_index + 1) * batch_size]
            y = train_y[batch_index *
                        batch_size: (batch_index + 1) * batch_size]
            loss = model(x, y)  # 计算loss  model.forward(x,y)
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, class_num)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型（单独命名，避免覆盖 TorchDemo.py 生成的二分类 model.bin）
    torch.save(model.state_dict(), "model_cross_entropy.bin")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return


# 使用训练好的模型做预测
def predict(model_path, input_vec, input_size=5, class_num=5):
    model = TorchModel(input_size, class_num)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测，得到每个类别的概率
    for vec, res in zip(input_vec, result):
        pred_class = int(torch.argmax(res))  # 概率最大的类别
        max_dim = int(np.argmax(vec))  # 真实的最大值所在维度
        print("输入：%s, 预测类别：%d(第%d维最大), 真实类别：%d, 概率值：%.4f, 是否正确：%s"
              % (vec, pred_class, pred_class + 1, max_dim, float(res[pred_class]),
                 "是" if pred_class == max_dim else "否"))


if __name__ == "__main__":
    # main()
    test_vec = [[0.81889086, 0.15229675, 0.31082123, 0.03504317, 0.88920843],
                [0.94963533, 0.5524256, 0.95758807, 0.95520434, 0.84890681],
                [0.90797868, 0.67482528, 0.13625847, 0.34675372, 0.19871392],
                [0.39349776, 0.59416669, 0.92579291, 0.41567412, 0.8358894]]
    predict("model_cross_entropy.bin", test_vec)
