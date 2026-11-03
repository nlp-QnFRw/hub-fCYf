import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


# 功能：定义一个5维向量，它最大的数在哪一维，就属于哪一类（第1维最大→第1类，……，第5维最大→第5类）
# 使用交叉熵损失函数来实现
# 标签设计：内部用 argmax 下标 0~4 作为标签，展示时 +1 对应"第1类~第5类"


class TorchModel(nn.Module):
    def __init__(self, input_size):
        # 调用父类nn.Module的初始化方法，pytorch固定写法
        super(TorchModel, self).__init__()
        # 定义一个线性层，输出5个值（5个类别的logits）
        self.linear = nn.Linear(input_size, 5)

        # 交叉熵损失函数
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        x = self.linear(x)  # 得到5维logits（未归一化）

        if y is not None:
            # 训练阶段：交叉熵损失内部自带softmax，必须直接吃logits，不能再提前softmax
            return self.loss(x, y)
        else:
            # 推理阶段：用softmax把logits转成概率分布，方便查看每类概率
            return torch.softmax(x, dim=1)


def build_dataset(sample_num):
    X, Y = [], []

    for i in range(sample_num):
        x, y = build_sample()
        # 把特征加入特征列表
        X.append(x)
        # 标签套一层列表，保证维度对齐：[y]，后面转tensor维度匹配
        Y.append([y])

    X = np.array(X)
    Y = np.array(Y)
    x_tensor = torch.from_numpy(X).float()
    y_tensor = torch.from_numpy(Y).long()
    y_tensor = y_tensor.squeeze()  # 重点：[B,1] -> [B]
    return x_tensor, y_tensor


# 生成一个五维的向量，并得到最大的那个数的下标并返回
def build_sample():
    # 生成一个五维的向量
    x = np.random.random(5)
    idx = x.argmax()  # 获取最大数的下标值，即类别（0~4）

    return x, idx


def evaluate(model):
    # 评估模式
    model.eval()

    test_sample_num = 100
    # 生成测试集数据
    x, y = build_dataset(test_sample_num)

    with torch.no_grad():
        y_pred = model(x)  # [100, 5] 概率分布

        pred = torch.argmax(y_pred, dim=1)  # 预测类别下标 0~4
        correct = torch.sum(pred == y).item()
        acc = correct / len(y)

    # 统计每个类别（第1类~第5类）的预测数量
    class_count = [0] * 5
    for p in pred:
        class_count[p.item()] += 1

    # 打印各类预测数量和准确率
    info = ", ".join("第%d类数量：%d" % (i + 1, class_count[i]) for i in range(5))
    print("各类预测数量：%s" % info)
    print("准确率：%f" % acc)

    return acc


def main():
    # 定义超参数
    learning_rate = 0.01  # 学习率
    epoch_num = 50  # 训练多少轮
    batch_size = 20  # 训练的批次大小
    sample_num = 5000  # 每一轮epoch里总的训练样本数量

    input_size = 5  # 5维向量

    model = TorchModel(input_size)
    # 优化器选择Adam
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

    log = []
    # 生成全部训练数据集
    train_x, train_y = build_dataset(sample_num)

    # 循环训练，遍历每一个epoch
    for epoch in range(epoch_num):
        # 切换模型到训练模式，启用梯度相关逻辑
        model.train()

        watch_loss = []

        for batch_index in range(sample_num // batch_size):
            x = train_x[batch_index * batch_size: (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]

            # 前向传播，计算当前batch的loss
            loss = model(x, y)
            # 反向传播：自动求导，计算每个参数的梯度
            loss.backward()

            optim.step()
            # 梯度清零！pytorch梯度会累加，不清零下一轮梯度就错了
            optim.zero_grad()

            # 把当前batch的loss数值加入列表，.item()取出tensor里的纯数字
            watch_loss.append(loss.item())

        # 打印当前epoch编号、本轮所有batch的平均loss
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        # 调用evaluate，测试本轮模型效果，拿到准确率
        acc = evaluate(model)
        # 把本轮准确率、平均loss存入日志
        log.append([acc, float(np.mean(watch_loss))])

    # 训练完成，保存模型权重到model.bin，只存参数，不存网络结构（标准做法）
    torch.save(model.state_dict(), "model.bin")
    # 打印完整日志数组
    print(log)
    # 绘制准确率曲线：x轴epoch序号，y轴acc，设置图例标签acc
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")
    # 绘制loss曲线：x轴epoch序号，y轴loss，设置图例标签loss
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")
    # 显示图例
    plt.legend()
    # 弹出绘图窗口展示图像
    plt.show()
    return


# 预测函数：加载训练好的权重，输入向量，输出预测结果
def predict(model_path, input_vec):
    # 输入向量维度固定5
    input_size = 5
    # 先实例化模型结构
    model = TorchModel(input_size)
    # 从文件加载保存的权重参数到模型里
    model.load_state_dict(torch.load(model_path))
    # 打印加载后的权重（线性层w和b），方便观察
    print("-" * 50)
    print(model.state_dict())

    # 切换模型为评估推理模式
    model.eval()
    # 推理阶段不计算梯度，节约资源
    with torch.no_grad():
        # 把输入list转成浮点张量，送入模型预测，得到 [N, 5] 的概率分布
        result = model(torch.FloatTensor(input_vec))
    # 取概率最大的下标作为预测类别（0~4），展示时+1对应第1类~第5类
    pred = torch.argmax(result, dim=1)
    # 循环遍历每一条输入向量和对应的预测结果
    for vec, cls, probs in zip(input_vec, pred, result):
        idx = cls.item()
        # 打印输入、预测类别（第几类）、该类概率
        print("输入：%s, 预测类别：第%d类, 该类别概率：%f" % (vec, idx + 1, probs[idx]))


if __name__ == "__main__":
    main()

    # 准备5条待预测的5维测试向量（每行必须且只能有5个元素）
    test_vec = [
        [0.81889086, 0.15229675, 0.31082123, 0.03504317, 0.88920843],
        [0.94963533, 0.5524256, 0.95758807, 0.95520434, 0.84890681],
        [0.90797868, 0.67482528, 0.13625847, 0.34675372, 0.19871392],
        [0.43544546, 0.67482528, 0.67645465, 0.34675372, 0.75565657],
        [0.39349776, 0.59416669, 0.92579291, 0.41567412, 0.8358894]
    ]
    # 调用预测函数，加载model.bin权重，对test_vec做预测
    predict("model.bin", test_vec)
