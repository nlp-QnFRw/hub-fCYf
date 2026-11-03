import numpy as np
import torch
import torch.nn as nn

'''
作业二
设置一个固定五维容器
对于任意五维向量，找到容器占比最大的一维
'''

# 五维容器上限，第 i 维最多能装 capacity_list[i]
capacity_list = [3, 7, 4, 9, 8]
capacity = np.array(capacity_list, dtype=np.float32)

# 配置参数
epoch_num = 100  # 训练轮数
batch_size = 32  # 每次训练样本个数
train_sample = 1000  # 训练集样本总数
test_sample = 1000  # 验证集样本总数（训练时不参与，最后单独验证）
input_size = 5  # 输入向量维度
class_num = 5  # 分类类别数（占比最大的一维有 5 种可能）
learning_rate = 0.02  # 学习率


# 定义模型：五分类，输出 5 个类别的打分（logits）
class PercentNetModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(input_size, class_num)  # 线性层
        self.loss = nn.functional.cross_entropy  # 多分类用交叉熵损失，不需要激活函数

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        y_pred = self.linear(x)  # (batch_size, 5) -> (batch_size, 5)
        # 线性分类模型，不需要激活函数
        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return y_pred  # 输出各类别的 logits


# 随机生成一个5维向量，找到容器占比最大的一维
def build_sample():
    # 每一维在 [0, 该维容量] 内随机取值，相当于随机往容器里装东西
    x = np.random.random(input_size) * capacity
    # 占比 = 该维数值 / 该维容器上限，占比最大的一维就是结果类型
    percent = x / capacity
    y = int(np.argmax(percent))
    return x, y


# 随机生成一批样本
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(np.array(X)), torch.LongTensor(Y)


# 用来测试模型的准确率
def evaluate(model, x, y):
    model.eval()
    with torch.no_grad():
        y_pred = torch.argmax(model(x), dim=1)  # 取打分最高的类别作为预测结果
    correct = (y_pred == y).sum().item()
    return correct / y.size(0), y_pred


# 打印每个类别的准确率
def print_class_acc(y_pred, y):
    for c in range(class_num):
        mask = y == c
        num = int(mask.sum().item())
        if num == 0:
            continue
        acc = (y_pred[mask] == y[mask]).float().mean().item()
        print("  类别%d：样本%4d个，准确率：%f" % (c, num, acc))


# 使用训练好的模型做预测
def predict(model, input_vec_list):
    model.eval()
    x = torch.FloatTensor(input_vec_list)
    with torch.no_grad():
        prob = torch.softmax(model(x), dim=1)  # logits 转成概率
        pred = torch.argmax(prob, dim=1)
    for i, vec in enumerate(input_vec_list):
        percent = [round(float(v) / c, 3) for v, c in zip(vec, capacity_list)]
        true_class = int(np.argmax(np.array(vec) / capacity))
        print("输入向量：%s" % vec)
        print("  各维占比：%s" % percent)
        print("  真实类别：%d，预测类别：%d，置信度：%f"
              % (true_class, pred[i].item(), prob[i][pred[i]].item()))


def main():
    np.random.seed(0)
    torch.manual_seed(0)

    # 1. 生成数据：训练集 + 验证集
    train_x, train_y = build_dataset(train_sample)
    test_x, test_y = build_dataset(test_sample)
    print("训练集样本数：%d，验证集样本数：%d" % (len(train_y), len(test_y)))

    # 2. 初始化模型和优化器
    model = PercentNetModel()
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 3. 启动训练
    for epoch in range(epoch_num):
        model.train()
        index = torch.randperm(train_sample)  # 每轮打乱顺序
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            idx = index[batch_index * batch_size:(batch_index + 1) * batch_size]
            x = train_x[idx]
            y = train_y[idx]
            loss = model(x, y)  # 1) 前向传播并计算loss
            loss.backward()  # 2) 反向传播计算梯度
            optim.step()  # 3) 更新权重
            optim.zero_grad()  # 4) 梯度归零
            watch_loss.append(loss.item())
        acc, _ = evaluate(model, test_x, test_y)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print("第%2d轮  平均loss：%f  验证集准确率：%f"
                  % (epoch + 1, np.mean(watch_loss), acc))

    # 4. 验证训练结果：用没参与训练的样本重新评估
    acc, y_pred = evaluate(model, test_x, test_y)
    print("\n========== 最终验证结果 ==========")
    print("验证集整体准确率：%f（共%d个样本）" % (acc, test_sample))
    print_class_acc(y_pred, test_y)

    # 5. 开发测试样本样本
    print("\n---------- 样例预测 ----------")
    demo_vecs = [
        [2.9, 1.0, 0.5, 1.0, 0.8],  # 第 0 维占比 2.9/3 ≈ 0.967 最大
        [1.5, 6.6, 2.0, 4.5, 3.0],  # 第 1 维占比 6.6/7 ≈ 0.943 最大
        [0.2, 1.0, 3.8, 2.0, 1.0],  # 第 2 维占比 3.8/4 = 0.950 最大
        [0.1, 1.0, 0.2, 8.5, 1.0],  # 第 3 维占比 8.5/9 ≈ 0.944 最大
        [0.5, 2.0, 1.0, 4.0, 7.8],  # 第 4 维占比 7.8/8 = 0.975 最大
    ]
    predict(model, demo_vecs)


if __name__ == "__main__":
    main()
