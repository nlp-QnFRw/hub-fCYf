import pandas as pd
import torch
import torch.nn as nn


# 处理数据，并返回训练集和测试集合
def data_handle():
    # 读取数据，16特征 7分类
    data = pd.read_excel("./data/Dry_Bean_Dataset.xlsx")
    # 特征和标签分离，转换成float 考虑数据精度问题,并转换成张量
    x = torch.FloatTensor(data.iloc[:, :16].to_numpy(dtype=float))
    # 标签转化成 0,1,2
    y_temp = torch.LongTensor(pd.Categorical(data.iloc[:, 16]).codes)
    # 转换成 One-hot
    y = nn.functional.one_hot(y_temp, num_classes=7).float()
    # 原数据集是顺序存储，先打乱,并返回随机的索引集合
    random = torch.randperm(len(x))
    # 先取出 10% 作为验证集
    v_index = int(len(x) * 0.9)
    x_v = x[random[v_index:]]
    y_v = y[random[v_index:]]
    # 剩下 90% 留下训练
    x = x[random[:v_index]]
    y = y[random[:v_index]]
    # 剩下的 8:2分，这里需要重新随机
    random = torch.randperm(len(x))
    split = int(len(x) * 0.8)
    x_train = x[random[:split]]
    x_test = x[random[split:]]
    y_train = y[random[:split]]
    y_test = y[random[split:]]

    # 计算训练集每个特征的均值和标准差
    mean = x_train.mean(dim=0)
    std = x_train.std(dim=0)
    # 防止标准差为0
    std = std.clamp_min(1e-8)
    # 训练集标准化
    x_train = (x_train - mean) / std
    # 测试集标准化
    x_test = (x_test - mean) / std
    # 验证集标准化
    x_v = (x_v - mean) / std
    return x_train, y_train, x_test, y_test, x_v, y_v


# 学习率
lr = 0.001
# 轮次
epoch_num = 100
# Mini_Batch
batch_size = 10


# 交叉熵
def cross_entropy(y_p, y):
    return -(y * torch.log(y_p.clamp_min(1e-12))).sum(dim=1).mean()


class DemoModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.hide_linear1 = nn.Linear(16, 32)
        self.hide_linear2 = nn.Linear(32, 16)
        # 一层隐藏层
        self.out_linear = nn.Linear(16, 7)
        # softmax 作为最外层激活函数，dim = 1 代表按照行的维度计算概率值
        self.last_activation = nn.Softmax(dim=1)
        # 损失函数为交叉熵
        self.loss = cross_entropy
        # relu 激活函数
        self.activation = nn.ReLU()

    # 定义前向传播，返回损失值
    def forward(self, x, y=None):
        # 预测时返回概率
        prop1 = self.activation(self.hide_linear1(x))
        prop2 = self.activation(self.hide_linear2(prop1))
        prop = self.last_activation(self.out_linear(prop2))
        if y is None:
            return prop
        # 训练时返回损失
        return self.loss(prop, y)


# 计算准确率
def calc_acc(model: DemoModule, x_test, y_test):
    # 切换至评估模式
    model.eval()
    with torch.no_grad():
        # 预测概率
        prop = model(x_test)
        # 获取最大概率的索引位置
        predict = prop.argmax(dim=1)
        true = y_test.argmax(dim=1)
        # 计算准确率
        acc = (predict == true).float().mean()
    return acc.item()

def calc_loss(model, x, y):
    model.eval()

    with torch.no_grad():
        loss = model(x, y)

    return loss.item()

if __name__ == '__main__':
    # 数据处理,并返回测试机和验证集合
    x_train, y_train, x_test, y_test, x_v, y_v = data_handle()
    # 定义模型
    model = DemoModule()
    # 梯度下降优化器，目前选择随机梯度下降SGD
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    acc_sum = 0
    for epoch in range(epoch_num):
        avg_loss = 0
        model.train()
        sum_loss = 0
        # 每轮打乱顺序，原数据集按照顺序排列
        indices = torch.randperm(len(x_train))
        for index in indices:
            # pytorch 推荐写法，自动调用 forward
            loss = model(x_train[index:index + 1],
                         y_train[index:index + 1])
            # 反向传播，计算梯度
            loss.backward()
            # 根据梯度更新权重
            optim.step()
            # 清空梯度
            optim.zero_grad()
            # 计算梯度
            sum_loss += loss.item()
        avg_loss = sum_loss / len(x_train)
        acc = calc_acc(model, x_test, y_test)
        acc_sum += acc
        print(f'第{epoch + 1}轮准确率:{acc},损失：{avg_loss}')
        if acc > 0.95:
            print(f'准确率为{acc_sum / epoch + 1}')
            break
    print("完成")
