'''
作业：
    任务目标：做一个五分类任务，通过输入身高，体重，性别，判断该人为正常体重，过轻，过重，肥胖，严重肥胖
    数据集：
        1. 伪造一个数据集，包含1000个样本，每个样本包含身高，体重，性别，BMI值
        2. 通过BMI值判断该人为正常体重，过轻，过重，肥胖，严重肥胖
    模型：
        1. 使用Adam优化器
        2. 损失函数使用交叉熵损失函数
        3. 激活函数使用ReLU
        4. 隐藏层使用50个神经元
        5. 输出层使用5个神经元，每个神经元对应一个类别
'''
import torch
import torch.nn as nn
import torch.optim as optim

# 自动选择设备：优先GPU，没有则回退到CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

data_size = 1000 # 生成数据量
epochs = 1000 # 训练轮数
learning_rate = 0.01 # 学习率
batch_size = 32 # 批量大小
hidden_size = 50 # 隐藏层神经元数
input_size = 3 # 输入层神经元数
output_size = 5 # 输出层神经元数

# 定义模型
class BMIModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        if y is not None:
            loss = self.cross_entropy(x, y)
            return loss
        return x


# 数据集生成
def generate_bmi_data(ds=data_size):
    #随机生成1000个样本,x中身高，体重，性别分别为0，1，2位 身高取值为150-200cm,体重取值为40-120kg,性别取值为0-1 男0女1
    import numpy as np
    np.random.seed(42)
    #生成一个1000*3的矩阵，每行表示一个样本，3列表示身高，体重，性别
    X = np.zeros((ds, 3))
    #随机生成身高
    X[:, 0] = np.random.randint(150, 200, size=ds)
    #随机生成体重
    X[:, 1] = np.random.randint(40, 120, size=ds)
    #随机生成性别
    X[:, 2] = np.random.randint(0, 2, size=ds)
    #遍历X计算x中每个元素的BMI，判断后放进Y数组中
    Y = np.zeros((ds))
    for i in range(ds):
        bmi = X[i, 1] / (X[i, 0] / 100) ** 2
        if bmi < 18.5:
            Y[i] = 0
        elif bmi < 24:
            Y[i] = 1
        elif bmi < 27:
            Y[i] = 2
        elif bmi < 30:
            Y[i] = 3
        else:
            Y[i] = 4
    X = torch.from_numpy(X).float()
    Y = torch.from_numpy(Y).long()
    return X, Y

def train_model():
    # 定义模型并移到GPU
    model = BMIModel().to(device)
    # 获取数据并移到GPU
    X, Y = generate_bmi_data()
    X = X.to(device)
    Y = Y.to(device)
    #定义优化器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        #开始训练
        model.train()
        for i in range(0, len(X), batch_size):
            # 提取当前批次的输入和标签
            x_batch = X[i:i+batch_size]
            y_batch = Y[i:i+batch_size]
            # 前向传播+计算损失
            loss = model(x_batch,y_batch)
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()
            # 清空梯度
            optimizer.zero_grad()
            # 打印损失
        # 设置早停机制
        if loss.item() < 0.1:
            print("损失小于0.1，提前结束训练")
            break
        print(f"第{epoch+1}轮, Loss: {loss.item():.4f}")
        # 保存模型
        print("保存最优模型")
        torch.save(model.state_dict(), 'bmi_model.pth')
    return model

# 评估模型
def evaluate_model(model):
    model.eval()
    X, Y = generate_bmi_data(30)
    X = X.to(device)
    Y = Y.to(device)
    with torch.no_grad():
        y_pred = torch.argmax(model(X), dim=1)
        accuracy = (y_pred == Y).float().mean().item()
    return accuracy
    
if __name__ == '__main__':
    # 用户键盘录入，请选择训练，评估，使用模型
    choice = input("请输入1训练模型，2评估模型，3使用模型：")
    
    if choice == '1':
        model = train_model();
    elif choice == '2':
        model = BMIModel().to(device)
        model.load_state_dict(torch.load('bmi_model.pth', map_location=device))
        model.eval()
        accuracy = evaluate_model(model)
        print(f"测试准确率: {accuracy:.4f}")
    elif choice == '3':
        # 加载模型
        model = BMIModel().to(device)
        model.load_state_dict(torch.load('bmi_model.pth', map_location=device))
        model.eval()
        print("模型加载成功")
        # 用户键盘录入，输入身高，体重，性别
        print("请输入身高，体重，性别，用空格隔开：")
        height, weight, gender = map(float, input().split())
        # 转换为张量并移到GPU
        x = torch.tensor([[height, weight, gender]]).float().to(device)
        # 前向传播
        y_pred = torch.argmax(model(x), dim=1)
        # 打印预测结果
        print(f"预测结果：{y_pred.item()}")
        # 解释预测结果
        if y_pred.item() == 0:
            print("此人为正常体重")
        elif y_pred.item() == 1:
            print("此人为过轻")
        elif y_pred.item() == 2:
            print("此人为过重")
        elif y_pred.item() == 3:
            print("此人为肥胖")
        else:
            print("此人为严重肥胖")

    else:
        print("输入错误")
    

