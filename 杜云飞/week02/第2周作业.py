#region 0. 任务说明：MNIST 手写数字十分类（仅全连接层 + 激活函数）
# 数据集：MNIST 手写数字，训练集 60000 张，测试集 10000 张，每张 28×28 灰度图。
#   输入：28×28 灰度图 → 展平成 784 维向量（全连接层要求一维输入）
#   类别：10 类，数字 0~9
# 任务：输入一张手写数字图片，输出它是 0~9 中的哪一个数字（10 分类）。
#
# 为什么全连接层能搞定：
#   1. MNIST 图像简单，像素本身就是有效特征，不需要卷积提取局部特征；
#   2. 全连接层参数量足够（约 53 万），能学会像素组合；
#   3. 业界经典 MLP 在 MNIST 上可达 97%~98% 准确率。
#   注意：输入必须先展平（flatten）成 (B, 784)，这是全连接层的要求。

# 模型结构（仅全连接层 + ReLU）：
#   输入 784 → Linear(784,512) → ReLU → Linear(512,10) → CrossEntropyLoss（只用了两层）
#
# 训练配置：Adam，lr=0.001，batch=128，epoch=10，评估总体准确率 + 每类准确率 + 样例核对。

#region 1. 导入库
import os                                      
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision                              # 提供 MNIST 数据集
import torchvision.transforms as transforms     # 图像预处理工具
from torchvision.utils import save_image        # 样例核对时存图用
#endregion


#region 2. 构建数据集    【数据集:MNIST，训练 60000，测试 10000，28×28 灰度】
transform = transforms.Compose([
    transforms.ToTensor(),                          # PIL → 张量 (1,28,28)，像素值 ∈ [0,1]
]) #.Compose([变换1,变换2])，依次执行变换   .Totensor=转张量+通道提前+除以 255。

#设置训练集，数据存'./data,'训练集，本地没有时联网下载，对图片进行预处理   60000 张，每张 (1,28,28)
train_dataset = torchvision.datasets.MNIST(root='./data',train=True,download=True,transform=transform)
#设置测试集   10000 张，每张 (1,28,28)
test_dataset = torchvision.datasets.MNIST(root='./data',train=False,download=True,transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)   # 训练集打乱，每个 batch (128,1,28,28)
test_loader  = DataLoader(test_dataset,  batch_size=512, shuffle=False)  # 测试集不打乱，每个 batch (512,1,28,28)
#endregion

#region 3. 定义网络
class MNISTMLP(nn.Module):
    """784维输入→10类输出的全连接分类器，结构：Linear(784,512) → ReLU → Linear(512,10)"""
    def __init__(self):
        super().__init__() #初始化父类
        self.fc1 = nn.Linear(784,512)  #输入 784 → 隐藏 512
        self.bn1 = nn.BatchNorm1d(512) #做归一化
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512,128) 
        self.bn2 = nn.BatchNorm1d(128) #做归一化
        self.fc3 = nn.Linear(128,10)   #尝试直接使用两层网络，隐藏512 → 10类logits
    def forward(self,x):
        """前向传播：输入(B,1,28,28),展平为(B,784)，输出(B,10)logits"""
        x = x.view(x.size(0),784)  #展平，x的总维度为B×1×28×28，重新解释形状，x.size(0)为B，-1自动计算，则为1×28×28=784
        x = self.bn1(self.fc1(x))  #全连接层+归一化
        x = torch.relu(x) #(B,784)→(B,512)，ReLU 激活
        x = self.dropout(x)
        x = self.bn2(self.fc2(x))  #全连接层+归一化
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)  #(B,512)→(B,10)，输出 logits，不加激活
        return x  #返回10类logits
model = MNISTMLP() #实例化模型
loss_fn = nn.CrossEntropyLoss()  #交叉熵损失，内部含softmax
optimizer = optim.Adam(model.parameters(),lr=0.001) #Adam优化器，学习率0.001
#endregion

#region 4.评估和训练
epochs = 10 #训练轮数
for epoch in range(epochs):  #逐轮训练
    #region 4.1 训练阶段
    model.train()  #切换到训练模式
    total_loss = 0 #累计损失
    for xb,yb in train_loader:  #逐个batch取数据和标签，loader返回值取决于dataset数量，xb:(B,1,28,28)，yb: (B,)
        optimizer.zero_grad() #清空上一步的梯度
        logits = model(xb) #前向传播，(B,1,28,28) → (B,10)
        loss = loss_fn(logits,yb)  #返回的是当前batch内所有样本的平均损失
        loss.backward() #反向传播
        optimizer.step() #更新参数
        #累加“损失 × 样本数”，便于算平均,.item()把张量转成普通python float，脱离计算图，xb.size(0)第0维的大小
        total_loss += loss.item()*xb.size(0)  
    avg_loss = total_loss/len(train_dataset) #本轮平均损失
    #endregion
    #region 4.2 测试阶段
    model.eval() #切换到评估模式
    correct = total = 0 #correct:预测对的样本数，total：总样本数，初始化
    with torch.no_grad(): #不计算梯度
        for xb,yb in test_loader: #逐batch取测试数据
            pred = torch.argmax(model(xb),dim=1) #取logits最大值的下标最为预测类别，(B,)
            correct += (pred==yb).sum().item() #计算正确预测数，.item()把 0 维张量里的值取出来，变成普通 Python 标量
            total += yb.size(0) 
    acc = correct / total  # 测试集准确率
    print(f"Epoch {epoch+1:02d} | Loss: {avg_loss:.4f} | Test Acc: {acc:.4f}")  # 打印当前轮损失与准确率
    #endregion
#endregion

#region 5. 最终测试+计算每类准确率
model.eval() 
all_pred = [] #空列表，放所有batch的预测结果和真实标签
all_label= []
with torch.no_grad():
    for xb,yb in test_loader:  
        all_pred.append(torch.argmax(model(xb),dim=1)) #添加预测到列表，每次是一个列表
        all_label.append(yb) #收集真实标签
all_pred  = torch.cat(all_pred)   # 拼成 (10000,)，之前是个带列表的列表
all_label = torch.cat(all_label)  # 拼成 (10000,)
print("\n最终测试准确率：", (all_pred == all_label).float().mean().item())  # 打印总体准确率
for k in range(10):  # 逐类别统计
    mask = (all_label == k)  # 布尔掩码，选出真实标签为 k 的样本
    acc_k = (all_pred[mask] == all_label[mask]).float().mean().item()  # 该类准确率
    print(f"数字 {k} 准确率：{acc_k:.4f}")  # 打印每个数字的准确率



"""
当使用两层全连接层时：
最终测试准确率： 0.98089998960495
数字 0 准确率：0.9908
数字 1 准确率：0.9894
数字 2 准确率：0.9797
数字 3 准确率：0.9871
数字 4 准确率：0.9786
数字 5 准确率：0.9798
数字 6 准确率：0.9802
数字 7 准确率：0.9786
数字 8 准确率：0.9692
数字 9 准确率：0.9742
当使用三层全连接层时：
Epoch 01 | Loss: 0.2059 | Test Acc: 0.9698
Epoch 02 | Loss: 0.0755 | Test Acc: 0.9756
Epoch 03 | Loss: 0.0479 | Test Acc: 0.9780
Epoch 04 | Loss: 0.0343 | Test Acc: 0.9778
Epoch 05 | Loss: 0.0264 | Test Acc: 0.9805
Epoch 06 | Loss: 0.0212 | Test Acc: 0.9792
Epoch 07 | Loss: 0.0178 | Test Acc: 0.9773
Epoch 08 | Loss: 0.0143 | Test Acc: 0.9813
Epoch 09 | Loss: 0.0134 | Test Acc: 0.9805
Epoch 10 | Loss: 0.0127 | Test Acc: 0.9823
当添加两层dropout时：
Epoch 01 | Loss: 0.2838 | Test Acc: 0.9642
Epoch 02 | Loss: 0.1222 | Test Acc: 0.9759
Epoch 03 | Loss: 0.0962 | Test Acc: 0.9760
Epoch 04 | Loss: 0.0771 | Test Acc: 0.9801
Epoch 05 | Loss: 0.0677 | Test Acc: 0.9807
Epoch 06 | Loss: 0.0619 | Test Acc: 0.9801
Epoch 07 | Loss: 0.0557 | Test Acc: 0.9819
Epoch 08 | Loss: 0.0472 | Test Acc: 0.9824
Epoch 09 | Loss: 0.0443 | Test Acc: 0.9820
Epoch 10 | Loss: 0.0409 | Test Acc: 0.9831
"""
