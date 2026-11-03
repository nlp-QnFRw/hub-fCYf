import numpy as np
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
#定义模型
class CrossEntropyModel(nn.Module):
    def __init__(self,input_size) :
        super(CrossEntropyModel,self).__init__()
        self.linear = nn.Linear(input_size,input_size)  # 线性层
    # 模型预测
    def forward(self,x):
        x=self.linear(x)
        return x
# 随机生成一个5维向量，给向量进行分类
def build_dataset(total_sample_num):
    X=[]
    Y=[]
    for i in range(total_sample_num):
        x,y=build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X),torch.LongTensor(Y)
# 返回向量和最大值
def build_sample():
    x=np.random.random(5).astype(np.float32)
    maxV=np.argmax(x)
    return x,maxV
#用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num=100
    x,y=build_dataset(test_sample_num)
    correct = 0
    total = 0
    with torch.no_grad():  # 评估时不需要计算梯度
        outputs = model(x)  # 获取模型输出的 logits (N, 5)
        _, predicted = torch.max(outputs, 1)  # 获取概率最大的类别索引
        
        # 比较预测值和真实标签
        correct += (predicted == y).sum().item()
        total += y.size(0)
        
    accuracy = correct / total
    print(f"测试集准确率: {accuracy:.4f} ({correct}/{total})")
    category_set={}
     # 可选：打印样本分布（调试用）
    category_set = {}
    for y_item in y:
        key = y_item.item()
        category_set[key] = category_set.get(key, 0) + 1
    for key in category_set:
        print(f"本次共采集{key}类样本{category_set[key]}个")
    return accuracy
def main():
     # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    learning_rate = 0.01  # 学习率
    model = CrossEntropyModel(input_size)
        # 定义损失函数：交叉熵损失
    criterion = nn.CrossEntropyLoss()
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
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            # 1. 前向传播：获取 logits
            outputs = model(x)
            loss = criterion(outputs, y)  # 计算loss  model.forward(x,y)
            optim.zero_grad()  # 梯度归零
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重      
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "model.bin")
    plot_log(log)
def plot_log(log):
    if not log:
        return
    epochs = range(1, len(log) + 1)
    accs = [l[0] for l in log]
    losses = [l[1] for l in log]
    
    plt.figure(figsize=(10, 5))
    
    # 绘制准确率
    plt.subplot(1, 2, 1)
    plt.plot(epochs, accs, 'b-o', label='Accuracy')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.ylim(0, 1.1)
    
    # 绘制损失
    plt.subplot(1, 2, 2)
    plt.plot(epochs, losses, 'r-o', label='Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def temperature_scaling(logits, temperature=2.0):
    # 温度T越大，输出的概率分布越平缓；T越小，分布越尖锐
    return logits / temperature
# 使用训练好的模型做预测
def predict(model_path, input_vec_list):
    input_size = 5
    model = CrossEntropyModel(input_size)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))) # 加载训练好的权重
    print(model.state_dict())
    input_tensor = torch.FloatTensor(input_vec_list)
    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        outputs = model(input_tensor)
        scaled_logits = temperature_scaling(outputs, temperature=0.5)
        # 1. 获取预测类别索引
        _, predicted_indices = torch.max(scaled_logits, 1)
        # 2. 获取概率分布 (Softmax)
        probabilities = torch.softmax(scaled_logits, dim=1)
    for i, vec in enumerate(input_vec_list):
        pred_class = predicted_indices[i].item()
        # 获取预测类别对应的概率值
        pred_prob = probabilities[i][pred_class].item()
        # 计算真实标签用于对比
        true_class = np.argmax(vec)
        
        print(f"输入向量: {vec}")
        print(f"  真实类别: {true_class}")
        print(f"  预测类别: {pred_class}")
        print(f"  预测置信度: {pred_prob:.4f}")
        print("-" * 30)
if __name__ == "__main__":
    main()
    # test_vec = [[0.81889086,0.15229675,0.31082123,0.03504317,0.88920843],
    #             [0.94963533,0.5524256,0.95758807,0.95520434,0.84890681],
    #             [0.90797868,0.67482528,0.13625847,0.34675372,0.19871392],
    #             [0.39349776,0.59416669,0.92579291,0.41567412,0.8358894]]
    # predict("model.bin", test_vec)
