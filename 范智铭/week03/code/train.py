# 训练循环
import torch
import torch.nn as nn

from config import (BATCH_SIZE, EPOCHS, LR, PATIENCE, GRAD_CLIP, DEVICE, LABELS, SEED,CKPT_FILE,NUM_CLASSES)
from model import Model
from trainData import data
from vocab import Vocab

# 训练模型
def train(model):
    # 加载训练数据
    train_data = data()
    best_acc = 0.0                              # 早停指标：dev accuracy，越大越好
    x_train = train_data.x_train
    y_train = train_data.y_train
    # 定义优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    # 定义交叉熵损失函数
    criterion = nn.CrossEntropyLoss()
    patience = 0
    #开始训练循环
    for epoch in range(EPOCHS):
        running_loss = 0.0
        model.train()
        indices = torch.randperm(len(x_train))         # 每轮都重新打乱
        # 以BATCH_SIZE为批次进行训练
        for i in range(0, len(x_train), BATCH_SIZE):
            batch_idx = indices[i : i + BATCH_SIZE]
            batch_x = x_train[batch_idx]
            batch_y = y_train[batch_idx]
            inputs, labels = batch_x.to(DEVICE), batch_y.to(DEVICE)
            # 梯度清零
            optimizer.zero_grad()
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            # 更新参数
            optimizer.step()
            running_loss += loss.item() * len(labels)
        print(f'Epoch {epoch+1}/{EPOCHS}  Train Loss: {running_loss/len(x_train):.4f}')

        # 每轮都评估 dev
        _, dev_acc = test(model, train_data, 'dev')
        if dev_acc > best_acc:
            best_acc = dev_acc
            torch.save(model.state_dict(), CKPT_FILE)
            patience = 0
        else:
            patience += 1
            if patience >= PATIENCE:
                print('早停触发，验证集 Acc 连续多轮无改善')
                break

# 评估模型：传入 model 和数据集类型（'test' 或 'dev'）
def test(model,test_data, type='test'):
    if type == 'test':
        x_data, y_data = test_data.x_test, test_data.y_test
    elif type == 'dev':
        x_data, y_data = test_data.x_dev, test_data.y_dev
    else:
        raise ValueError(f"未知数据集类型: {type}")

    criterion = nn.CrossEntropyLoss()
    model.eval()
    with torch.no_grad():
        running_loss = 0.0
        correct = 0
        total = 0
        for i in range(0, len(x_data), BATCH_SIZE):
            batch_x = x_data[i:i+BATCH_SIZE]
            batch_y = y_data[i:i+BATCH_SIZE]
            inputs, labels = batch_x.to(DEVICE), batch_y.to(DEVICE)
            outputs = model(inputs)                          # [B, 12]
            loss = criterion(outputs, labels)
            running_loss += loss.item() * len(labels)
            preds = outputs.argmax(dim=1)                    # [B] — 每条的预测类别索引
            correct += (preds == labels).sum().item()
            total += len(labels)
        acc = correct / total
        print(f'[{type}] Loss: {running_loss/total:.4f}  Acc: {acc:.4f}')
    return running_loss / total, acc

if __name__ == '__main__':
    CKPT_FILE.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    # 加载模型
    model = Model()
    model.to(DEVICE)
    while True:
        # 用户键盘录入，请选择训练，评估，使用模型
        choice = input("请输入1训练模型，2评估模型，3使用模型，4结束：")

        if choice == '1':
            # 训练模型
            train(model);
        elif choice == '2':
            # 加载模型
            model.load_state_dict(torch.load(CKPT_FILE, map_location=DEVICE))
            print("模型加载成功")
            # 评估模型
            test(model,data());
        elif choice == '3':
            # 加载模型
            model.load_state_dict(torch.load(CKPT_FILE, map_location=DEVICE))
            model.eval()
            print("模型加载成功")
            print("请输入语句：")
            sentence = input()
            # 转换为张量并移到GPU
            vocab = Vocab()
            x = vocab.text_to_ids(sentence).unsqueeze(0).to(DEVICE)  # [64] -> [1, 64]
            with torch.no_grad():
                logits = model(x)  # [1, 12]
            y_pred = logits.argmax(dim=1).item()
            print(f"预测结果：{LABELS[y_pred]}  置信度：{logits.softmax(1)[0, y_pred]:.3f}")
        elif choice == '4':
            print("结束")
            break
        else :
            print("输入错误")
