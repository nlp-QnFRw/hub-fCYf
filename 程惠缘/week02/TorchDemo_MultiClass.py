# coding:utf8
# 解决 OpenMP 库冲突问题
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ---------------- 全局配置 ----------------
INPUT_SIZE = 5  # 输入向量维度
NUM_CLASSES = 5  # 类别数：0,1,2,3,4
MODEL_PATH = "model_multi.pt"


class MultiClassModel(nn.Module):
    def __init__(self, input_size=INPUT_SIZE, num_classes=NUM_CLASSES):
        super(MultiClassModel, self).__init__()
        # 【改动1】输出维度从 1 改成 num_classes：
        #   (batch, 5) -> (batch, 5)，5 个数字分别是 5 个类别的"打分"
        self.linear = nn.Linear(input_size, num_classes)

        # 【改动2】不需要 sigmoid / softmax 层！
        #   nn.CrossEntropyLoss 内部已经包含 log_softmax，模型只负责输出原始打分
        self.loss = nn.CrossEntropyLoss()

    # 当输入真实标签，返回 loss 值；无真实标签，返回预测打分
    def forward(self, x, y=None):
        logits = self.linear(x)  # (batch, 5)，叫做 logits（未归一化的打分）

        if y is not None:
            # 【改动3】交叉熵：内部 = log_softmax + NLLLoss
            #   所以这里必须喂原始 logits，千万不要先自己做 softmax（否则等于 softmax 两遍）
            return self.loss(logits, y)

        # 预测时返回 logits，需要概率时在外面套一层 softmax 即可
        return logits


def build_sample():
    """生成一个样本：随机 5 维向量，最大值在第几维，就是第几类。"""
    x = np.random.random(INPUT_SIZE)
    y = int(np.argmax(x))  # 【改动4】标签是"类别下标" 0~4 的整数
    return x, y


def build_dataset(total_sample_num):
    """随机生成一批样本。5 个维度同分布，所以 5 个类别理论上各占 20%。"""
    X, Y = [], []
    for _ in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    # 注意标签用 LongTensor（int64），CrossEntropyLoss 要求标签是 long 型
    # 先转成整块 numpy 数组再转 tensor，比直接用 list 快很多
    return (torch.from_numpy(np.array(X, dtype=np.float32)),
            torch.from_numpy(np.array(Y, dtype=np.int64)))


def evaluate(model, total=1000, show_matrix=False):
    """测试准确率；show_matrix=True 时额外打印 5x5 混淆矩阵。"""
    model.eval()
    x, y = build_dataset(total)
    with torch.no_grad():
        logits = model(x)
        # 【改动5】argmax 取分数最大的那一维作为预测类别
        #   注意：argmax 不需要先 softmax，softmax 不改变大小顺序
        pred = torch.argmax(logits, dim=1)

    correct = int((pred == y).sum())
    acc = correct / total
    print("本次预测 %d 个样本，正确 %d 个，准确率：%.4f（随机乱猜的基线 = 1/%d = %.2f）"
          % (total, correct, acc, NUM_CLASSES, 1.0 / NUM_CLASSES))

    if show_matrix:
        # 混淆矩阵：行 = 真实类别，列 = 预测类别
        # 对角线是分对的，非对角线能看出"哪两类容易混"
        cm = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.int64)
        for t, p in zip(y.tolist(), pred.tolist()):
            cm[t, p] += 1
        print("混淆矩阵（行=真实类别，列=预测类别）：")
        print("           pred0  pred1  pred2  pred3  pred4  | 召回率")
        for t in range(NUM_CLASSES):
            row = "true %d   " % t + "".join("%7d" % int(cm[t, p]) for p in range(NUM_CLASSES))
            recall = cm[t, t].item() / max(cm[t].sum().item(), 1)
            print(row + "  | %.3f" % recall)
    return acc


def main():
    # 配置参数（和 TorchDemo.py 完全一致）
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    learning_rate = 0.005  # 学习率

    model = MultiClassModel(INPUT_SIZE, NUM_CLASSES)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_x, train_y = build_dataset(train_sample)
    print("训练集类别分布：", torch.bincount(train_y, minlength=NUM_CLASSES).tolist())

    log = []
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size: (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]
            loss = model(x, y)  # 计算 loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())

        avg_loss = float(np.mean(watch_loss))
        print("\n========= 第%d轮 平均loss：%.4f" % (epoch + 1, avg_loss))
        acc = evaluate(model)  # 每轮评估一次
        log.append([acc, avg_loss])

    # 保存模型
    torch.save(model.state_dict(), MODEL_PATH)

    # 训练结束后看一次混淆矩阵
    print("\n########## 训练完成，最终评估 ##########")
    evaluate(model, total=1000, show_matrix=True)

    # 画图：acc 和 loss 曲线
    plt.figure(figsize=(7, 4))
    plt.plot(range(len(log)), [l[0] for l in log], label="acc", marker="o")
    plt.plot(range(len(log)), [l[1] for l in log], label="loss", marker="s")
    plt.xlabel("epoch")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("train_curve.png", dpi=120)  # 存一张图，方便看
    print("训练曲线已保存到 train_curve.png")
    plt.show()


def predict(model_path, input_vec):
    """用训练好的模型做预测：输出 5 个概率，取最大的那一类。"""
    model = MultiClassModel(INPUT_SIZE, NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))  # 加载训练好的权重
    model.eval()

    with torch.no_grad():
        logits = model(torch.FloatTensor(input_vec))
        # 【改动6】softmax 把 5 个打分变成 5 个概率（和为 1），只在"要看概率"时用
        probs = torch.softmax(logits, dim=1)

    for vec, prob in zip(input_vec, probs):
        cls = int(torch.argmax(prob))
        print("输入：%s" % np.round(vec, 4))
        print("  各类概率：%s" % ", ".join(
            "第%d类=%.4f" % (i, float(prob[i])) for i in range(NUM_CLASSES)))
        print("  => 预测类别：%d，置信度：%.4f" % (cls, float(prob[cls])))


if __name__ == "__main__":
    # main()

    test_vec = [[0.47889086, 0.15229675, 0.31082123, 0.53504317, 0.18920843],
                [0.49635330, 0.55242560, 0.95758807, 0.65520434, 0.84890681],
                [0.48797868, 0.67482528, 0.13625847, 0.34675372, 0.09871392],
                [0.49349776, 0.59416669, 0.92579291, 0.41567412, 0.73588940]]
    print("\n########## 预测示例 ##########")
    predict(MODEL_PATH, test_vec)
