import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class TorchModel(nn.Module):
    def __init__(self, input_size:int, hidden1_size:int, hidden2_size:int):
        super(TorchModel, self).__init__()
        self.l1 = nn.Linear(input_size, hidden1_size)
        self.l2 = nn.Linear(hidden1_size, hidden2_size)
        self.activedfunc =torch.sigmoid
        self.loss =nn.CrossEntropyLoss()

    def forward(self,x,y_true=None):
        x =self.l1(x)
        x= self.activedfunc(x)
        y=self.l2(x)
        if y_true is not None:
            return self.loss(y,y_true)
        else:
            return y

def build_test():
    data =np.random.random(5)
    lab = int(np.argmax(data))
    return data,lab

#建立训练集
def build_dataset(n:int):
    X=[]
    Y =[]
    for i in range(n):
        x,y =build_test()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(np.array(X)), torch.LongTensor(np.array(Y))

def train():
    input_size = 5
    hidden_size = 10
    num_classes = 5
    learning_rate = 0.01

    model = TorchModel(input_size, hidden_size, num_classes)
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)

    X_train, Y_train = build_dataset(5000)
    X_val, Y_val = build_dataset(1000)
    print("训练数据:", X_train.shape, Y_train.shape)

    best_loss = float('inf')
    patience = 20
    counter = 0
    log = []

    for ep in range(100000):
        model.train()
        logits = model(X_train)
        loss = model.loss(logits, Y_train)

        opt.zero_grad()
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val)
            val_loss = model.loss(val_logits, Y_val).item()
            pred = val_logits.argmax(dim=1)
            val_acc = (pred == Y_val).float().mean().item()

        log.append([val_acc, val_loss])  

        if val_loss < best_loss - 1e-4:
            best_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), "best.pt")
        else:
            counter += 1

        if ep % 20 == 0:
            print(f"epoch {ep:3d}: train_loss={loss.item():.4f}, "
                  f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

        if counter >= patience:
            print(f"早停于 epoch {ep}")
            break

    model.load_state_dict(torch.load("best.pt"))
    torch.save(model.state_dict(), "model.bin")

    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")
    plt.legend()
    plt.show()

    return model


def predict(model, input_vec):
    model.eval()
    with torch.no_grad():
        x = torch.FloatTensor(np.array(input_vec))
        logits = model(x)
        prob = torch.softmax(logits, dim=1)
        pred = logits.argmax(dim=1)
    for vec, p, pr in zip(input_vec, pred, prob):
        print(f"输入：{np.round(vec, 3)}, 预测类别：{int(p)}, "
              f"概率：{np.round(pr.numpy(), 3)}")


if __name__ == "__main__":
    model = train()

    test_vec = [
        [0.81, 0.15, 0.31, 0.03, 0.88],
        [0.94, 0.55, 0.95, 0.95, 0.84],
        [0.90, 0.67, 0.13, 0.34, 0.19],
        [0.39, 0.59, 0.92, 0.41, 0.83],
    ]
    print("\n预测结果：")
    predict(model, test_vec)
