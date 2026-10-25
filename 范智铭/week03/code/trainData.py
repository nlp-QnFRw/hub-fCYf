import csv
from config import LABELS,TRAIN_FILE,TEST_FILE,DEV_FILE
import torch
from vocab import Vocab

# 加载训练数据
class data:
    def __init__(self):
        vocab = Vocab()
        # 加载训练数据
        self.x_train,self.y_train = self.loadTrain()
        self.x_train = torch.tensor(vocab.texts_to_ids(self.x_train), dtype=torch.long)
        self.y_train = torch.tensor(self.y_train, dtype=torch.long)
        # 加载验证数据
        self.x_dev,self.y_dev = self.loadDev()
        self.x_dev = torch.tensor(vocab.texts_to_ids(self.x_dev), dtype=torch.long)
        self.y_dev = torch.tensor(self.y_dev, dtype=torch.long)
        # 加载测试数据
        self.x_test,self.y_test = self.loadTest()
        self.x_test = torch.tensor(vocab.texts_to_ids(self.x_test), dtype=torch.long)
        self.y_test = torch.tensor(self.y_test, dtype=torch.long)

    def loadTrain(self):
        x_train = []
        y_train = []
        with open(TRAIN_FILE, 'r', encoding='utf-8') as f:
            #创建csv读取器
            reader = csv.reader(f)
            #遍历读取器
            for row in reader:
                #跳过表头
                if row[0] == 'text':
                    continue
                x_train.append(row[0])
                y_train.append(LABELS.index(row[1]))
        return x_train,y_train

    def loadDev(self):
        x_dev = []
        y_dev = []
        with open(DEV_FILE, 'r', encoding='utf-8') as f:
            #创建csv读取器
            reader = csv.reader(f)
            #遍历读取器
            for row in reader:
                #跳过表头
                if row[0] == 'text':
                    continue
                x_dev.append(row[0])
                y_dev.append(LABELS.index(row[1]))
        return x_dev,y_dev
        
    def loadTest(self):
        x_test = []
        y_test = []
        with open(TEST_FILE, 'r', encoding='utf-8') as f:
            #创建csv读取器
            reader = csv.reader(f)
            #遍历读取器
            for row in reader:
                #跳过表头
                if row[0] == 'text':
                    continue
                x_test.append(row[0])
                y_test.append(LABELS.index(row[1]))
        return x_test,y_test

if __name__ == '__main__':
    train_data = data()
    print(train_data.x_train[:10])
    print(train_data.y_train[:10])
