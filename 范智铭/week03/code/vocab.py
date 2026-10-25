''' 读取训练集构建词汇表 '''
import csv
import os
from config import (MAX_LEN,TRAIN_FILE,TEST_FILE,VOCAB_FILE)
import torch

class Vocab:
    def __init__(self, vocab_file=VOCAB_FILE):
        # 初始化id对应词的字典
        self.id2word = {}
        # 初始化词对应id的字典
        self.word2id = {}
        # 初始化默认读取文件如果没有文件则先运行read_all_csv函数
        if not os.path.exists(vocab_file):
            self.read_all_csv()
        with open(vocab_file, 'r', encoding='utf-8') as f:
            for line in f:
                word, _ = line.rstrip('\r\n').split('\t')
                self.id2word[len(self.id2word)] = word
                self.word2id[word] = len(self.id2word) - 1

    #创建读取三个csv文件的函数
    def read_all_csv(self):
        #创建词表集合 并且记录词的出现频率
        vocab = {}
        #读取train.csv文件将第一行以外的数据按字符进行词表构建
        with open(TRAIN_FILE, 'r', encoding='utf-8') as f:
            #创建csv读取器
            reader = csv.reader(f)
            #遍历读取器
            for row in reader:
                #跳过表头
                if row[0] == 'text':
                    continue
                #对row[0]进行字符级分词
                words = list(row[0])
                for word in words:
                    #将words添加到vocab_set中
                    vocab[word] = vocab.get(word, 0) + 1
        # 按照出现频率排序 从高到低排序
        vocab = dict(sorted(vocab.items(), key=lambda x: x[1], reverse=True))
        # 开头添加pad和unk
        vocab = {'<PAD>': 0, '<UNK>': 0} | vocab
        # 将词表转为文件
        with open(VOCAB_FILE, 'w', encoding='utf-8') as f:
            for word, freq in vocab.items():
                f.write(f'{word}\t{freq}\n')
        #打印词表
        print(vocab)

    # 传入一批语句转为id列表
    def texts_to_ids(self, texts):
        idsList = []
        for text in texts:
            ids = []
            # 截断或填充到 MAX_LEN
            text = text[:MAX_LEN]
            for word in text:
                ids.append(self.word2id.get(word, self.word2id['<UNK>']))
            # 填充到 MAX_LEN
            while len(ids) < MAX_LEN:
                ids.append(self.word2id['<PAD>'])
            idsList.append(ids)
        return idsList

    def text_to_ids(self, text):
        ids = []
        # 截断或填充到 MAX_LEN
        text = text[:MAX_LEN]
        for word in text:
            ids.append(self.word2id.get(word, self.word2id['<UNK>']))
        # 填充到 MAX_LEN
        while len(ids) < MAX_LEN:
            ids.append(self.word2id['<PAD>'])
        z = torch.tensor(ids, dtype=torch.long)
        return z

    def get_vocab_size(self):
        return self.word2id, self.id2word


if __name__ == '__main__':
    vocab = Vocab()
    w,i = vocab.get_vocab_size()
    print(len(w))
    print(i)
