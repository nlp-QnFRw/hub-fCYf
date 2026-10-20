import torch
import math
import numpy as np
from pathlib import Path
from transformers import BertModel
from transformers import BertTokenizer
'''

关于transformers自带的序列化工具
模型文件下载 https://huggingface.co/models

'''

MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "week6"
    / "text_classification项目"
    / "pretrain_models"
    / "bert-base-chinese"
)

# bert = BertModel.from_pretrained(MODEL_PATH, return_dict=False)
tokenizer = BertTokenizer.from_pretrained(str(MODEL_PATH))

string = "咱呀么老百姓今儿个真高兴"
#分字
tokens = tokenizer.tokenize(string)
print("分字：", tokens)
#编码，前后自动添加了[cls] string [sep]
encoding = tokenizer.encode(string)
print("编码：", encoding)
#文本对编码, 形式[cls] string1 [sep] string2 [sep]
string1 = "今天天气真不错"
string2 = "明天天气怎么样"
encoding = tokenizer.encode(string1, string2)
print("文本对编码：", encoding)



