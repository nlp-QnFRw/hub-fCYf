"""sftlab —— 一个从零实现的最小 SFT（监督微调）实验框架。

模块划分：
    tokenizer 字符级分词器（不依赖任何预训练权重）
    template  chat template + loss mask（本项目的核心教学点）
    data      jsonl 读取、样本编码、Dataset / collate
    model     小型 decoder-only Transformer
    generate  采样与生成
    train     通用训练循环
    utils     配置、随机种子、日志等杂项
"""

__version__ = "0.1.0"
