# 补充分析（自动生成）

## 截断率（字符长度超过训练 max_len 的比例）

| 数据集 | 训练 max_len | test 超长句占比 | test 落在截断后的实体占比 |
| --- | ---: | ---: | ---: |
| resume | 512 | 0.0% | 0.00% |
| cluener | 128 | 0.0% | 0.00% |
| msra | 256 | 0.3% | 0.97% |

## 书名号/引号占比（CLUENER 的标注口径）

- cluener: 带书名号/引号的实体 303/2410 = 12.6%
- msra: 带书名号/引号的实体 5/2785 = 0.2%
- resume: 带书名号/引号的实体 0/1052 = 0.0%

## 把书名号从两边去掉后的 F1（只作分析，不进正式表）

| 数据集 | 方法 | strict F1 | 去书名号后 F1 | 差值 |
| --- | --- | ---: | ---: | ---: |
| cluener | bert | 0.7385 | 0.7463 | +0.0079 |
| cluener | llm | 0.6009 | 0.7024 | +0.1016 |
| msra | bert | 0.9320 | 0.9332 | +0.0012 |
| msra | llm | 0.7881 | 0.7881 | +0.0000 |
| resume | bert | 0.9767 | 0.9767 | +0.0000 |

## 各数据集排位

| 数据集 | 第 1 | 第 2 | 第 3 | 第 4 | 第 5 |
| --- | --- | --- | --- | --- | --- |
| resume | DeepSeek 3-shot 0.981 | RoBERTa 0.980 | DeepSeek zero-shot 0.977 | BERT 0.977 | BiLSTM-CRF 0.913 |
| cluener | BERT 0.770 | RoBERTa 0.755 | BiLSTM-CRF 0.656 | DeepSeek 3-shot 0.601 | DeepSeek zero-shot 0.536 |
| msra | RoBERTa 0.935 | BERT 0.921 | BiLSTM-CRF 0.817 | DeepSeek 3-shot 0.788 | DeepSeek zero-shot 0.775 |
