# nlp_roberta_backbone_base_std 实验记录汇总

数据来源：`bert_ckpt_parameters.txt`

## 实验记录表

实验名对应 `bert_train_log_N.json`，N 见首列。

|实验 | pool | epochs | no_class_weight | label_smoothing | focal_gamma | fgm | fgm_eps | val Macro-F1 |
|---|---|---|---|---|---|---|---|---|
| 6  | cls  | 3  | False | 0.0 | 0.0 | False | 1.0 | 0.5669 |
| 7  | cls  | 10 | False | 0.0 | 0.0 | False | 1.0 | 0.5821 |
| 8  | max  | 10 | False | 0.0 | 0.0 | False | 1.0 | 0.5671 |
| 9  | mean | 10 | False | 0.0 | 0.0 | False | 1.0 | 0.5646 |
| 10 | cls  | 10 | True  | 0.0 | 0.0 | False | 1.0 | 0.5725 |
| 11 | cls  | 10 | False | 0.1 | 0.0 | False | 1.0 | 0.5774 |
| 12 | cls  | 10 | False | 0.0 | 2.0 | False | 1.0 | 0.5809 |
| 13 | cls  | 10 | False | 0.0 | 0.0 | True  | 1.0 | 0.5820 |
| 14 | cls  | 10 | False | 0.0 | 0.0 | True  | 0.5 | **0.5833** |

## 固定参数

全部实验共用、未变化的参数：

| 参数 | batch_size | lr | head_lr_mult | max_length | max_steps | max_eval | no_save |
|---|---|---|---|---|---|---|---|
| 取值 | 64 | 2e-5 | 5.0 | 64 | 0 | 0 | False |

## 说明

- 最佳结果是 `log_14`（cls 池化 + FGM ε=0.5），F1 = 0.5833；`log_7`（0.5821）与 `log_13`（0.5820）紧随其后。
- `pool=cls`（0.5669~0.5833）整体优于 `mean`（0.5646）和 `max`（0.5671）。



# LLM和tf-idf+svm实验记录

## 实验记录表格

| 方法 | 准确率 | Macro-F1 | 解析失败 |
|---|---|---|---|
| Qwen3 + lora | 0.538 | 0.4873 | 0 |
| Qwen3 + full_ft | 0.544 | **0.5093** | 0 |
| Qwen3 zero-shot | 0.165 | 0.1094 | 0 |
| TF-IDF+SVM (ngram=1,2, C=1.0) | 0.5296 | 0.5087 | 0 |

最佳结果是 Qwen3 + full_ft（0.5093）,但是相比roberta-base-str(0.5833)差了0.074（7.4%）