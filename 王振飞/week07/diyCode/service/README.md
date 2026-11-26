# 本地网页和接口

在 `diyCode` 根目录运行：

```bash
/d/miniconda3/envs/py312/python.exe -m service.app
```

服务监听 `127.0.0.1:8000`；按 Ctrl+C 结束。`GET /` 为抽取页面，`GET /bio` 为 BIO 教学页面。第一次请求 BERT 时加载 checkpoint，后续复用；锁用于保护模型加载和推理。

## POST /api/extract

请求 `{"text":"甲公司拟回购股份。","method":"bert_linear"}`，方法为 `bert_linear`、`bert_crf`、`llm_zero` 或 `llm_few`。最多输入 5000 字。返回 `method`、`events`、`needs_review`、`review_reasons`。空结果、校验警告或调用错误会分别提示；请以返回值为准。

## POST /api/bio

- 训练模式：`{"mode":"training","index":0}`，从训练集选短样本，返回字符标签、真实 token 训练目标及当前 Linear 预测。
- 推理模式：`{"mode":"prediction","text":"某公司拟回购股份。"}`，最多 500 字；返回原始 token 预测、字符投影，以及 `structured_result`。

两种模式均加载 Linear checkpoint。教学展示截断为一次模型输入；推理 JSON 则复用 `/api/extract` 的分段与校验流程，长文时不能与展示区逐个字符对应。原始 token softmax 最大值并非事件正确概率。

该服务是本机教学演示，没有用户认证、线上部署配置、独立事件检测器或规则补全；请勿直接作为生产系统对外暴露。
