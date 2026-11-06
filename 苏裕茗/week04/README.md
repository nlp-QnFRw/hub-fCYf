# 循环神经网络语言模型 —— 字符级预测实验

## 项目说明

本项目基于字符级语言模型，支持 **RNN / LSTM / Transformer** 三种结构切换训练，并可通过浏览器前端进行下一个字的预测与续写演示。

## 做了什么

1. **语料扩充**
   - 原有语料：`corpus_general.txt`（约 1740 万字，通用问答语料）
   - 新增语料：从扫描版 PDF《金融学》474 页中通过 OCR 提取出 `corpus_finance.txt`（约 94 万字，金融专业文本）
   - 合并后总语料约 640 万字，字表上限 4000 字

2. **新增金融 Transformer 模型**
   - 使用 `language_model.py` 在混合语料（`corpus_general.txt + corpus_finance.txt`）上重新训练
   - 模型结构：2 层 × 8 头 Transformer（d_model=128, FFN=256, Pre-LN）
   - 训练参数：`--stride 64`（与 seq_len 相同，非重叠切块，避免样本数爆炸）
   - 输出模型：`best_model_finance_transformer.pt`
   - 验证 PPL：93.69

3. **网页端支持**
   - 修改 `export_lm_model.py`，支持自定义模型注册键名（`--key`）
   - 导出金融模型为 `lm_model_transformer_finance.js`
   - 修改 `next_char_predict.html`，下拉框新增「Transformer·金融」选项
   - 四个模型（Transformer / Transformer·金融 / LSTM / RNN）可在同一页面并列对比

4. **PDF 提取脚本**
   - 编写 `extract_pdf_corpus.py`，支持文字版与扫描版 PDF 自动识别
   - 扫描版自动调用 `pymupdf + EasyOCR` 逐页识别并清洗输出

## 当前结果与问题

**金融模型预测效果不佳，成句率差。**

具体表现：
- 单字预测概率分布合理，但贪心续写或采样生成的句子**语义不连贯、语法断裂感明显**
- 相比通用 Transformer 模型，金融模型在专业术语上的单字命中率有所提升，但整体**长文本生成质量反而下降**

可能原因：
1. **语料比例失衡**：金融语料仅 94 万字，占总语料约 5%，模型未能充分学到金融领域的句法结构
2. **OCR 噪声**：扫描版 PDF 存在识别错误（如"金融"误识为"金触"），噪声随训练融入模型
3. **字符级建模的固有局限**：中文以字为单位，对长距离语义依赖的捕捉能力弱于词级或子词级模型
4. **训练不足**：CPU 训练速度受限，金融模型仅完成默认 20 epoch，可能尚未收敛到最佳点
5. **stride=64 的代价**：非重叠切块导致样本量减少，模型见过的上下文组合多样性下降

## 文件清单

| 文件 | 说明 |
|------|------|
| `language_model.py` | 训练脚本（RNN / LSTM / Transformer） |
| `extract_pdf_corpus.py` | PDF 语料提取脚本 |
| `export_lm_model.py` | 模型导出为网页 JS 脚本 |
| `corpus_general.txt` | 通用语料 |
| `corpus_finance.txt` | 金融语料（OCR 提取） |
| `best_model_*.pt` | 各模型权重 |
| `next_char_predict.html` | 浏览器预测演示页面 |
| `lm_engine.js` | 前端推理引擎 |
| `lm_model_*.js` | 前端模型权重（fp16 + base64） |

## 后续待解决

- [ ] **语料质量**：清洗 OCR 错误，或引入更高质量的金融文本语料
- [ ] **语料比例**：增加金融语料至 500 万字以上，或采用领域自适应/微调策略而非从头混合训练
- [ ] **建模粒度**：尝试子词级（BPE）或词级建模，提升中文语义连贯性
- [ ] **训练策略**：降低 stride（如 32 或 16）增加样本多样性，或增加 epoch / 学习率衰减
- [ ] **评估指标**：补充 BLEU、ROUGE 或人工成句率评测，替代单一 PPL

---
*注：本项目为课程实验，当前结论为初步观察，未做消融实验验证。*
