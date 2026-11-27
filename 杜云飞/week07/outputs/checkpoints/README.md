# ⚠️ 此目录下的 checkpoint 文件因体积过大，未上传到 GitHub

判别式 A 组的 **13 个模型权重（`.pt`，合计约 3.9 GB）** 没有提交到本仓库：

- GitHub 单文件硬上限 **100 MB**，其中 12L 权重单文件就有 **391 MB**、4L 权重 175 MB，13 个合计 3.9 GB；
- 用 Git LFS 托管这么多权重会让仓库 clone 变得很重，也没有必要——
  **它们完全可以从代码 + 派生数据重训出来**（单配置 3–10 分钟，13 个配置约 2 小时）。

## 缺失的文件清单

| 文件名 | 配置 | 体积 |
|---|---|---|
| `bert12L_linear.pt` / `bert12L_crf.pt` | bert-base-chinese，12 层，Linear / CRF | 391 MB ×2 |
| `bert4L_linear.pt` / `bert4L_crf.pt` | bert-base-chinese，4 层，Linear / CRF | 175 MB ×2 |
| `bert4Lrand_crf.pt` | bert 4 层**随机初始化**（初始化消融对照） | 175 MB |
| `roberta12L_linear.pt` / `roberta12L_crf.pt` | chinese-roberta-wwm-ext，12 层 | 391 MB ×2 |
| `roberta4L_linear.pt` / `roberta4L_crf.pt` | chinese-roberta-wwm-ext，4 层 | 175 MB ×2 |
| `macbert12L_linear.pt` / `macbert12L_crf.pt` | chinese-macbert-base，12 层 | 391 MB ×2 |
| `macbert4L_linear.pt` / `macbert4L_crf.pt` | chinese-macbert-base，4 层 | 175 MB ×2 |

## 想看结果 / 想复评怎么办？

**不需要权重也能看全部结果**，仓库里已经上传了：

- `outputs/metrics/disc/*.json` —— 44 份指标文件，每个配置的 trigger / argument / event F1、内层验证 spanF1、
  argv、epoch 预算、训练时长、峰值显存、截断句数、n200 投影指标全在里面；
- `outputs/disc/pred_*.jsonl` —— 22 份统一格式的预测结果（判别式 4087 行 / LLM 200 行），
  直接喂给 `common/evaluate_events.py` 就能重算指标，不需要重新推理。

**要重新生成权重**，一条命令跑完整个网格（断点续跑，`--skip-done` 会跳过已有产物）：

```bash
python disc/run_grid.py --skip-done --deadline 07:00
```

权重默认落在 `outputs/checkpoints/`，命名规则 `{encoder}{层数}_{decoder}.pt`（如 `macbert12L_crf.pt`）。
每个 `.pt` 里同时存了 `state_dict` 和 `label_scheme`，所以**单独加载一个 checkpoint 就能复评**，不依赖训练时的 label 顺序。
