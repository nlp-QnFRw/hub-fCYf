# 遍历 checkpoints 下所有模型，统一在 dev / test 上评估，输出对比日志和对比图
import sys
import importlib
from pathlib import Path

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd

# 确保能 import 同目录下的 config / model / vocab / trainData
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from trainData import data

# ---------------------------------------------------------------- 设置中文字体（Windows）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ---------------------------------------------------------------- 核心函数：按指定 MODEL_TYPE 构建模型并加载权重
def build_model(model_type: str, device: str):
    """临时修改 config.MODEL_TYPE → reload model → 新建 Model → 加载对应权重"""
    config.MODEL_TYPE = model_type
    # CKPT_FILE 也要跟着重新生成（因为 config 里是 f'{MODEL_TYPE}_best.pt'）
    config.CKPT_FILE = config.BASE_DIR / 'checkpoints' / f'{model_type}_best.pt'
    # 动态 reload，让 model.py 里的 import 重新绑定
    import model as model_module
    importlib.reload(model_module)
    model = model_module.Model()
    model.load_state_dict(torch.load(config.CKPT_FILE, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model

# ---------------------------------------------------------------- 在指定数据集上评估
def evaluate(model, x_data, y_data, device, batch_size):
    """返回 (avg_loss, accuracy)"""
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(0, len(x_data), batch_size):
            batch_x = x_data[i:i + batch_size].to(device)
            batch_y = y_data[i:i + batch_size].to(device)
            outputs = model(batch_x)                          # [B, 12]
            loss = criterion(outputs, batch_y)
            running_loss += loss.item() * len(batch_y)
            preds = outputs.argmax(dim=1)
            correct += (preds == batch_y).sum().item()
            total += len(batch_y)
    return running_loss / total, correct / total

# ---------------------------------------------------------------- 主流程
def main():
    ckpt_dir = config.BASE_DIR / 'checkpoints'
    # 扫描所有 *_best.pt 文件，从文件名提取模型类型
    ckpt_files = sorted(ckpt_dir.glob('*_best.pt'))
    if not ckpt_files:
        print(f'⚠️  {ckpt_dir} 下没有找到 *_best.pt 文件')
        return

    # 从文件名推断模型类型
    model_types = [f.stem.replace('_best', '') for f in ckpt_files]
    print(f'发现 {len(model_types)} 个 checkpoint: {model_types}\n')

    # 加载一次数据，所有模型共用
    torch.manual_seed(config.SEED)
    dataset = data()
    DEVICE = config.DEVICE
    BS = config.BATCH_SIZE

    results = []   # 存每个模型的评估结果

    # 逐个模型评估
    for mt in model_types:
        print(f'━━━ 加载 [{mt}] ━━━')
        model = build_model(mt, DEVICE)

        # Dev 集
        dev_loss, dev_acc = evaluate(model, dataset.x_dev, dataset.y_dev, DEVICE, BS)
        print(f'   Dev  →  Loss: {dev_loss:.4f}   Acc: {dev_acc:.4f} ({dev_acc*100:.2f}%)')

        # Test 集
        test_loss, test_acc = evaluate(model, dataset.x_test, dataset.y_test, DEVICE, BS)
        print(f'   Test →  Loss: {test_loss:.4f}   Acc: {test_acc:.4f} ({test_acc*100:.2f}%)')

        # 参数量
        params = sum(p.numel() for p in model.parameters())
        print(f'   Params: {params / 1024:.1f} K  ({params:,})\n')

        results.append({
            '模型': mt.upper(),
            'Dev Loss': round(dev_loss, 4),
            'Dev Acc': round(dev_acc, 4),
            'Test Loss': round(test_loss, 4),
            'Test Acc': round(test_acc, 4),
            '参数量(K)': round(params / 1024, 1),
        })

    # ---------------------------------------------------------------- 汇总表格
    df = pd.DataFrame(results)
    print('=' * 60)
    print('📊 对比汇总')
    print('=' * 60)
    print(df.to_string(index=False))

    # 保存 CSV
    out_dir = Path(__file__).resolve().parent / 'reports'
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / 'model_comparison.csv'
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f'\n✅ 日志已保存 → {csv_path}')

    # ---------------------------------------------------------------- 画图
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('RNN / LSTM / BiLSTM 语言识别效果对比', fontsize=14, fontweight='bold')

    models = df['模型'].tolist()

    # 左图：Accuracy（越大越好）
    x = range(len(models))
    w = 0.35
    axes[0].bar([i - w/2 for i in x], df['Dev Acc'], width=w, label='Dev Acc', color='#4C72B0')
    axes[0].bar([i + w/2 for i in x], df['Test Acc'], width=w, label='Test Acc', color='#DD8452')
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(models)
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('准确率对比（越高越好）')
    axes[0].legend()
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)
    # 在柱子上标数值
    for i, row in df.iterrows():
        axes[0].text(i - w/2, row['Dev Acc'] + 0.005, f"{row['Dev Acc']:.4f}", ha='center', fontsize=8)
        axes[0].text(i + w/2, row['Test Acc'] + 0.005, f"{row['Test Acc']:.4f}", ha='center', fontsize=8)

    # 右图：Loss（越小越好）
    axes[1].bar([i - w/2 for i in x], df['Dev Loss'], width=w, label='Dev Loss', color='#4C72B0')
    axes[1].bar([i + w/2 for i in x], df['Test Loss'], width=w, label='Test Loss', color='#DD8452')
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(models)
    axes[1].set_ylabel('Loss')
    axes[1].set_title('损失对比（越低越好）')
    axes[1].legend()
    axes[1].grid(axis='y', linestyle='--', alpha=0.5)
    for i, row in df.iterrows():
        axes[1].text(i - w/2, row['Dev Loss'] + 0.005, f"{row['Dev Loss']:.4f}", ha='center', fontsize=8)
        axes[1].text(i + w/2, row['Test Loss'] + 0.005, f"{row['Test Loss']:.4f}", ha='center', fontsize=8)

    plt.tight_layout()
    fig_path = out_dir / 'model_comparison.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f'✅ 对比图已保存 → {fig_path}')
    plt.show()


if __name__ == '__main__':
    main()
