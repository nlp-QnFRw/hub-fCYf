"""BERT 训练入口：python -m bert.train --decoder linear|crf。

先从训练集划分内部验证集，按验证损失保存最佳模型；独立 dev 集只在评测时使用。
"""
import argparse, json, math, random
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from bert.dataset import EventDataset, collate
from bert.model import EventModel
from common.event_data import BERT, DATA, ROOT, labels_for, read_jsonl, schema_at

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--decoder', choices=['linear', 'crf'], default='linear')
    ap.add_argument('--data-dir', type=Path, default=DATA)
    ap.add_argument('--model-path', type=Path, default=BERT)
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--batch-size', type=int, default=2)
    ap.add_argument('--grad-accum', type=int, default=4)
    ap.add_argument('--max-length', type=int, default=256)
    ap.add_argument('--lr', type=float, default=2e-05)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.grad_accum < 1 or (args.max_length < 4):
        ap.error('训练参数必须为正，max-length 至少为 4')
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device)
    rows = read_jsonl(args.data_dir / 'train_extract.jsonl')
    if len(rows) < 2:
        raise ValueError('训练集至少需要两条样本')
    # 内部验证只从训练集中划分，不用 dev 选择模型。
    random.shuffle(rows)
    n = max(1, len(rows) // 10)
    val, train = (rows[:n], rows[n:])
    # 标签表和 tokenizer 必须在训练与推理时保持一致。
    labels = labels_for(schema_at(args.data_dir))
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), use_fast=True, local_files_only=True)

    def loader(rows, shuffle):
        return DataLoader(EventDataset(rows, tokenizer, labels, args.max_length, args.decoder == 'crf'), batch_size=args.batch_size, shuffle=shuffle, collate_fn=collate)
    train_dl, val_dl = (loader(train, True), loader(val, False))
    model = EventModel(args.model_path, len(labels), args.decoder).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    # 梯度累积每 grad_accum 个 batch 才更新一次，降低单次显存需求。
    total_steps = math.ceil(len(train_dl) / args.grad_accum) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * 0.1), total_steps)
    output = ROOT / 'outputs'
    output.mkdir(exist_ok=True)
    best = float('inf')
    history = []
    print(f'设备={device} 训练={len(train)} 内部验证={len(val)} 标签={len(labels)}', flush=True)
    for epoch in range(1, args.epochs + 1):
        # 一轮训练：前向计算损失 -> 反向传播 -> 梯度裁剪 -> 参数更新。
        model.train()
        optimizer.zero_grad()
        total = 0
        for i, batch in enumerate(train_dl):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch)
            (loss / args.grad_accum).backward()
            total += loss.item()
            if (i + 1) % args.grad_accum == 0 or i + 1 == len(train_dl):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            if (i + 1) % 50 == 0:
                print(f'epoch {epoch} step {i + 1}/{len(train_dl)} loss={total / (i + 1):.4f}', flush=True)
        # 在内部验证集上选择模型，不计算梯度，也不更新参数。
        model.eval()
        val_loss = 0
        with torch.inference_mode():
            for batch in val_dl:
                val_loss += model(**{k: v.to(device) for k, v in batch.items()}).item()
        val_loss /= len(val_dl)
        entry = {'epoch': epoch, 'train_loss': total / len(train_dl), 'val_loss': val_loss}
        history.append(entry)
        print(json.dumps(entry, ensure_ascii=False), flush=True)
        # 保存权重、标签表和模型配置，推理时才能正确恢复。
        if val_loss < best:
            best = val_loss
            torch.save({'state_dict': model.state_dict(), 'labels': labels, 'decoder': args.decoder, 'max_length': args.max_length, 'model_path': str(args.model_path.resolve()), 'epoch': epoch}, output / f'bert_{args.decoder}.pt')
        (output / f'bert_{args.decoder}_history.json').write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding='utf-8')
    output_path = output / f'bert_{args.decoder}.pt'
    print(f'完成，模型：{output_path}')
if __name__ == '__main__':
    main()
