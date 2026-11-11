#加载数据
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from my_module import MyModule
from tokenizer import CharTokenizer
from util.dataset import MyDataset


def set_seed(seed: int):
    """固定所有随机种子，保证结果可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_data(cfg, tok):
    """用同一个 tokenizer 构建 train / test DataLoader。"""
    train_dataset = MyDataset(cfg, tok, 'train')
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.hparams.batch_size,
        shuffle=True,
        pin_memory=True,
    )
    test_dataset = MyDataset(cfg, tok, 'test')
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=cfg.hparams.batch_size,
        shuffle=False,
        pin_memory=True,
    )
    return train_dataloader, test_dataloader


def evaluate(model, loader, criterion, device):
    """在 test set 上跑一遍，返回平均 loss。"""
    model.eval()
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out.reshape(-1, out.size(-1)), y.reshape(-1))
            total_loss += loss.item()
            n += 1
    model.train()
    return total_loss / n


def train(cfg, train_dataloader, test_dataloader, tok, device):
    set_seed(cfg.hparams.seed)

    model = MyModule(cfg).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=tok.pad_id)
    optimizer = optim.Adam(model.parameters(), lr=cfg.hparams.lr)

    # checkpoint 保存目录
    ckpt_dir = Path(__file__).resolve().parent.parent / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    bad_epochs = 0  # test loss 没下降的连续 epoch 数
    patience = getattr(cfg.hparams, "early_stop_patience", 0)
    print(f"训练开始 | device={device} | 每 epoch {len(train_dataloader)} 步 | early_stop_patience={patience}")

    for epoch in range(cfg.hparams.epochs):
        model.train()
        running_loss = 0.0
        t0 = time.time()

        for i, (inputs, labels) in enumerate(train_dataloader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            logits = outputs.reshape(-1, outputs.size(-1))
            targets = labels.reshape(-1)
            loss = criterion(logits, targets)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item()

            # 每 100 步打印一次 step 级日志
            if (i + 1) % 100 == 0:
                avg = running_loss / (i + 1)
                print(f"  Epoch {epoch+1} [{i+1}/{len(train_dataloader)}] | step_loss {loss.item():.4f} | avg {avg:.4f}")

        train_loss = running_loss / len(train_dataloader)
        val_loss = evaluate(model, test_dataloader, criterion, device)
        elapsed = time.time() - t0
        ppl = math.exp(val_loss) if val_loss < 100 else float("inf")

        print(
            f"Epoch {epoch+1}/{cfg.hparams.epochs} | "
            f"train {train_loss:.4f} | test {val_loss:.4f} | ppl {ppl:.1f} | "
            f"{elapsed:.1f}s"
        )

        # 保存 best model + 早停计数
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            bad_epochs = 0
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "cfg": OmegaConf.to_container(cfg, resolve=True),
                "epoch": epoch,
                "val_loss": val_loss,
            }, ckpt_dir / "best.pt")
            print(f"  ✓ 新 best，已保存到 checkpoints/best.pt")
        else:
            bad_epochs += 1
            print(f"  test loss 未下降 ({bad_epochs}/{patience})")

        # 早停判断
        if patience > 0 and bad_epochs >= patience:
            print(f"\n早停触发：test loss 连续 {patience} 个 epoch 未下降，停止训练")
            break


if __name__ == '__main__':
    cfg = OmegaConf.load(Path(__file__).resolve().parent.parent / "config" / "config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # tokenizer 只加载一次，复用给 MyModule 和 Dataset
    tok = CharTokenizer.from_file(cfg)
    train_dataloader, test_dataloader = load_data(cfg, tok)
    train(cfg, train_dataloader, test_dataloader, tok, device)
