import os
import sys
import time
import json
import argparse
import subprocess


"""
通过这组实验获取最优的epochs，和最优和池化方式，还有是否使用class_weight
"""
TRAIN_COMMAND = [
    # "python bert/train_bert.py",  # 基线
    # "python bert/train_bert.py --pool cls --epochs 10", # 基线到10 epochs
    "python bert/train_bert.py --pool max --epochs 10",  # max pool
    "python bert/train_bert.py --pool mean --epochs 10", # mean pool
    "python bert/train_bert.py --pool cls --epochs 10 --no_class_weight", # 关闭class weight
]

# 这组实验算出最好的参数
TRAIN_COMMAND_exp2 = [
    "python bert/train_bert.py --label_smoothing 0.1 --epochs 10",      # 软标签（常取 0.05~0.1）
    "python bert/train_bert.py --focal_gamma 2 --epochs 10",            # Focal Loss（与类别加权共用同一张 alpha 表）
    "python bert/train_bert.py --fgm --epochs 10",                      # FGM 对抗训练（训练时间约 ×1.5）
    "python bert/train_bert.py --fgm --fgm_eps 0.5 --epochs 10",        # 扰动强度默认 1.0，可调小
]



def main():
    """按顺序循环运行所有训练命令。"""
    total = len(TRAIN_COMMAND_exp2)
    work_dir = os.path.dirname(os.path.abspath(__file__))

    print(f"长度：{total}")
    for idx, cmd in enumerate(TRAIN_COMMAND_exp2, start=1):
        print(f"\n[{idx}/{total}] 开始运行: {cmd}")
        started = time.time()
        try:
            with open("outputs/bert_ckpt_parameters.txt", encoding='utf-8',mode='a' )as f:
                f.write(cmd+'\n')
            result = subprocess.run(cmd, shell=True, cwd=work_dir)
            elapsed = time.time() - started
            if result.returncode == 0:
                print(f"[{idx}/{total}] 运行成功，耗时 {elapsed:.1f} 秒")
            else:
                print(f"[{idx}/{total}] 运行失败，返回码 {result.returncode}，耗时 {elapsed:.1f} 秒")
        except Exception as exc:
            print(f"[{idx}/{total}] 执行异常: {exc}")


if __name__ == "__main__":
    main()
