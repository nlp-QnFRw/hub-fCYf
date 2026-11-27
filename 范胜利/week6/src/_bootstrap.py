# -*- coding: utf-8 -*-
"""
_bootstrap.py —— 所有入口脚本的第一行 import，只做两件事

【为什么需要这个文件】
    在这台 Intel Mac 上，只要 **numpy 比 torch 先 import**，训练跑到第一个 batch
    就会 segfault（lldb 抓到的栈是 libomp.dylib 的 __kmp_suspend_initialize_thread）。

    原因：这份环境里有两个 OpenMP 运行时
      · numpy 走 conda 的 MKL → 带 libiomp5.dylib
      · torch 2.2 的 macOS wheel → 自带 libomp.dylib
    先 import numpy 会把 MKL 的 libiomp5 先装载并初始化线程池；torch 之后用自己的
    libomp 再开线程时，两套运行时抢同一个线程状态，直接崩在 native 层——
    Python 层连异常都看不到（表现为 exit code 139，日志停在某个 step 之前）。

    修复不是加环境变量（KMP_DUPLICATE_LIB_OK / MKL_THREADING_LAYER 都试过，无效），
    而是保证**只有一个运行时被初始化**：让 torch 先 import。

【为什么放在单独文件】
    train.py / compare.py / 以后加的任何入口脚本都可能踩同一个坑。
    与其在每处都写「注意 import 顺序」的注释，不如把顺序固化成一个 import。
    这也是「隐性问题显性化」：后来的人只要看到 `import _bootstrap` 就知道有讲究。
"""

import os

# tokenizers 内部用 rayon 线程池，和 DataLoader 混用会刷警告；关掉不影响结果
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch  # noqa: E402,F401  ← 关键：必须早于 numpy / sklearn / pandas
