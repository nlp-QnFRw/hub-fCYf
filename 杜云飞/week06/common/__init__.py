"""公共模块包：labels / paths / seed / text / data / metrics / contract / probe。

各方法目录（rule / tfidf / bert / llm_*）只允许 import 这里的模块，**禁止互相 import**，
新增一个方法 = 新增一个目录 + 产出符合契约的 pred_*.jsonl，不改动已有方法。
"""
