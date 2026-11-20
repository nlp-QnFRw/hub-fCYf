#region 0. 模块说明
"""
标签体系的**唯一真源**。所有方法目录（rule / tfidf / bert / llm_*）都必须
`from common.labels import LABELS, LABEL2ID`，不允许各自再写一份列表。

四层结构（对应项目拆解规范 §3.3）：
  层 1 原始标签：THUCNews 官方的 14 个类别，顺序与官方一致（不要重排！
        重排会让历史 run 的 label id 全部错位）
  层 2 主题映射：14 类 → 6 个业务主题分组，用于把"细分类"聚合成"能汇报的口径"
  层 3 派生指标：由标签派生的属性（是否长尾类、主题归属），不依赖模型
  层 4 动作路由：标签 + 置信度 → 队列（用于错误分析里的"该谁复核"）

设计重点：
  - 换数据集只改这个文件，模型代码一行不动；
  - 长尾类的判定阈值集中在这里（TAIL_RATIO_THRESHOLD），不要在画图脚本里再写一遍。

使用方式：
  from common.labels import LABELS, LABEL2ID, ID2LABEL, to_group, queue_name
  print(len(LABELS), LABELS[:3])
"""
#endregion

#region 1. 导入依赖
from __future__ import annotations

#endregion

#region 2. 层 1：原始标签体系（顺序 = THUCNews 官方目录名顺序）
# 注意：这是数据集的真实标签，**顺序不可改**。改顺序等于改变 label id 语义。
LABELS: list[str] = [
    "财经",  # 0
    "彩票",  # 1
    "房产",  # 2
    "股票",  # 3
    "家居",  # 4
    "教育",  # 5
    "科技",  # 6
    "社会",  # 7
    "时尚",  # 8
    "时政",  # 9
    "体育",  # 10
    "星座",  # 11
    "游戏",  # 12
    "娱乐",  # 13
]
NUM_LABELS: int = len(LABELS)
LABEL2ID: dict[str, int] = {name: i for i, name in enumerate(LABELS)}
ID2LABEL: dict[int, str] = {i: name for i, name in enumerate(LABELS)}
#endregion

#region 3. 层 2：业务主题映射（细分类 → 报告用的 6 个组）
GROUP_MAP: dict[str, str] = {
    "财经": "财经投资",
    "股票": "财经投资",
    "彩票": "财经投资",
    "房产": "房产家居",
    "家居": "房产家居",
    "科技": "科技数码",
    "游戏": "科技数码",
    "教育": "教育",
    "社会": "社会时政",
    "时政": "社会时政",
    "体育": "文体娱乐",
    "娱乐": "文体娱乐",
    "时尚": "文体娱乐",
    "星座": "文体娱乐",
}
GROUPS: list[str] = ["财经投资", "房产家居", "科技数码", "教育", "社会时政", "文体娱乐"]

# 长尾类：样本数不足头部的 1/TAIL_RATIO 时预警（规范 §3.2 的 10× 阈值，本数据集约 23×）
TAIL_RATIO_THRESHOLD: float = 10.0
#endregion

#region 4. 层 3：派生指标（纯函数，无模型依赖）
def to_group(label: str | int) -> str:
    """把细标签（类名或 id）映射到业务主题组。未知标签返回 "未知"。"""
    name = ID2LABEL.get(label, "") if isinstance(label, int) else label
    return GROUP_MAP.get(name, "未知")


def is_tail(name: str, counts: dict[str, int] | None = None) -> bool:
    """判断某个类是否为长尾类。

    参数：
      name   : 类名
      counts : 各类样本数（不给则用数据集的近似静态分布判定）
    返回：
      True 表示该类样本数 < 头部类 / TAIL_RATIO_THRESHOLD
    """
    if counts is None:
        return name in STATIC_TAIL
    if not counts:
        return False
    head = max(counts.values())
    return counts.get(name, 0) < head / TAIL_RATIO_THRESHOLD


# 静态兜底（THUCNews 全量真实分布：科技 162,818 为头部，星座 3,576 为尾部，长尾比 45.5×）
# 判定阈值：头部 / 10 = 16,282 → 低于它的类才算长尾
STATIC_TAIL: set[str] = {"彩票", "时尚", "星座"}
#endregion

#region 5. 层 4：动作路由（标签 + 置信度 → 队列）
def queue_name(label: str | int, conf: float, conf_threshold: float = 0.8,
               tail_threshold: float = 0.6) -> str:
    """把"分类结果"翻译成"业务动作"。

    规则（阈值集中在此，不要在调用方硬编码）：
      conf < conf_threshold         → "review"  人工复核
      长尾类且 conf < tail_threshold → "review"  长尾类模型普遍更没底
      其余                          → "auto"    直接采纳
    """
    name = ID2LABEL.get(label, "") if isinstance(label, int) else str(label)
    if conf < conf_threshold:
        return "review"
    if name in STATIC_TAIL and conf < tail_threshold:
        return "review"
    return "auto"
#endregion

#region 6. 自检
if __name__ == "__main__":
    assert NUM_LABELS == 14, f"THUCNews 应为 14 类，实际 {NUM_LABELS}"
    assert len(set(LABELS)) == NUM_LABELS, "标签有重复"
    assert all(name in GROUP_MAP for name in LABELS), "有标签没进主题映射"
    assert set(GROUPS) == set(GROUP_MAP.values()), "主题组的取值集合不一致"
    print(f"类别数 {NUM_LABELS}：{'、'.join(LABELS)}")
    print(f"主题组 {len(GROUPS)}：{'、'.join(GROUPS)}")
    print("示例路由：", queue_name("时尚", 0.55), queue_name("财经", 0.91))
#endregion
