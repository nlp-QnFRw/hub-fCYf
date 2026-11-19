"""
标签口径（THUCNews 10 类 → 投研 4 组）

与原项目（TNEWS 15 类）的差别：
  - 数据集换成 THUCNews，类别从 15 个新闻栏目收到 10 个，且是**正文**而不是标题；
  - 但「先定义口径、再谈模型」这件事没变：10 个栏目对投研依然不是主题，
    所以照样映射成 金融 / 产业 / 宏观 / 噪音 四组，相关性由组派生，零人工标注。

这张表是**业务定义**，写在代码里而不是模型里：换领域只改这里。
"""

# THUCNews 标准 10 类（cnews.train.txt 的第一列）
CLASSES = ["体育", "娱乐", "家居", "房产", "教育", "时尚", "时政", "游戏", "科技", "财经"]
CLASS2ID = {name: i for i, name in enumerate(CLASSES)}

# 10 个栏目 → 4 个业务桶
GROUP_MAP = {
    "财经": "金融", "房产": "金融",
    "科技": "产业", "家居": "产业",
    "时政": "宏观",
    "体育": "噪音", "娱乐": "噪音", "教育": "噪音", "时尚": "噪音", "游戏": "噪音",
}
GROUPS = ["金融", "产业", "宏观", "噪音"]

# 相关性：金融 + 产业 才是投研要看的；宏观弱相关，噪音直接丢
RELEVANT_GROUPS = {"金融", "产业"}


def to_group(label_id: int) -> str:
    """10 类 id → 4 组主题"""
    return GROUP_MAP[CLASSES[label_id]]


def is_relevant(label_id: int) -> bool:
    """10 类 id → 是否与投研相关（由主题组派生）"""
    return to_group(label_id) in RELEVANT_GROUPS


def queue_name(label_id: int, conf: float, conf_threshold: float = 0.8) -> str:
    """
    分类结果落到哪个队列——「分类」变成「动作」的地方。
      不相关 → drop；相关但置信不足 → review；相关且高置信 → research
    """
    if not is_relevant(label_id):
        return "drop"
    if conf < conf_threshold:
        return "review"
    return "research"
