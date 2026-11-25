"""简历信息抽取数据生成器（模板 + 槽位 + 版式变体 + 难例）。

为什么要自己造数据：
    简历含隐私，不能拿真实简历做作业；而公开 NER 数据没有「学历/专业/薪资」这类字段。
    所以用固定池子合成一套**可控、可复现、无隐私**的数据，专门测「小样本 + 模板化」这一档。

三条硬约束（prepare_data.py 会再验一遍）：
    1. 所有实体 offset 由拼接过程产生，保证 text[start:end] == 实体原文；
    2. 人名/学校/公司/邮箱前缀等槽位池按 split 切分，train/dev/test 不共享字符串；
    3. 干扰项（性别/年龄/籍贯/邮编/QQ/绩点/CET/项目人数/项目金额）一律不标。

用法：
    python -m src.gen_resume --out data/processed/resume --train 500 --dev 60 --test 80
"""
from __future__ import annotations

import argparse
import random
from dataclasses import dataclass, field, replace
from pathlib import Path

from .common import DATA_PROCESSED, set_seed, write_json, write_jsonl

LABELS = ["NAME", "TEL", "EMAIL", "EDU", "SCHOOL", "MAJOR", "ORG", "POS", "TIME", "SALARY"]

SURNAMES = "王李张刘陈杨黄赵吴周徐孙马朱胡郭何高林罗郑梁谢宋唐许韩冯邓曹彭曾肖田董袁潘蒋蔡余杜叶程苏魏吕丁任沈姚卢姜崔钟谭陆汪范金石廖贾夏韦付方白邹孟熊秦邱江尹薛闫段雷侯龙史陶黎贺顾毛郝龚邵万钱严覃武戴莫孔向汤"
GIVEN = ["伟", "芳", "娜", "秀英", "敏", "静", "丽", "强", "磊", "军", "洋", "勇", "艳", "杰", "娟",
         "涛", "明", "超", "秀兰", "霞", "平", "刚", "桂英", "文", "辉", "建华", "雪", "婷", "浩",
         "雨墨", "子轩", "梓涵", "宇航", "晨曦", "若曦", "俊杰", "天佑", "思远", "嘉怡", "欣怡",
         "博文", "皓轩", "雨欣", "一诺", "梓萱", "明轩", "诗涵", "昊天", "雅静", "志强"]

SCHOOLS = ["河北科技大学", "江南大学", "南京工程学院", "杭州电子科技大学", "武汉理工大学", "西安邮电大学",
           "郑州轻工业大学", "合肥工业大学", "长沙理工大学", "山东建筑大学", "成都信息工程大学",
           "南昌航空大学", "太原理工大学", "东北林业大学", "江苏大学", "浙江工业大学", "安徽大学",
           "福州大学", "昆明理工大学", "桂林电子科技大学", "兰州交通大学", "青岛科技大学",
           "天津工业大学", "沈阳航空航天大学", "河南工业大学", "湖北工业大学", "重庆邮电大学",
           "西南石油大学", "华北电力大学", "上海海事大学", "广东工业大学", "陕西科技大学",
           "北京信息科技大学", "南京邮电大学", "哈尔滨理工大学", "长春理工大学", "济南大学",
           "烟台大学", "常州大学", "嘉兴学院", "绍兴文理学院", "宁波工程学院", "台州学院",
           "徐州工程学院", "盐城工学院", "淮阴工学院", "皖西学院", "黄山学院", "莆田学院", "宜春学院"]
# 再用「城市 + 办学层次」兜一批，保证 test 里学校字符串足够分散
SCHOOLS += [f"{c}{k}" for c in ["杭州", "南京", "合肥", "武汉", "西安", "成都", "长沙", "郑州", "济南", "福州", "南昌", "太原"]
            for k in ["大学", "学院", "理工学院"]]

MAJORS = ["计算机科学与技术", "软件工程", "数据科学与大数据技术", "人工智能", "网络工程", "信息安全",
          "电子信息工程", "通信工程", "自动化", "数学与应用数学", "统计学", "信息管理与信息系统",
          "物联网工程", "数字媒体技术", "电子商务", "金融工程", "会计学", "工商管理", "市场营销",
          "物流管理", "工业设计", "机械设计制造及其自动化", "电气工程及其自动化", "应用物理学",
          "生物信息学", "地理信息科学", "测绘工程", "化学工程与工艺", "材料科学与工程", "环境工程",
          "智能科学与技术", "机器人工程", "数据计算及应用", "数字经济", "供应链管理", "审计学",
          "翻译", "汉语言文学", "新闻学", "广告学", "应用心理学", "生物技术", "食品科学与工程", "车辆工程"]

COMPANIES = ["杭州云启科技有限公司", "上海数联信息技术有限公司", "北京知微网络科技有限公司",
             "深圳前海智算科技有限公司", "南京云杉数据服务有限公司", "广州瀚海软件有限公司",
             "成都天府数智科技有限公司", "武汉光谷智能科技有限公司", "苏州工业园云计算有限公司",
             "西安丝路信息科技有限公司", "合肥智谷软件有限公司", "长沙湘江网络科技有限公司",
             "厦门鹭岛数据科技有限公司", "青岛海纳信息技术有限公司", "天津滨海智能装备有限公司",
             "重庆两江数字科技有限公司", "郑州中原软件有限公司", "济南泉城网络科技有限公司",
             "宁波甬江数据服务有限公司", "无锡太湖云计算有限公司", "佛山智造科技有限公司",
             "东莞松山湖软件有限公司", "珠海横琴金融科技有限公司", "福州榕城信息科技有限公司",
             "南昌赣江数据有限公司", "贵阳大数据交易服务有限公司", "兰州丝路云科技有限公司",
             "太原晋阳软件有限公司", "石家庄冀科信息技术有限公司", "沈阳浑南智能科技有限公司"]
# 批量兜底：城市 + 字号 + 行业，保证公司名在 test 中足够分散
COMPANIES += [f"{c}{x}{k}有限公司"
              for c in ["杭州", "上海", "北京", "深圳", "南京", "广州", "成都", "武汉", "苏州", "西安"]
              for x in ["启明", "恒信", "联创", "博远", "天工", "智联", "云图", "新元"]
              for k in ["科技", "信息", "数据"]]

POSITIONS = ["后端开发工程师", "前端开发工程师", "算法工程师", "数据分析师", "测试工程师",
             "运维工程师", "产品经理", "项目经理", "Java开发工程师", "Python开发工程师",
             "大数据开发工程师", "机器学习工程师", "自然语言处理工程师", "全栈工程师",
             "数据库管理员", "网络安全工程师", "嵌入式软件工程师", "移动端开发工程师",
             "数据仓库工程师", "BI工程师", "解决方案架构师", "技术支持工程师", "实施顾问",
             "运营专员", "市场推广经理"]
POSITIONS += [f"{x}工程师" for x in ["Java", "Python", "Go", "C++", "大数据", "算法", "前端", "后端",
                                    "测试", "运维", "安全", "嵌入式", "移动端", "数据", "推荐", "搜索"]]
POSITIONS += ["技术总监", "研发经理", "架构师", "数据分析专家", "产品专员", "项目助理",
              "客户成功经理", "售前工程师", "交付经理", "质量经理"]

EMAIL_PREFIX = ["zhangwei", "luyumo", "wangfang", "liqiang", "chenjing", "yanghao", "zhaolei",
                "huangmin", "zhoujie", "wuxia", "xuyang", "sunli", "majun", "zhutao", "huyan",
                "guoxin", "helin", "gaofei", "luobin", "zhengkai", "liangyu", "xiepeng", "songna",
                "tangwei", "hanxue", "fengyu", "dengchao", "caoyang", "pengli", "zengming",
                "liuyang", "wangyu", "chenxi", "zhaomin", "sunhao", "zhouli", "wuting", "zhengbo",
                "fenglin", "xuming", "hubin", "guoyu", "luqian", "mayi", "xuefeng", "yujia",
                "hanbing", "jiangnan"]
EMAIL_DOMAIN = ["163.com", "qq.com", "126.com", "gmail.com", "outlook.com", "sina.com", "foxmail.com"]

EDUS = ["大专", "本科", "硕士", "博士", "研究生"]
CITIES = ["杭州", "上海", "北京", "深圳", "南京", "广州", "成都", "武汉", "苏州", "西安", "合肥", "长沙"]
PROVINCES = ["浙江", "江苏", "山东", "河南", "湖北", "湖南", "四川", "安徽", "福建", "江西"]


@dataclass
class Canvas:
    """边拼文本边记 offset，杜绝手算下标出错。"""

    parts: list[str] = field(default_factory=list)
    entities: list[dict] = field(default_factory=list)
    _pos: int = 0

    def add(self, s: str, label: str | None = None) -> None:
        # @doc ** offset 由拼接累加，事后 find 会定位到第一次出现
        # 「后端开发工程师」在一条简历里出现 3 次，find 永远定位到第一次
        if label:
            self.entities.append({"start": self._pos, "end": self._pos + len(s), "label": label, "text": s})
        self.parts.append(s)
        self._pos += len(s)

    def render(self) -> tuple[str, list[dict]]:
        text = "".join(self.parts)
        for e in self.entities:
            assert text[e["start"]:e["end"]] == e["text"], f"offset 错位: {e}"
        return text, self.entities


def split_pool(pool: list, seed: int, frac=(0.68, 0.12, 0.20)) -> dict[str, list]:
    """把槽位池按比例切成 train/dev/test，三份互不重叠——这是防泄漏的关键。"""
    p = pool[:]
    random.Random(seed).shuffle(p)
    n = len(p)
    a, b = int(n * frac[0]), int(n * (frac[0] + frac[1]))
    return {"train": p[:a], "dev": p[a:b], "test": p[b:]}


def make_pools(seed: int = 42) -> dict[str, dict[str, list]]:
    names = [s + g for s in SURNAMES for g in GIVEN]
    pools = {
        "name": split_pool(names, seed + 1),
        "school": split_pool(SCHOOLS, seed + 2),
        "major": split_pool(MAJORS, seed + 3),
        "company": split_pool(COMPANIES, seed + 4),
        "position": split_pool(POSITIONS, seed + 5),
        "email": split_pool(EMAIL_PREFIX, seed + 6),
        "city": split_pool(CITIES, seed + 7),
        "province": split_pool(PROVINCES, seed + 8),
    }
    # 只有「会变成实体」的池子才需要切得足够散；城市/省份只当背景文字，不参与断言
    for k in ("name", "school", "major", "company", "position", "email"):
        v = pools[k]
        assert all(len(x) >= 5 for x in v.values()), (k, {kk: len(vv) for kk, vv in v.items()})
    return pools


@dataclass
class Profile:
    name: str
    gender: str
    age: int
    tel: str
    email: str
    city: str
    province: str
    zipcode: str
    qq: str
    edu: str
    school: str
    major: str
    gpa: str
    edu_start: str
    edu_end: str
    company: str
    pos: str
    work_start: str
    work_end: str
    salary: str
    cet: int
    team: int
    budget: int


@dataclass
class Opts:
    """版式开关：正常样本一套，难例在此基础上改。"""

    date_style: str = "slash"   # slash 2017/09 | dash 2017.09 | cn 2017年09月
    date_sep: str = "—"         # — | - | 至
    gap: str = "　"             # 全角空格
    extra_space: bool = False   # 额外插空格（难例）


def _date(year: int, month: int, style: str) -> str:
    if style == "slash":
        return f"{year}/{month:02d}"
    if style == "dash":
        return f"{year}.{month:02d}"
    return f"{year}年{month}月"


def _range(a: str, b: str, sep: str) -> str:
    return f"{a}{sep}{b}"


def make_profile(rng: random.Random, split: str, pools: dict) -> Profile:
    pick = lambda k: rng.choice(pools[k][split])  # noqa: E731
    gender = rng.choice(["男", "女"])
    name = pick("name")
    age = rng.randint(22, 45)
    tel = "1" + rng.choice("3578") + "".join(rng.choice("0123456789") for _ in range(9))
    email = f"{pick('email')}{rng.randint(1, 99)}@{rng.choice(EMAIL_DOMAIN)}"
    grad_year = rng.randint(2010, 2023)
    m1, m2 = rng.randint(6, 9), rng.randint(6, 7)
    edu_start = f"{grad_year - 4}/{m1:02d}"
    edu_end = f"{grad_year}/{m2:02d}"
    wm = m2 + 1 if m2 < 12 else 1
    work_start = f"{grad_year}/{wm:02d}"
    work_end = "至今" if rng.random() < 0.45 else f"{rng.randint(grad_year + 1, 2026)}/{rng.randint(1, 12):02d}"
    salary_style = rng.choice(["K", "K月", "Wan", "Neg"])
    if salary_style == "K":
        lo = rng.choice([8, 10, 12, 15, 18, 20, 25, 30])
        salary = f"{lo}K-{lo + rng.choice([5, 8, 10, 15])}K"
    elif salary_style == "K月":
        lo = rng.choice([8, 10, 12, 15, 20])
        salary = f"{lo}-{lo + rng.choice([5, 8, 10])}K/月"
    elif salary_style == "Wan":
        lo = rng.choice([20, 25, 30, 35, 40])
        salary = f"{lo}万-{lo + rng.choice([10, 15, 20])}万/年"
    else:
        salary = "面议"
    return Profile(
        name=name, gender=gender, age=age, tel=tel, email=email, city=pick("city"),
        province=pick("province"), zipcode=f"{rng.randint(100000, 999999)}",
        qq=str(rng.randint(10000000, 999999999)), edu=rng.choice(EDUS),
        school=pick("school"), major=pick("major"),
        gpa=f"{rng.uniform(2.8, 4.0):.1f}/{rng.choice([4.0, 5.0])}",
        edu_start=edu_start, edu_end=edu_end,
        company=pick("company"), pos=pick("position"),
        work_start=work_start, work_end=work_end, salary=salary,
        cet=rng.choice([425, 480, 512, 535, 560, 590]),
        team=rng.randint(3, 30), budget=rng.randint(50, 800),
    )


def _fullwidth(s: str) -> str:
    return "".join(chr(ord(c) - ord("0") + ord("０")) if c.isdigit() else c for c in s)


def _t(p: Profile, o: Opts) -> str:
    return _fullwidth(p.tel) if p.tel.isdigit() and getattr(o, "fullwidth_tel", False) == p.tel else p.tel


def render_colon(c: Canvas, p: Profile, o: Opts) -> None:
    g = "  " if o.extra_space else o.gap
    c.add("【基本信息】\n")
    c.add("姓名："); c.add(p.name, "NAME"); c.add("\n")
    c.add(f"性别：{p.gender}{g}年龄：{p.age}\n")
    c.add("电话："); c.add(p.tel, "TEL"); c.add("\n")
    c.add("邮箱："); c.add(p.email, "EMAIL"); c.add("\n")
    c.add(f"现居城市：{p.city}{g}籍贯：{p.province}{g}邮编：{p.zipcode}\n")
    c.add(f"QQ：{p.qq}\n\n")

    c.add("【教育经历】\n")
    c.add(_range(_date(int(p.edu_start[:4]), int(p.edu_start[5:]), o.date_style),
                 _date(int(p.edu_end[:4]), int(p.edu_end[5:]), o.date_style), o.date_sep), "TIME")
    c.add(g); c.add(p.school, "SCHOOL"); c.add(g); c.add(p.major, "MAJOR"); c.add(g)
    c.add(p.edu, "EDU"); c.add("\n")
    c.add(f"绩点：{p.gpa}{g}专业排名：{random.randint(1, 20)}/{random.randint(60, 180)}\n")
    c.add(f"CET-6：{p.cet}\n\n")

    c.add("【工作经历】\n")
    c.add(_range(_date(int(p.work_start[:4]), int(p.work_start[5:]), o.date_style),
                 "至今" if p.work_end == "至今" else _date(int(p.work_end[:4]), int(p.work_end[5:]), o.date_style),
                 o.date_sep), "TIME")
    c.add(g); c.add(p.company, "ORG"); c.add(g); c.add(p.pos, "POS"); c.add("\n")
    c.add(f"工作内容：负责核心模块设计与开发，项目 {p.team} 人，预算 {p.budget} 万元，"
          f"性能提升 {random.randint(2, 9)} 倍。\n\n")

    c.add("【求职意向】\n")
    c.add("期望职位："); c.add(p.pos, "POS"); c.add("\n")
    c.add("期望薪资："); c.add(p.salary, "SALARY"); c.add("\n")
    c.add("到岗时间：一个月内\n\n")

    c.add("【自我评价】\n")
    c.add(f"{p.edu}学历，{random.randint(1, 8)} 年开发经验，熟悉分布式系统；曾就职于")
    c.add(p.company, "ORG"); c.add("，担任"); c.add(p.pos, "POS")
    c.add(f"，沟通顺畅、抗压能力强（{p.city}，{p.age} 岁）。\n")


def render_table(c: Canvas, p: Profile, o: Opts) -> None:
    g = "  " if o.extra_space else " "
    c.add("姓名 | "); c.add(p.name, "NAME"); c.add(f" | 性别 | {p.gender}\n")
    c.add("电话 | "); c.add(p.tel, "TEL"); c.add(" | 邮箱 | "); c.add(p.email, "EMAIL"); c.add("\n")
    c.add(f"年龄 | {p.age} | 籍贯 | {p.province} | 邮编 | {p.zipcode}\n\n")
    c.add("【教育经历】\n")
    c.add(_range(_date(int(p.edu_start[:4]), int(p.edu_start[5:]), o.date_style),
                 _date(int(p.edu_end[:4]), int(p.edu_end[5:]), o.date_style), o.date_sep), "TIME")
    c.add(" | "); c.add(p.school, "SCHOOL"); c.add(" | "); c.add(p.major, "MAJOR"); c.add(" | ")
    c.add(p.edu, "EDU"); c.add("\n")
    c.add(f"绩点 {p.gpa} | CET-6 {p.cet}\n\n")
    c.add("【工作经历】\n")
    c.add(_range(_date(int(p.work_start[:4]), int(p.work_start[5:]), o.date_style),
                 "至今" if p.work_end == "至今" else _date(int(p.work_end[:4]), int(p.work_end[5:]), o.date_style),
                 o.date_sep), "TIME")
    c.add(" | "); c.add(p.company, "ORG"); c.add(" | "); c.add(p.pos, "POS"); c.add("\n")
    c.add(f"项目人数 {p.team} 人 | 项目金额 {p.budget} 万元{g}\n\n")
    c.add("【求职意向】\n")
    c.add("期望职位 | "); c.add(p.pos, "POS"); c.add(" | 期望薪资 | "); c.add(p.salary, "SALARY"); c.add("\n")


def render_sentence(c: Canvas, p: Profile, o: Opts) -> None:
    c.add("我叫"); c.add(p.name, "NAME"); c.add(f"，{p.gender}，{p.age} 岁，现居{p.city}。")
    c.add("联系电话"); c.add(p.tel, "TEL"); c.add("，邮箱"); c.add(p.email, "EMAIL"); c.add("。")
    c.add("我于")
    c.add(_date(int(p.edu_start[:4]), int(p.edu_start[5:]), o.date_style), "TIME")
    c.add("入学，")
    c.add(_date(int(p.edu_end[:4]), int(p.edu_end[5:]), o.date_style), "TIME")
    c.add("毕业于"); c.add(p.school, "SCHOOL"); c.add("，"); c.add(p.major, "MAJOR")
    c.add("专业，"); c.add(p.edu, "EDU")
    c.add(f"学历，绩点 {p.gpa}，CET-6 {p.cet} 分。")
    c.add("毕业至今在"); c.add(p.company, "ORG"); c.add("担任"); c.add(p.pos, "POS")
    c.add("，时间从")
    c.add(_range(_date(int(p.work_start[:4]), int(p.work_start[5:]), o.date_style),
                 "至今" if p.work_end == "至今" else _date(int(p.work_end[:4]), int(p.work_end[5:]), o.date_style),
                 o.date_sep), "TIME")
    c.add("。")
    c.add(f"期间带过 {p.team} 人的小组，项目预算 {p.budget} 万元。")
    c.add("求职意向："); c.add(p.pos, "POS"); c.add("，期望薪资"); c.add(p.salary, "SALARY"); c.add("。")


def make_sample(rng: random.Random, split: str, sid: str, pools: dict, hard: bool) -> dict:
    p = make_profile(rng, split, pools)
    style = rng.choice(["colon", "table", "sentence"]) if hard else rng.choice(["colon", "colon", "table"])
    o = Opts()

    if hard:
        # 难例：换日期格式/连接符、插空格、全角电话、大写邮箱，四类扰动各自独立
        if rng.random() < 0.55:
            o.date_style = rng.choice(["cn", "dash"])
            o.date_sep = "至" if o.date_style == "cn" else "-"
        if rng.random() < 0.4:
            o.extra_space = True
        if rng.random() < 0.5:
            p = replace(p, tel=_fullwidth(p.tel))
        if rng.random() < 0.5:
            p = replace(p, email=p.email.upper())

    c = Canvas()
    if style == "table":
        render_table(c, p, o)
    elif style == "sentence":
        render_sentence(c, p, o)
    else:
        render_colon(c, p, o)

    text, ents = c.render()
    return {
        "id": sid,
        "text": text,
        "entities": [{"start": e["start"], "end": e["end"], "label": e["label"]} for e in ents],
        "meta": {"split": split, "hard": bool(hard), "style": style,
                 "n_chars": len(text), "n_entities": len(ents)},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DATA_PROCESSED / "resume"))
    ap.add_argument("--train", type=int, default=500)
    ap.add_argument("--dev", type=int, default=60)
    ap.add_argument("--test", type=int, default=80)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    set_seed(args.seed)
    pools = make_pools(args.seed)
    out = Path(args.out)
    rng = random.Random(args.seed)

    stats = {}
    for split, n, hard_ratio in (("train", args.train, 0.0), ("dev", args.dev, 0.0), ("test", args.test, 0.4)):
        rows = []
        for i in range(n):
            hard = split == "test" and i < int(n * hard_ratio)
            rows.append(make_sample(rng, split, f"resume-{split}-{i:06d}", pools, hard))
        write_jsonl(out / f"{split}.jsonl", rows)
        stats[split] = {
            "n": len(rows),
            "chars": sum(len(r["text"]) for r in rows),
            "entities": sum(len(r["entities"]) for r in rows),
            "hard": sum(1 for r in rows if r["meta"]["hard"]),
        }
    write_json(out / "labels.json", {"labels": LABELS})
    write_json(out / "stats.json", stats)
    print(f"[gen_resume] -> {out}")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
