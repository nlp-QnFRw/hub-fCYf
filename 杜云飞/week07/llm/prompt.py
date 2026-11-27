# -*- coding: utf-8 -*-
"""
FewFC 事件抽取 Prompt 构造 + 模型输出解析（zero-shot / few-shot / API / 本地模型共用）

在课程 llm_zero_shot/event_prompt.py 基础上改两处：
  1. **多事件**：FewFC 30.7% 的句子含多个事件（最多 23 个），prompt 明确要求
     "抽取全部事件（0 个、1 个或多个）"，输出 events 为数组 —— 这是与课程
     "一句一个事件"口径最大的不同，也是生成式方法在本数据集上的结构性优势；
  2. schema 直接用 build_dataset 产出的 {type: [roles]} 扁平字典（课程版包了一层 event_types）。

其余工程原则原样保留：只输出文本 span 不输出 offset（程序 text.find 对齐）、
原文没有的角色省略、三层 JSON 容错修复、示例全部取自 train（绝不用 dev，防泄漏）。
"""
import json
import re

SYSTEM_PROMPT_TEMPLATE = """你是金融文本事件抽取专家。输入是**一句**金融领域的句子，请抽取其中发生的**全部事件**（可能没有、一个或多个，每个事件独立成一条）。

## 事件类型与论元角色（共 {n_types} 类）
{type_lines}

## 抽取要求
1. 判断事件类型，必须是上面 {n_types} 类之一，逐字照抄类型名。一句有多个事件时，每个事件单独一条记录，即使它们的触发词相同。
2. 触发词：原文中标志该事件发生的词，必须是从原文**逐字截取**的连续片段，不要改写。
3. 论元：抽取该类型下出现的角色及其原文片段，同样逐字来自原文：
   - 原文里没有出现的角色，直接省略，不要猜、不要编；
   - 同一角色在同一事件中有多个取值时（如多个受让方），用多条记录分别表示。
4. **不要输出字符下标（start/end）**：程序会用 text.find() 在原文里重新对齐。
5. 数字、日期、比例要与原文写法完全一致（如"1.35亿""2019年10月15日""不超过2%"）。

## 输出格式（严格，只输出一个 JSON 对象，不要解释、不要加 markdown 代码块）
{{"events": [{{"type": "事件类型", "trigger": {{"text": "触发词"}}, "arguments": [{{"role": "角色名", "argument": "论元文本"}}]}}]}}
一个事件都抽不到时输出：{{"events": []}}"""


def schema_type_lines(schema: dict) -> str:
    """schema 渲染成一行一个类型（分隔符用「、」——分隔符不能出现在字段值里，课程坑 #26）。"""
    return "\n".join(f"- {etype}：" + "、".join(roles) for etype, roles in schema.items())


def build_system_prompt(schema: dict) -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(n_types=len(schema),
                                         type_lines=schema_type_lines(schema))


# ── Few-shot 示例（全部逐字取自 train_extract.jsonl；覆盖 9/10 类，含多事件句与易混类型）──

FEW_SHOT_EXAMPLES = [
    {   # 质押 + 同句多事件（两个质押事件，质权人不同）
        "text": "乐融致新上述两笔同等数额的股权此前分别质押给了民生信托和中航信托,正好与本次司法拍卖相对应。",
        "target": '{"events": [{"type": "质押", "trigger": {"text": "质押"}, "arguments": [{"role": "sub-org", "argument": "乐融致新"}, {"role": "obj-org", "argument": "民生信托"}, {"role": "collateral", "argument": "股权"}]}, {"type": "质押", "trigger": {"text": "质押"}, "arguments": [{"role": "sub-org", "argument": "乐融致新"}, {"role": "obj-org", "argument": "中航信托"}, {"role": "collateral", "argument": "股权"}]}]}',
    },
    {   # 减持 + 同角色多取值（5 个减持对象 = 5 个事件）
        "text": "颇为值得一提的是,冯柳在三季度还减持了5只股票,包括金域医学、广日股份、康缘药业、海利得、羚锐制药。",
        "target": '{"events": [{"type": "减持", "trigger": {"text": "减持"}, "arguments": [{"role": "sub", "argument": "冯柳"}, {"role": "obj", "argument": "金域医学"}, {"role": "date", "argument": "三季度"}]}, {"type": "减持", "trigger": {"text": "减持"}, "arguments": [{"role": "sub", "argument": "冯柳"}, {"role": "obj", "argument": "广日股份"}, {"role": "date", "argument": "三季度"}]}, {"type": "减持", "trigger": {"text": "减持"}, "arguments": [{"role": "sub", "argument": "冯柳"}, {"role": "obj", "argument": "康缘药业"}, {"role": "date", "argument": "三季度"}]}, {"type": "减持", "trigger": {"text": "减持"}, "arguments": [{"role": "sub", "argument": "冯柳"}, {"role": "obj", "argument": "海利得"}, {"role": "date", "argument": "三季度"}]}, {"type": "减持", "trigger": {"text": "减持"}, "arguments": [{"role": "sub", "argument": "冯柳"}, {"role": "obj", "argument": "羚锐制药"}, {"role": "date", "argument": "三季度"}]}]}',
    },
    {   # 投资 与 股份股权转让 同句共存（同一句属于多个类型）
        "text": "2017年7月,智慧能源与远东控股集团分别以现金收购肖鹏持有的北京京航安机场工程有限公司51%、49%股权。",
        "target": '{"events": [{"type": "投资", "trigger": {"text": "收购"}, "arguments": [{"role": "sub", "argument": "智慧能源"}, {"role": "obj", "argument": "北京京航安机场工程有限公司"}, {"role": "date", "argument": "2017年7月"}]}, {"type": "投资", "trigger": {"text": "收购"}, "arguments": [{"role": "sub", "argument": "远东控股集团"}, {"role": "obj", "argument": "北京京航安机场工程有限公司"}, {"role": "date", "argument": "2017年7月"}]}, {"type": "股份股权转让", "trigger": {"text": "收购"}, "arguments": [{"role": "sub-per", "argument": "肖鹏"}, {"role": "obj-org", "argument": "智慧能源"}, {"role": "collateral", "argument": "股权"}, {"role": "date", "argument": "2017年7月"}, {"role": "proportion", "argument": "51%"}, {"role": "target-company", "argument": "北京京航安机场工程有限公司"}]}]}',
    },
    {   # 起诉 + 多事件（两个被告 = 两个事件）
        "text": "无奈之下,2013年9月,浙江博元把金华嘉华和青州佳华起诉到山东省潍坊市中级人民法院,要求对方支付剩余工程款。",
        "target": '{"events": [{"type": "起诉", "trigger": {"text": "起诉"}, "arguments": [{"role": "sub-org", "argument": "浙江博元"}, {"role": "obj-org", "argument": "金华嘉华"}, {"role": "date", "argument": "2013年9月"}]}, {"type": "起诉", "trigger": {"text": "起诉"}, "arguments": [{"role": "sub-org", "argument": "浙江博元"}, {"role": "obj-org", "argument": "青州佳华"}, {"role": "date", "argument": "2013年9月"}]}]}',
    },
    {   # 收购（金额、方式、比例、对象齐全）
        "text": "1月舜宇光学以1707.1万元现金收购了无锡为森额外13.96%的股权,持股比例提升至60.07%,实现并表。",
        "target": '{"events": [{"type": "收购", "trigger": {"text": "收购"}, "arguments": [{"role": "date", "argument": "1月"}, {"role": "sub-org", "argument": "舜宇光学"}, {"role": "money", "argument": "1707.1万元"}, {"role": "way", "argument": "现金"}, {"role": "obj-org", "argument": "无锡为森"}, {"role": "proportion", "argument": "13.96%"}]}]}',
    },
    {   # 判决
        "text": "9月23日公告显示,苏州中院一审判决全资子公司康得菲尔归还农行张家港分行借款本金9999万元及相关利息。",
        "target": '{"events": [{"type": "判决", "trigger": {"text": "判决"}, "arguments": [{"role": "date", "argument": "9月23日"}, {"role": "institution", "argument": "苏州中院"}, {"role": "obj-org", "argument": "康得菲尔"}, {"role": "sub-org", "argument": "农行张家港分行"}, {"role": "money", "argument": "9999万元"}]}]}',
    },
    {   # 中标
        "text": "城发环境(000885)7月9日晚间公告,公司中标伊川县生活垃圾焚烧发电招商项目,估算投资额为5.7亿元。",
        "target": '{"events": [{"type": "中标", "trigger": {"text": "中标"}, "arguments": [{"role": "sub", "argument": "城发环境"}, {"role": "date", "argument": "7月9日"}, {"role": "obj", "argument": "伊川县"}, {"role": "amount", "argument": "5.7亿元"}]}]}',
    },
    {   # 签署合同 + 同角色多取值
        "text": "此次,沈阳城建集团与中国银行、建设银行、光大银行、盛京银行等四家银行签约,进一步强化战略合作,共获得260亿元授信。",
        "target": '{"events": [{"type": "签署合同", "trigger": {"text": "签约"}, "arguments": [{"role": "sub-org", "argument": "沈阳城建集团"}, {"role": "obj-org", "argument": "中国银行"}, {"role": "obj-org", "argument": "建设银行"}, {"role": "obj-org", "argument": "光大银行"}, {"role": "obj-org", "argument": "盛京银行"}, {"role": "amount", "argument": "260亿元"}]}]}',
    },
    {   # 担保
        "text": "2018年4月20日,实控人林永飞以公司名义,为其向自然人周子聪借入的1.5亿元债务提供连带责任担保。",
        "target": '{"events": [{"type": "担保", "trigger": {"text": "担保"}, "arguments": [{"role": "date", "argument": "2018年4月20日"}, {"role": "sub-per", "argument": "林永飞"}, {"role": "amount", "argument": "1.5亿元"}, {"role": "way", "argument": "连带责任担保"}]}]}',
    },
]


def build_messages(text: str, schema: dict, few_shot: bool = False) -> list[dict]:
    """组装 OpenAI 兼容 messages。few-shot 示例放 user/assistant 对，user 消息保持干净。"""
    messages = [{"role": "system", "content": build_system_prompt(schema)}]
    if few_shot:
        for ex in FEW_SHOT_EXAMPLES:
            messages.append({"role": "user", "content": ex["text"]})
            messages.append({"role": "assistant", "content": ex["target"]})
    messages.append({"role": "user", "content": f"句子：{text}"})
    return messages


# ═══════════════ 解析：模型输出 → 结构化事件（三层容错，课程同逻辑）════════════════

def _strip_code_fence(raw: str) -> str:
    s = raw.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9_+-]*\s*", "", s)
        s = re.sub(r"\s*```\s*$", "", s)
    return s.strip()


def _first_json_object(s: str) -> str | None:
    """括号配平扫描取第一个完整 JSON 对象（正则贪婪/非贪婪都会挂，必须配平扫）。"""
    start = s.find("{")
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start:i + 1]
    return None


def _close_truncated(s: str) -> str | None:
    """max_tokens 截断兜底：补引号 + 补右括号，宁可丢最后一个论元。"""
    start = s.find("{")
    if start < 0:
        return None
    stack = []
    in_str = esc = False
    for ch in s[start:]:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if stack:
                stack.pop()
    if not stack and not in_str:
        return None
    fixed = s[start:]
    if in_str:
        fixed += '"'
    fixed = re.sub(r',\s*"[^"]*"\s*:\s*$', "", fixed)
    fixed = re.sub(r',\s*$', "", fixed)
    for ch in reversed(stack):
        fixed += "}" if ch == "{" else "]"
    return fixed


def _light_repair(s: str) -> str:
    """只去尾随逗号 —— 改引号/标点可能改坏论元文本本身，宁可判失败。"""
    return re.sub(r",\s*([}\]])", r"\1", s)


def _repair_brackets(s: str) -> str | None:
    """栈式括号修复：小模型最常见的畸形是少右括号，必须在缺失处插入（去尾逗号救不回）。"""
    start = s.find("{")
    if start < 0:
        return None
    out, stack = [], []
    in_str = esc = False
    for ch in s[start:]:
        if in_str:
            out.append(ch)
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            out.append(ch)
        elif ch in "{[":
            stack.append(ch)
            out.append(ch)
        elif ch in "}]":
            want = "{" if ch == "}" else "["
            if stack and stack[-1] == want:
                stack.pop()
                out.append(ch)
            elif stack:
                while stack and stack[-1] != want:
                    out.append("}" if stack.pop() == "{" else "]")
                if stack:
                    stack.pop()
                out.append(ch)
        else:
            out.append(ch)
    if in_str:
        out.append('"')
    fixed = re.sub(r",\s*$", "", "".join(out))
    while stack:
        fixed += "}" if stack.pop() == "{" else "]"
    return fixed


def _try_loads(candidate: str):
    for text, repaired in ((candidate, False), (_light_repair(candidate), True)):
        try:
            return json.loads(text), repaired
        except json.JSONDecodeError:
            continue
    fixed = _repair_brackets(candidate)
    if fixed:
        try:
            return json.loads(_light_repair(fixed)), True
        except json.JSONDecodeError:
            pass
    closed = _close_truncated(candidate)
    if closed:
        try:
            return json.loads(_light_repair(closed)), True
        except json.JSONDecodeError:
            return None, False
    return None, False


def align(text: str, span: str) -> tuple[int, int]:
    """原文检索对齐（取首次出现）；找不到 = 幻觉，记 (-1,-1) 不强行纠正。"""
    idx = text.find(span)
    return (idx, idx + len(span)) if idx >= 0 else (-1, -1)


def _to_event_list(obj, schema: dict, source_text: str, info: dict) -> list[dict]:
    """规整成统一事件列表；兼容 events/event/裸对象、trigger/触发词、arguments/args 等写法。"""
    if isinstance(obj, dict) and isinstance(obj.get("events"), list):
        raw_events = obj["events"]
    elif isinstance(obj, dict) and isinstance(obj.get("event"), dict):
        raw_events = [obj["event"]]
    elif isinstance(obj, dict) and ("type" in obj or "事件类型" in obj):
        raw_events = [obj]
    elif isinstance(obj, list):
        raw_events = obj
    else:
        raw_events = []

    events = []
    for ev in raw_events:
        if not isinstance(ev, dict):
            continue
        etype = str(ev.get("type", ev.get("event_type", ev.get("事件类型", "")))).strip()
        if not etype:
            info["type_invalid"] += 1
            continue
        if etype not in schema:
            info["type_invalid"] += 1  # 不替模型纠正，原样留下（评测算 FP，这才诚实）

        trig = ev.get("trigger", ev.get("触发词", ""))
        trig_text = str(trig.get("text", trig.get("word", "")) if isinstance(trig, dict)
                        else trig or "").strip()
        if not trig_text:
            continue
        t_start, t_end = align(source_text, trig_text)
        if t_start < 0:
            info["trigger_not_in_source"] += 1

        raw_args = ev.get("arguments")
        if raw_args is None:
            raw_args = ev.get("args")
        if raw_args is None:
            raw_args = ev.get("论元") or []
        if isinstance(raw_args, dict):
            raw_args = [{"role": k, "argument": v} for k, v in raw_args.items()]

        args, seen = [], set()
        for a in raw_args:
            if isinstance(a, str) or not isinstance(a, dict):
                info["role_invalid"] += 1
                continue
            role = str(a.get("role", a.get("type", "")) or "").strip()
            text = str(a.get("argument", a.get("text", a.get("论元", ""))) or "").strip()
            if not role or not text:
                info["role_invalid"] += 1
                continue
            if etype in schema and role not in schema[etype]:
                info["role_invalid"] += 1
            if (role, text) in seen:
                continue
            seen.add((role, text))
            if source_text.find(text) < 0:
                info["arg_not_in_source"] += 1
            a_start, a_end = align(source_text, text)
            args.append({"role": role, "text": text, "start": a_start, "end": a_end})

        events.append({"type": etype,
                       "trigger": {"text": trig_text, "start": t_start, "end": t_end},
                       "args": args})
    return events


def parse_model_output(raw: str, source_text: str, schema: dict) -> tuple[list[dict], dict]:
    """模型原始输出 → (事件列表, 解析统计)。工程指标（JSON 合法率等）全部来自这里。"""
    info = {"empty_reply": False, "json_ok": False, "repaired": False, "noise_stripped": False,
            "type_invalid": 0, "role_invalid": 0,
            "trigger_not_in_source": 0, "arg_not_in_source": 0}
    if not raw or not raw.strip():
        info["empty_reply"] = True
        return [], info
    cleaned = _strip_code_fence(raw)
    candidate = _first_json_object(cleaned)
    info["noise_stripped"] = bool(candidate and candidate != raw.strip())
    obj, repaired = (None, False)
    if candidate:
        obj, repaired = _try_loads(candidate)
    if obj is None:
        obj, repaired = _try_loads(cleaned)
    if obj is None:
        return [], info
    info["json_ok"] = True
    info["repaired"] = bool(repaired)
    return _to_event_list(obj, schema, source_text, info), info
