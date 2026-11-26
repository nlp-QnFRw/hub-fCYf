"""公共数据层：路径、事件类型、BIO 标签、结果校验和长文本切分。"""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
TEACHER = ROOT.parent / 'code'
DATA = TEACHER / 'data' / 'processed'
BERT = TEACHER / 'pretrain_models' / 'bert-base-chinese'

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def read_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]

def schema_at(data_dir=DATA):
    return {k: v for k, v in read_json(Path(data_dir) / 'event_schema.json').items() if k != '__enums__'}

def labels_for(schema):
    """创建联合标签：O + 13 类触发词和 61 个角色各自的 B/I 标签。"""
    entities = ['TRG:' + t for t in sorted(schema)] + ['ARG:' + r for r in sorted({r for roles in schema.values() for r in roles})]
    return ['O'] + [p + '-' + e for e in entities for p in ('B', 'I')]

def validate(text, events, schema):
    """检查事件类型、角色和原文片段，统一输出左闭右开的字符位置。"""
    out, warnings = ([], [])
    if not isinstance(events, list):
        return ([], ['events 不是数组'])
    for ev in events:
        if not isinstance(ev, dict) or ev.get('type') not in schema:
            warnings.append('存在未知事件类型，已移除')
            continue

        # LLM 不一定给出位置，因此在原文中寻找片段；有重复时提示人工复核。
        def locate(obj):
            if isinstance(obj, str):
                obj = {'text': obj}
            if not isinstance(obj, dict):
                return None
            value = obj.get('text')
            if not isinstance(value, str) or not value:
                return None
            start = obj.get('start')
            if not isinstance(start, int) or text[start:start + len(value)] != value:
                start = text.find(value)
                if start < 0:
                    warnings.append('片段不在原文中：' + value)
                    return None
                if text.find(value, start + 1) >= 0:
                    warnings.append('原文中存在重复片段，位置可能有歧义：' + value)
            return {'text': value, 'start': start, 'end': start + len(value)}
        trigger = locate(ev.get('trigger'))
        if not trigger:
            warnings.append('缺少有效触发词，事件已移除')
            continue
        args, seen = ([], set())
        for a in ev.get('arguments', ev.get('args', [])) or []:
            if not isinstance(a, dict) or a.get('role') not in schema[ev['type']]:
                warnings.append('存在不属于事件类型的角色，已移除')
                continue
            loc = locate(a)
            if loc and (a['role'], loc['start'], loc['end']) not in seen:
                seen.add((a['role'], loc['start'], loc['end']))
                args.append({'role': a['role'], **loc})
        out.append({'type': ev['type'], 'trigger': trigger, 'arguments': args})
    return (out, warnings)

def segments(text, max_chars=220):
    """切分长文本，同时保留每个片段在原文中的起点。

    片段不重叠，尽量在标点处断开；跨片段事件目前可能遗漏。
    """
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        if end < len(text):
            cut = max((text.rfind(c, start + max_chars // 2, end) for c in '。！？；\n'))
            if cut >= start:
                end = cut + 1
        yield (start, text[start:end])
        start = end
