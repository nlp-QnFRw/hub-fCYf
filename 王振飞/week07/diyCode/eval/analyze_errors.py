"""分析 BERT 验证集错误。运行：python -m eval.analyze_errors --device cuda。"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from bert.predict import BertPredictor
from common.event_data import DATA, ROOT, read_jsonl, schema_at, segments
from eval.evaluate import score


def argument_set(event):
    """保持与现有评测相同的集合口径：事件类型 + 角色 + 文本。"""
    return {(event['type'], a['role'], a['text']) for a in event.get('arguments', event.get('args', []))}


def main():
    parser = argparse.ArgumentParser(description='保存预测并统计论元错误和切分影响')
    parser.add_argument('--decoder', choices=['linear', 'crf'], default='linear')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--data-dir', type=Path, default=DATA)
    args = parser.parse_args()
    rows = read_jsonl(args.data_dir / 'dev_extract.jsonl')
    model = BertPredictor(args.decoder, args.device)
    schema = schema_at(args.data_dir)
    counts = Counter()
    strata = defaultdict(Counter)
    by_role = defaultdict(Counter)
    cases = defaultdict(list)
    predictions = []
    for row in rows:
        text, gold = row['text'], row['event']
        events, warnings = model.extract(text, schema)
        parts = list(segments(text, max_chars=max(1, model.max_length - 2)))
        group = '多片段' if len(parts) > 1 else '单片段'
        g = argument_set(gold)
        p = set().union(*(argument_set(e) for e in events)) if events else set()
        tp, fp, fn = g & p, p - g, g - p
        for target in (counts, strata[group]):
            target['rows'] += 1
            target['tp'] += len(tp)
            target['fp'] += len(fp)
            target['fn'] += len(fn)
            target['pred_events'] += len(events)
            target['extra_events'] += max(0, len(events)-1)
            target['zero_events'] += not events
        for item in tp: by_role[item[1]]['tp'] += 1
        for item in fp: by_role[item[1]]['fp'] += 1
        for item in fn: by_role[item[1]]['fn'] += 1
        counts['warnings'] += len(warnings)
        # 每个误报只归入一个类别；这只是字符串启发式，不证明语义根因。
        labels = []
        for typ, role, value in sorted(fp):
            if any(gt == typ and gr != role and gv == value for gt, gr, gv in fn):
                category = '同类型同文本角色不同'
            elif any(gt == typ and gr == role and gv != value and (value in gv or gv in value) for gt, gr, gv in fn):
                category = '同类型同角色文本包含（疑似边界）'
            elif any(gt != typ and gr == role and gv == value for gt, gr, gv in fn):
                category = '同角色同文本事件类型不同'
            else:
                category = '其他误报'
            counts['heuristic_' + category] += 1
            labels.append({'role': role, 'text': value, 'category': category})
            if len(cases[category]) < 4:
                cases[category].append({'id': row.get('id'), 'sent_id': row.get('sent_id'), 'text': text, 'gold': gold, 'predicted': events, 'extra': {'type': typ, 'role': role, 'text': value}, 'missing': [dict(type=t, role=r, text=v) for t,r,v in sorted(fn)], 'segments': len(parts)})
        if len(parts) > 1 and len(events) > 1 and len(cases['多片段多事件']) < 4:
            cases['多片段多事件'].append({'id':row.get('id'), 'sent_id':row.get('sent_id'), 'text':text, 'gold':gold, 'predicted':events, 'segments':len(parts)})
        predictions.append({'id':row.get('id'), 'sent_id':row.get('sent_id'), 'text':text, 'gold':gold, 'events':events, 'warnings':warnings, 'segment_count':len(parts), 'false_positive':labels, 'false_negative':[dict(type=t, role=r, text=v) for t,r,v in sorted(fn)]})
    result = {'samples':len(rows), 'decoder':args.decoder, 'device':args.device, 'counts':dict(counts), 'segment_groups':{}, 'roles':{}, 'examples':dict(cases), 'note':'误报类型为基于未匹配集合的启发式分类，不能证明根因；与现有评测一样按集合去重。'}
    for group,c in strata.items(): result['segment_groups'][group] = {**c, 'argument':score(c['tp'],c['fp'],c['fn'])}
    for role,c in by_role.items(): result['roles'][role] = score(c['tp'],c['fp'],c['fn'])
    output = ROOT/'outputs'; output.mkdir(exist_ok=True)
    (output/f'bert_{args.decoder}_error_analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    with (output/f'bert_{args.decoder}_predictions.jsonl').open('w',encoding='utf-8') as f:
        for item in predictions: f.write(json.dumps(item,ensure_ascii=False)+'\n')
    print(json.dumps({'counts':dict(counts),'segment_groups':result['segment_groups'],'examples':result['examples']},ensure_ascii=False,indent=2))

if __name__ == '__main__':
    main()
