"""固定已有模型，比较空白符解码前后效果并检查检测集误报。"""
import json
from pathlib import Path
from collections import Counter
from bert.predict import BertPredictor
from common.event_data import DATA, ROOT, read_jsonl, schema_at
from eval.evaluate import score

model = BertPredictor('linear', 'cuda')
schema = schema_at()
old = read_jsonl(ROOT/'outputs/bert_linear_predictions.jsonl')
counts = Counter(); examples={'变好': [], '变差': []}
for row in old:
    gold = row['gold']; predicted, _ = model.extract(row['text'], schema)
    def key(events):
        return {(e['type'], a['role'], a['text']) for e in events for a in e.get('arguments', [])}
    g = {(gold['type'], a['role'], a['text']) for a in gold['args']}
    before, after = key(row['events']), key(predicted)
    for prefix, values in [('before',before),('after',after)]:
        counts[prefix+'_tp'] += len(g & values); counts[prefix+'_fp'] += len(values-g); counts[prefix+'_fn'] += len(g-values)
    before_err = len(before-g)+len(g-before); after_err = len(after-g)+len(g-after)
    if before != after:
        counts['changed_rows'] += 1
        category = '变好' if after_err < before_err else '变差' if after_err > before_err else '变化但错误数相同'
        counts[category] += 1
        if category in examples and len(examples[category]) < 5:
            examples[category].append({'text':row['text'],'gold':sorted(g),'before':sorted(before),'after':sorted(after)})

detect = read_jsonl(DATA/'dev_detect_clean.jsonl'); detection=Counter(); negexamples=[]
for i,row in enumerate(detect):
    events,_=model.extract(row['text'],schema); positive=bool(events); gold=bool(row['has_event'])
    detection[('tp' if positive else 'fn') if gold else ('fp' if positive else 'tn')] += 1
    if positive and not gold and len(negexamples)<10:
        negexamples.append({'text':row['text'],'events':events})
    if (i+1)%1000==0: print(f'检测集评估 {i+1}/{len(detect)}',flush=True)
result={'dataset':'dev_extract.jsonl','samples':len(old),'counts':dict(counts),'before':score(counts['before_tp'],counts['before_fp'],counts['before_fn']),'after':score(counts['after_tp'],counts['after_fp'],counts['after_fn']),'examples':examples,'detection_dataset':'dev_detect_clean.jsonl','detection_counts':dict(detection),'negative_examples':negexamples,'notes':'同一模型与数据，仅比较解码；样本级变化依据论元 FP+FN，检测集 has_event 是数据标注而非人工复核结论。'}
(ROOT/'outputs/bert_linear_空白修复及检测评估.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k not in ('examples','negative_examples')},ensure_ascii=False,indent=2))
