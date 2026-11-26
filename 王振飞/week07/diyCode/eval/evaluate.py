"""评测入口：python -m eval.evaluate --decoder linear|crf。

比较预测与 dev 金标的触发词、论元和完整事件，采用文本精确匹配。
"""
import argparse, json
from pathlib import Path
from common.event_data import DATA, read_jsonl, schema_at
from bert.predict import BertPredictor

def score(tp, fp, fn):
    """TP 是命中数，FP 是多报数，FN 是漏报数。"""
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    return {'precision': precision, 'recall': recall, 'f1': 2 * precision * recall / (precision + recall) if precision + recall else 0, 'tp': tp, 'fp': fp, 'fn': fn}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--decoder', choices=['linear', 'crf'], default='linear')
    ap.add_argument('--data-dir', type=Path, default=DATA)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--device', default=None)
    args = ap.parse_args()
    rows = read_jsonl(args.data_dir / 'dev_extract.jsonl')
    rows = rows[:args.limit] if args.limit else rows
    predictor = BertPredictor(args.decoder, args.device)
    schema = schema_at(args.data_dir)
    counts = {k: [0, 0, 0] for k in ('trigger', 'argument', 'event_exact')}
    # 每条样本单独比较，避免相同文本跨样本误匹配。
    for row in rows:
        pred, _ = predictor.extract(row['text'], schema)
        gold = row['event']

        def convert(ev):
            typ = ev['type']
            trg = ev.get('trigger') or {}
            args = ev.get('arguments', ev.get('args', []))
            trigger = {(typ, trg.get('text'))} if trg.get('text') else set()
            arguments = {(typ, a['role'], a['text']) for a in args}
            return (trigger, arguments)
        gt, ga = convert(gold)
        pt, pa = (set(), set())
        for event in pred:
            t, a = convert(event)
            pt |= t
            pa |= a
        pairs = {'trigger': (gt, pt), 'argument': (ga, pa), 'event_exact': ({(tuple(sorted(gt)), tuple(sorted(ga)))}, {(tuple(sorted(convert(e)[0])), tuple(sorted(convert(e)[1]))) for e in pred})}
        for key, (g, p) in pairs.items():
            counts[key][0] += len(g & p)
            counts[key][1] += len(p - g)
            counts[key][2] += len(g - p)
    print(json.dumps({'samples': len(rows), 'method': args.decoder, **{k: score(*v) for k, v in counts.items()}}, ensure_ascii=False, indent=2))
if __name__ == '__main__':
    main()
