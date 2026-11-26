"""教学展示：真实训练标签与 BERT+Linear 原始 token 预测。"""
import torch
from bert.dataset import EventDataset
from common.event_data import DATA, read_jsonl


def explain(model, text, row=None):
    """同时返回原文字符、token 和标签；训练模式直接使用 EventDataset。"""
    tokenizer = model.tokenizer
    encoded = tokenizer(text, max_length=model.max_length, truncation=True, return_offsets_mapping=True)
    offsets = encoded['offset_mapping']
    token_ids = encoded['input_ids']
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    visible = [False] * len(text)
    # 用当前已训练模型做一次前向推理；它不改变参数。
    ids = torch.tensor([token_ids], device=model.device)
    mask = torch.tensor([encoded['attention_mask']], device=model.device)
    with torch.inference_mode():
        logits = model.model(ids, mask)
        probs = torch.softmax(logits, dim=-1)
        confidence, indices = probs.max(dim=-1)
    predicted_ids = indices[0].tolist()
    probabilities = confidence[0].tolist()
    predicted_labels = [model.labels[i] for i in predicted_ids]
    actual = None
    if row is not None:
        # 直接复用训练数据处理，目标标签和实际训练保持同一口径。
        actual = EventDataset([row], tokenizer, model.labels, model.max_length, crf=False)[0]['labels']
        tags = ['O'] * len(text)
        for span, name in [(a, 'ARG:' + a['role']) for a in row['event'].get('args', [])] + [(row['event']['trigger'], 'TRG:' + row['event']['type'])]:
            start, end = span['start'], span['end']
            if 0 <= start < end <= len(text) and text[start:end] == span['text']:
                tags[start] = 'B-' + name
                for i in range(start + 1, end):
                    tags[i] = 'I-' + name
    else:
        tags = [None] * len(text)
    character_predictions = [None] * len(text)
    character_matches = [None] * len(text)
    token_rows = []
    for i, (token, (start, end), label, prob) in enumerate(zip(tokens, offsets, predicted_labels, probabilities)):
        special = start == end
        target_id = actual[i] if actual is not None else None
        target = model.labels[target_id] if target_id is not None and target_id != -100 else None
        match = (target_id == predicted_ids[i]) if target_id is not None and target_id != -100 else None
        for j in range(start, end):
            visible[j] = True
            # 多字符 token 只预测一次；后续字符标签是展示用展开。
            projected = ('I-' + label[2:]) if j > start and label.startswith('B-') else label
            character_predictions[j] = projected
            character_matches[j] = match
            if row is None:
                tags[j] = projected
        token_rows.append({'index': i, 'token': token, 'start': start, 'end': end,
                           'label': (target if target is not None else '忽略 (-100)') if row is not None else (label if not special else '特殊 token（不展示预测）'),
                           'target': target, 'prediction': label if not special else None,
                           'match': match, 'probability': prob if not special else None, 'special': special})
    return {'text': text, 'mode': 'training' if row else 'prediction',
            'characters': [{'index': i, 'char': ch, 'label': tag, 'token_covered': visible[i], 'prediction': character_predictions[i], 'match': character_matches[i]}
                           for i, (ch, tag) in enumerate(zip(text, tags))],
            'tokens': token_rows, 'max_length': model.max_length,
            'truncated': len(tokenizer(text, add_special_tokens=True)['input_ids']) > model.max_length,
            'gold_event': row['event'] if row else None}


def find_sample(index):
    rows = read_jsonl(DATA / 'train_extract.jsonl')
    # 筛选短句便于观察，并保留来源；每次都从同一份训练集读取。
    short = [r for r in rows if 15 <= len(r['text']) <= 75]
    if not short:
        short = rows
    return short[index % len(short)], len(short)
