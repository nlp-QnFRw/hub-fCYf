"""BERT 推理：加载训练权重，将 token 标签还原为原文中的事件片段。"""
import torch
from transformers import AutoTokenizer
from bert.model import EventModel
from common.event_data import BERT, ROOT, segments, validate

class BertPredictor:
    """服务启动后按需加载模型，同一实例可重复处理多个请求。"""

    def __init__(self, decoder, device=None, merge_whitespace=True):
        # 仅允许相同实体的 I 标签跨纯空白续接；保留开关便于消融比较。
        self.merge_whitespace = merge_whitespace
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        path = ROOT / 'outputs' / ('bert_' + decoder + '.pt')
        if not path.exists():
            raise FileNotFoundError(f'尚未训练 {decoder} 模型：{path}。请先运行 python -m bert.train。')
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        model_path = ckpt.get('model_path', str(BERT))
        self.labels, self.max_length = (ckpt['labels'], ckpt['max_length'])
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, local_files_only=True)
        self.model = EventModel(model_path, len(self.labels), decoder).to(self.device)
        self.model.load_state_dict(ckpt['state_dict'])
        self.model.eval()

    def extract(self, text, schema):
        events, warnings = ([], [])
        # 按长度切分，base 用于把片段位置还原为全文位置。
        for base, part in segments(text, max_chars=max(1, self.max_length - 2)):
            enc = self.tokenizer(part, return_offsets_mapping=True, truncation=True, max_length=self.max_length, return_tensors='pt')
            offsets = enc.pop('offset_mapping')[0].tolist()
            ids, mask = (enc['input_ids'].to(self.device), enc['attention_mask'].to(self.device))
            with torch.inference_mode():
                path = self.model.predict(ids, mask)[0]
            # 合并连续的 B/I 标签，得到触发词或角色的字符片段。
            spans = []
            active = None
            for (s, e), label_id in zip(offsets, path):
                if s == e:
                    continue
                label = self.labels[label_id]
                gap = part[active['end']:s] if active and s >= active['end'] else None
                can_join = gap == '' or (self.merge_whitespace and gap is not None and gap.isspace())
                if label.startswith('I-') and active and active['label'] == label[2:] and can_join:
                    active['end'] = e
                else:
                    if active:
                        spans.append(active)
                    active = {'label': label[2:], 'start': s, 'end': e} if label.startswith(('B-', 'I-')) else None
            if active:
                spans.append(active)
            # 训练是单句单事件，这里每个片段只取第一个触发词。
            triggers = [s for s in spans if s['label'].startswith('TRG:')]
            if not triggers:
                continue
            if len(triggers) > 1:
                warnings.append('片段中存在多个触发词，BERT 仅输出第一个事件')
            trg = triggers[0]
            typ = trg['label'][4:]

            def loc(s):
                return {'text': part[s['start']:s['end']], 'start': base + s['start'], 'end': base + s['end']}
            args = [{'role': s['label'][4:], **loc(s)} for s in spans if s['label'].startswith('ARG:')]
            events.append({'type': typ, 'trigger': loc(trg), 'arguments': args})
        # 丢弃非法角色或无法对应原文的片段，同时返回复核提示。
        events, issues = validate(text, events, schema)
        return (events, warnings + issues)
