"""验证角色编码、回答监督边界、参数继承和检查点往返。"""

import json
import tempfile
import unittest
from pathlib import Path

import torch

from prepare_sft_data import convert_file
from sft_common import (
    IGNORE_INDEX, SPECIAL_TOKENS, Sample, SFTDataset, collate_batch, extend_vocab,
    generate, initialize_sft_model, load_sft_checkpoint, load_tagged, make_model,
    make_tagged_text, parse_tagged_text, save_checkpoint,
)
from sft_train import run_epoch


class SFTTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.base_vocab = {char: index for index, char in enumerate("问题答案甲乙")}
        self.cfg = dict(model="transformer", embed_dim=8, hidden_dim=16,
                        num_layers=1, nhead=2, seq_len=4, dropout=0.0)
        base_model = make_model(self.cfg, len(self.base_vocab))
        self.base = {"args": self.cfg, "char2idx": self.base_vocab,
                     "model_state": {k: v.clone() for k, v in base_model.state_dict().items()}}
        self.samples = [Sample("问题甲", "答案乙新"), Sample("问", "答")]
        self.vocab, _ = extend_vocab(self.base_vocab, self.samples)

    def test_conversion_roundtrip_and_idempotence(self):
        rows = [{"instruction": "问题：\n原样保留 空格", "output": '答案含"引号"'}]
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory) / "source.jsonl", Path(directory) / "tagged.jsonl"
            source.write_text(json.dumps(rows[0], ensure_ascii=False) + "\n", encoding="utf-8")
            original = source.read_bytes()
            self.assertEqual(convert_file(source, target), 1)
            first = target.read_bytes()
            self.assertEqual(load_tagged(target), [Sample(**rows[0])])
            self.assertEqual(set(json.loads(target.read_text(encoding="utf-8"))), {"text"})
            convert_file(source, target)
            self.assertEqual(target.read_bytes(), first)
            self.assertEqual(source.read_bytes(), original)
            with self.assertRaises(ValueError):
                convert_file(source, source)

    def test_invalid_tagged_data_and_long_sample_are_rejected(self):
        for text in ("无标识", "<|user|>问<|end|>",
                     "<|user|>问<|assistant|>答<|end|><|end|>",
                     "<|user|><|assistant|>答<|end|>",
                     "<|user|>问<|assistant|><|pad|><|end|>"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_tagged_text(text)
        with self.assertRaises(ValueError):
            SFTDataset(self.samples, self.vocab, max_seq_len=3)

    def test_atomic_markers_and_shifted_answer_labels(self):
        sample = self.samples[0]
        self.assertEqual(parse_tagged_text(make_tagged_text(sample.instruction, sample.output)), sample)
        data = SFTDataset(self.samples, self.vocab, max_seq_len=16)
        inputs, labels = data[0]
        assistant_position = 1 + len(sample.instruction)
        self.assertEqual(inputs[0].item(), self.vocab[SPECIAL_TOKENS["user"]])
        self.assertEqual(inputs[assistant_position].item(), self.vocab[SPECIAL_TOKENS["assistant"]])
        self.assertTrue(labels[:assistant_position].eq(IGNORE_INDEX).all().item())
        self.assertEqual(labels[assistant_position:].tolist(),
                         [self.vocab[c] for c in sample.output] + [self.vocab[SPECIAL_TOKENS["end"]]])
        self.assertEqual(len(inputs), len(sample.instruction) + len(sample.output) + 2)
        batch_inputs, batch_labels = collate_batch([data[0], data[1]], self.vocab[SPECIAL_TOKENS["pad"]])
        self.assertEqual(batch_inputs.shape, batch_labels.shape)
        short_length = len(data[1][0])
        self.assertTrue(batch_labels[1, short_length:].eq(IGNORE_INDEX).all().item())
        self.assertTrue(batch_inputs[1, short_length:].eq(self.vocab[SPECIAL_TOKENS["pad"]]).all().item())

    def test_parameter_inheritance_and_no_validation_vocab_leak(self):
        model, cfg = initialize_sft_model(self.base, self.vocab, 16, 0.0)
        self.assertEqual(cfg["seq_len"], 16)
        for char, index in self.base_vocab.items():
            self.assertEqual(self.vocab[char], index)
        for name, old in self.base["model_state"].items():
            actual = model.state_dict()[name]
            if name in {"embed.weight", "fc.weight", "fc.bias"}:
                actual = actual[:len(self.base_vocab)]
            elif name == "position_embed.weight":
                self.assertTrue(torch.equal(actual, old[torch.arange(16) % 4]))
                actual = actual[:4]
            self.assertTrue(torch.equal(actual, old), name)
        validation = SFTDataset([Sample("验证", "答")], self.vocab, 16)
        self.assertEqual(validation.unknown_instruction, 2)
        self.assertNotIn("验", self.vocab)

    def test_backward_checkpoint_roundtrip_and_padding_causality(self):
        model, cfg = initialize_sft_model(self.base, self.vocab, 16, 0.0)
        data = SFTDataset(self.samples, self.vocab, 16)
        batch = collate_batch([data[0], data[1]], self.vocab[SPECIAL_TOKENS["pad"]])
        old_weight = model.fc.weight.detach().clone()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        metrics = run_epoch(model, [batch], torch.device("cpu"), optimizer)
        self.assertTrue(torch.isfinite(torch.tensor(metrics["loss"])).item())
        self.assertEqual(metrics["tokens"], sum(len(s.output) + 1 for s in self.samples))
        self.assertFalse(torch.equal(old_weight, model.fc.weight))
        model.eval()
        with torch.no_grad():
            expected = model(batch[0])
            short_input = data[1][0].unsqueeze(0)
            torch.testing.assert_close(expected[1, :short_input.size(1)], model(short_input)[0], atol=1e-5, rtol=1e-5)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            save_checkpoint(path, model, self.vocab, cfg, {"seed": 42}, metrics)
            loaded, checkpoint = load_sft_checkpoint(path, torch.device("cpu"))
            self.assertEqual(checkpoint["char2idx"], self.vocab)
            with torch.no_grad():
                torch.testing.assert_close(expected, loaded(batch[0]), atol=0, rtol=0)

    def test_generation_stops_at_end_and_suppresses_other_markers(self):
        vocab = self.vocab

        class EndModel(torch.nn.Module):
            max_seq_len = 16

            def forward(self, inputs):
                scores = torch.zeros((*inputs.shape, len(vocab)), device=inputs.device)
                scores[..., vocab[SPECIAL_TOKENS["user"]]] = 100
                scores[..., vocab[SPECIAL_TOKENS["end"]]] = 99
                return scores

        result = generate(EndModel(), vocab, "问题", torch.device("cpu"))
        self.assertEqual(result, {"text": "", "ended": True, "generated_tokens": 1})
        with self.assertRaises(ValueError):
            generate(EndModel(), vocab, "问题", torch.device("cpu"), max_new_tokens=0)


if __name__ == "__main__":
    unittest.main()
