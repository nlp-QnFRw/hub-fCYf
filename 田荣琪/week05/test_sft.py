"""验证回答监督边界和不同回答长度下的梯度归一化，不训练真实模型。"""
from pathlib import Path
import argparse
import runpy
import unittest

ns=runpy.run_path(str(Path(__file__).with_name('sft_bert.py')))
torch=ns['torch']


class SupervisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer=ns['BertTokenizer'].from_pretrained(str(ns['DEFAULT_MODEL']),local_files_only=True)
        cls.sets,cls.audit=ns['prepare'](cls.tokenizer,128)

    def test_prompt_keeps_assistant_suffix(self):
        ids=ns['prompt_ids'](self.tokenizer,'学习'*100)
        tail=self.tokenizer.encode('\n助手：',add_special_tokens=False)
        self.assertEqual(ids[-len(tail):],tail)
        self.assertEqual(len(ids),64)

    def test_no_train_validation_prompt_overlap(self):
        train={tuple(x['prompt_ids']) for x in self.sets['train']}
        val={tuple(x['prompt_ids']) for x in self.sets['val']}
        self.assertFalse(train & val)
        self.assertEqual(len(val),len(self.sets['val']))

    def test_response_only_labels_and_stop(self):
        for split in self.sets.values():
            for row in split:
                n=len(row['prompt_ids'])
                self.assertEqual(row['labels'][:n],[-100]*n)
                self.assertEqual(row['labels'][n:],row['input_ids'][n:])
                self.assertGreater(len(row['labels']),n)
                self.assertLessEqual(len(row['input_ids']),128)
                if not row['truncated']:self.assertEqual(row['labels'][-1],self.tokenizer.sep_token_id)

    def test_originals_match_recorded_hash(self):
        for split in ['train','val']:
            self.assertEqual(ns['digest'](ns['HERE']/f'sft_{split}.jsonl'),self.audit['sources'][split]['sha256'])

    def test_gradient_accumulation_is_target_weighted(self):
        torch.manual_seed(4)
        logits=torch.randn(2,5,7,requires_grad=True)
        labels=torch.tensor([[-100,-100,2,3,4],[-100,-100,-100,-100,1]])
        count=(labels[:,1:]!=-100).sum()
        whole=ns['loss_sum'](logits,labels)/count
        expected=torch.autograd.grad(whole,logits,retain_graph=True)[0]
        accum=sum(ns['loss_sum'](logits[i:i+1],labels[i:i+1])/count for i in range(2))
        actual=torch.autograd.grad(accum,logits)[0]
        torch.testing.assert_close(actual,expected)
        self.assertTrue((actual[:,3][1]!=0).any())
        self.assertTrue((actual[:,-1]==0).all())


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,default=ns['DEFAULT_MODEL'])
    args=parser.parse_args()
    ns['DEFAULT_MODEL']=args.model
    unittest.main(argv=['test_sft.py'])
