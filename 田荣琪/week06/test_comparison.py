"""关键数据与池化行为的独立检查。"""
import sys, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import compare_classifiers as c
from unittest.mock import patch

class Checks(unittest.TestCase):
    def test_normalization(self):
        self.assertEqual(c.norm('Ａ B!'),c.norm('ab'))
        self.assertNotEqual(c.norm('中文'),c.norm('日本語'))

    def test_macro_metric(self):
        y=list(range(12)); p=y.copy(); p[0]=1
        r=c.metrics(y,p)
        self.assertAlmostEqual(r['accuracy'],11/12)
        self.assertAlmostEqual(r['macro_f1'],(10+2/3)/12)
        self.assertEqual(sum(map(sum,r['confusion_matrix'])),12)

    def test_padding_excluded(self):
        class Dummy(c.nn.Module):
            config=type('Config',(),{'hidden_size':2})()
            def gradient_checkpointing_enable(self,**kw): pass
            def forward(self,input_ids,**kw):
                x=input_ids.float()
                return type('Out',(),{'last_hidden_state':c.torch.stack((x,x*2),dim=-1)})()
        with patch.object(c.AutoModel,'from_pretrained',return_value=Dummy()):
            m=c.Classifier('unused','full_mean').eval()
        mask=c.torch.tensor([[1,1,0]])
        a=m(c.torch.tensor([[1,2,0]]),mask)
        b=m(c.torch.tensor([[1,2,999]]),mask)
        c.torch.testing.assert_close(a,b,rtol=0,atol=0)

if __name__=='__main__': unittest.main(testRunner=unittest.TextTestRunner(stream=sys.stdout,verbosity=2))
