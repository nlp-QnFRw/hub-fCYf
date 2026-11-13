"""提前停止与最佳模型选择的行为测试，不训练模型。"""
import runpy
from pathlib import Path
import unittest

Monitor=runpy.run_path(str(Path(__file__).with_name('continue_sft.py')))['Monitor']


class MonitorTests(unittest.TestCase):
    def test_rising_accuracy_does_not_hide_worsening_loss(self):
        m=Monitor({'loss':2.,'token_accuracy':.5})
        for i in range(3):
            best,stop=m.observe({'loss':2.1+i*.1,'token_accuracy':.6})
            self.assertFalse(best)
            self.assertEqual(stop,i==2)
        self.assertEqual(m.best_loss,2.)

    def test_small_improvements_do_not_reset_plateau_forever(self):
        m=Monitor({'loss':2.,'token_accuracy':.5})
        for value in [1.999,1.998,1.997]:best,stop=m.observe({'loss':value,'token_accuracy':.51})
        self.assertTrue(best);self.assertTrue(stop)

    def test_meaningful_improvement_resets_patience(self):
        m=Monitor({'loss':2.,'token_accuracy':.5})
        m.observe({'loss':2.,'token_accuracy':.5})
        self.assertEqual(m.observe({'loss':1.9,'token_accuracy':.55}),(True,False))
        self.assertEqual(m.stale,0)

    def test_nonfinite_evaluation_is_rejected(self):
        m=Monitor({'loss':2.,'token_accuracy':.5})
        with self.assertRaises(ValueError):m.observe({'loss':float('nan'),'token_accuracy':.8})


if __name__=='__main__':unittest.main()
