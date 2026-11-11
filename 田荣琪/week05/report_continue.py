"""整理已完成续训的真实记录，不训练或更改评分。"""
import argparse
import json
from pathlib import Path


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    args=parser.parse_args();r=args.run_dir.resolve()
    s=json.loads((r/'summary.json').read_text(encoding='utf-8'))
    reasons={'epoch_budget':'额外轮次预算用尽','time_budget':'时间预算用尽','early_stopping':'验证平台触发提前停止'}
    base=s['baseline'];best=s['best']['validation'];bp=s['baseline_train_probe']
    minimum=base['loss'];previous_probe=bp['loss'];warnings=[]
    for row in s['validation_history']:
        if row['validation']['loss']>minimum+.01 and row['train_probe']['loss']<previous_probe:
            warnings.append(row['update'])
        minimum=min(minimum,row['validation']['loss']);previous_probe=row['train_probe']['loss']
    text=['# 第五周续训结果', '',
          f"实际新增{s['updates']}次更新，停止原因：{reasons.get(s['stop_reason'],s['stop_reason'])}。",
          f"预算内最低验证损失模型位于新增第{s['best']['update']}次更新。",
          '这是有限训练中的观测结果，不能据此证明全局峰值或从未过拟合。','',
          '| 指标 | 续训前 | 推荐模型 |','| --- | ---: | ---: |',
          f"| 验证损失 | {base['loss']:.6f} | {best['loss']:.6f} |",
          f"| 条件词元准确率 | {100*base['token_accuracy']:.3f}% | {100*best['token_accuracy']:.3f}% |",
          f"| 正确词元 / {base['targets']} | {base['correct']} | {best['correct']} |",'',
          f"最高观测条件词元准确率：{100*s['highest_observed_accuracy']:.3f}%。",
          f"80%目标：{'达到' if s['target_80_reached'] else '未达到'}。这不是完整问答正确率。",'',
          '| 新增更新 | 验证损失 | 验证准确率 | 训练探针损失 | 训练探针准确率 | 损失差距 |',
          '| ---: | ---: | ---: | ---: | ---: | ---: |',
          f"| 0 | {base['loss']:.6f} | {100*base['token_accuracy']:.3f}% | {bp['loss']:.6f} | {100*bp['token_accuracy']:.3f}% | {base['loss']-bp['loss']:.6f} |"]
    for row in s['validation_history']:
        v,p=row['validation'],row['train_probe']
        text.append(f"| {row['update']} | {v['loss']:.6f} | {100*v['token_accuracy']:.3f}% | {p['loss']:.6f} | {100*p['token_accuracy']:.3f}% | {row['loss_gap']:.6f} |")
    text+=['',f'训练探针继续改善而验证损失高于此前最低值0.01以上的检查点：{warnings}。',
           '没有该信号也不构成无过拟合证明；验证数据已经用于历史评估和本轮选模，不是新独立测试。','',
           f"最佳权重重载状态一致：{s['best_state_identical_after_reload']}；重载完整验证与记录一致。",
           f"显存已分配峰值：{s['peak_allocated_mib']:.2f}MiB；本轮训练与验收计时：{s['total_seconds']:.2f}秒。",'',
           '推荐模型：', '```text',s['best']['directory'],'```','',
           '本轮没有推送或修改提交仓库。原模型与历史结果保留。','',
           '## 固定生成样本（原始解码，未改写）','']
    for example in s['generation_after']:
        text += [f"问题：{example['instruction']}",f"生成：{example['prediction']}",'']
    (r/'REPORT.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
