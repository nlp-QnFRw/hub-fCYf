"""从真实续训记录生成静态学习曲线，不加载模型或训练。"""
import argparse
import json
import os
from pathlib import Path

HERE=Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR']=str(HERE/'outputs/.mplconfig')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    args=p.parse_args();r=args.run_dir
    read=lambda name:json.loads((r/name).read_text(encoding='utf-8'))
    base=read('baseline.json')
    history=read('validation_history.json') if (r/'validation_history.json').exists() else []
    rows=[{'update':0,'validation':base['validation'],'train_probe':base['train_probe']}]+history
    plt.rcParams['font.family']='Microsoft YaHei'
    plt.rcParams['axes.unicode_minus']=False
    fig,axes=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
    for key,label,color in [('train_probe','固定训练探针（128条）','#3979ad'),('validation','固定验证（200条）','#d06834')]:
        axes[0].plot([x['update'] for x in rows],[x[key]['loss'] for x in rows],marker='o',label=label,color=color)
        axes[1].plot([x['update'] for x in rows],[x[key]['token_accuracy'] for x in rows],marker='o',label=label,color=color)
    best=read('best.json')
    axes[0].scatter([best['update']],[best['validation']['loss']],s=90,facecolors='none',edgecolors='green',label='推荐模型')
    axes[1].axhline(.8,color='gray',ls='--',label='目标80%（不代表已达到）')
    axes[0].set_title('损失曲线：越低越好');axes[0].set_ylabel('回答位置交叉熵')
    axes[1].set_title('条件词元准确率：不是问答正确率');axes[1].set_ylabel('准确率')
    axes[1].yaxis.set_major_formatter(PercentFormatter(1));axes[1].set_ylim(0,1)
    for ax in axes:
        ax.set_xlabel('新增参数更新次数');ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.savefig(r/'learning_curves.png',dpi=160)
    fig.savefig(r/'learning_curves.svg')
    plt.close(fig)
