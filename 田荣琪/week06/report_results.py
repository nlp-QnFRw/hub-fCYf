"""从真实实验记录生成可提交的小体积报告，不复制数据原文和权重。"""
import argparse,csv,json,shutil,sys,hashlib
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','DejaVu Sans']
plt.rcParams['axes.unicode_minus']=False

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    p=argparse.ArgumentParser(); p.add_argument('--run',required=True,type=Path); args=p.parse_args()
    root=Path(__file__).resolve().parent; run=args.run.resolve(); out=root/'results'/run.name
    out.mkdir(parents=True,exist_ok=True)
    read=lambda n:json.loads((run/n).read_text(encoding='utf-8'))
    all_results=read('summary.json'); good=[r for r in all_results if r['status']=='passed']
    assert len(good)==4 and not read('config.json')['args']['smoke']
    data=read('split_manifest.json'); labels=read('config.json')['labels']
    for n in ['summary.json','config.json','data_audit.json','token_coverage.json']:
        shutil.copyfile(run/n,out/n)
    indices={k:[{'source_row':d['source_row'],'label':d['label'],'text_sha256':hashlib.sha256(d['text'].encode()).hexdigest()} for d in a] for k,a in data.items()}
    (out/'split_indices.json').write_text(json.dumps(indices,indent=2),encoding='utf-8')
    lines=['# 实际实验结果','',f'实验目录：`{run.name}`。所有方法使用同一独立测试集。','',
           '| 方法 | 准确率 | 宏平均精确率 | 宏平均召回率 | 宏平均F1 | 训练秒数 | 可训练参数 | 峰值显存MiB |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    errors={}; rows=[]; best=max(good,key=lambda r:r['test']['macro_f1'])
    for r in good:
        m=r['method']; t=r['test']; cm=np.array(t['confusion_matrix']); rec=np.diag(cm)/cm.sum(1)
        lines.append(f"| {m} | {t['accuracy']:.4f} | {t['macro_precision']:.4f} | {t['macro_recall']:.4f} | {t['macro_f1']:.4f} | {r['train_seconds']:.2f} | {r['trainable_parameters']:,} | {r['peak_allocated_MiB']:.1f} |")
        for i,lang in enumerate(labels): rows.append({'method':m,'language':lang,'recall':float(rec[i]),'support':int(cm[i].sum())})
        pairs=sorted([(int(cm[i,j]),labels[i],labels[j]) for i in range(12) for j in range(12) if i!=j],reverse=True)
        errors[m]={'top_confusions':[{'count':v,'true':a,'predicted':b} for v,a,b in pairs[:5]],
                   'examples':[{'source_row':x['source_row'],'true':labels[x['label']],'predicted':labels[y],
                                'text_sha256':hashlib.sha256(x['text'].encode()).hexdigest()} for x,y in zip(data['test'],r['test_predictions']) if x['label']!=y][:20]}
        h=run/m/'history.json'
        if h.exists(): shutil.copyfile(h,out/(m+'_history.json'))
        plt.figure(figsize=(8,7)); plt.imshow(cm, cmap='Blues'); plt.xticks(range(12),labels); plt.yticks(range(12),labels)
        plt.xlabel('预测标签');plt.ylabel('真实标签');plt.title(m);plt.colorbar();plt.tight_layout();plt.savefig(out/(m+'_confusion.png'),dpi=160);plt.close()
    with (out/'per_class.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    (out/'error_analysis.json').write_text(json.dumps(errors,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    fig,ax=plt.subplots(1,2,figsize=(11,4)); names=[r['method'] for r in good]
    ax[0].bar(names,[r['test']['macro_f1'] for r in good]);ax[0].set_ylim(0,1);ax[0].set_title('测试集宏平均F1');ax[0].tick_params(axis='x',rotation=20)
    ax[1].bar(names,[r['train_seconds'] for r in good]);ax[1].set_title('训练耗时（秒，使用不同设备）');ax[1].tick_params(axis='x',rotation=20)
    fig.tight_layout();fig.savefig(out/'comparison.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for r in good:
        h=run/r['method']/'history.json'
        if not h.exists():continue
        a=json.loads(h.read_text(encoding='utf-8')); x=[d['epoch'] for d in a]
        axes[0].plot(x,[d['train_loss'] for d in a],label=r['method']);axes[1].plot(x,[d['validation']['macro_f1'] for d in a],label=r['method'])
    axes[0].set_title('训练损失');axes[1].set_title('验证集宏平均F1')
    for ax in axes:ax.set_xlabel('训练轮次');ax.legend()
    fig.tight_layout();fig.savefig(out/'learning_curves.png',dpi=180);plt.close(fig)
    print('\n'.join(lines)); print('BEST',best['method']);print('REPORT',out)

if __name__=='__main__':main()
