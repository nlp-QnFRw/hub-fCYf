"""只读加载本地实验最佳模型，推理或复核测试指标。"""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import compare_classifiers as c

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path)
    p.add_argument('--method',choices=c.METHODS,required=True);p.add_argument('--text')
    p.add_argument('--evaluate',action='store_true');p.add_argument('--model-dir',type=Path)
    args=p.parse_args();cfg=json.loads((args.run/'config.json').read_text(encoding='utf-8'))
    if not args.text and not args.evaluate:p.error('需要 --text 或 --evaluate')
    if args.evaluate:
        assert not cfg['args']['smoke'], '小规模实验不提供正式测试成绩'
        items=json.loads((args.run/'split_manifest.json').read_text(encoding='utf-8'))['test']
    else:items=[{'text':args.text,'label':0}]
    if args.method=='tfidf':
        import joblib
        vec,clf=joblib.load(args.run/'tfidf/model.joblib')
        pred=clf.predict(vec.transform([d['text'] for d in items])).tolist()
    else:
        path=args.model_dir or Path(cfg['args']['model_dir']);c.seed_all(42)
        model=c.Classifier(path,args.method)
        state=c.load_file(str(args.run/args.method/'best.safetensors'))
        missing,unexpected=model.load_state_dict(state,strict=args.method!='lora_mean')
        assert not unexpected and all('lora_' not in n and not n.startswith('classifier.') for n in missing)
        device='cuda' if c.torch.cuda.is_available() else 'cpu';model.to(device)
        tok=c.AutoTokenizer.from_pretrained(str(path),local_files_only=True)
        _,pred,_=c.evaluate(model,c.encode(tok,items),device)
    result=c.metrics([d['label'] for d in items],pred) if args.evaluate else {'text':args.text,'label':c.LABELS[pred[0]],'label_id':pred[0]}
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
