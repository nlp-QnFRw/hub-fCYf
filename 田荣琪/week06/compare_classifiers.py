"""课程分类对比。模型结构改编自课堂 bert/train_bert.py；不修改原材料。"""
import os
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('USE_TF', '0')
import argparse, csv, gc, hashlib, json, math, random, sys, time, unicodedata
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from safetensors.torch import save_file, load_file
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
LABELS = ['ar','en','es','hi','ja','ko','pt','ru','th','tr','vi','zh']
METHODS = ['tfidf','full_cls','full_mean','lora_mean']

def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def norm(s):
    return ''.join(c for c in unicodedata.normalize('NFKC',s).casefold() if unicodedata.category(c)[0] in 'LM')

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def prepare(data_dir, caps):
    seen=set(); parts={}; audit={}; rng=random.Random(42)
    for name, cap in zip(['train','dev','test'], caps):
        p=data_dir/(name+'.csv')
        with p.open(encoding='utf-8-sig',newline='') as f: raw=list(csv.DictReader(f))
        groups={k:[] for k in LABELS}; removed=0
        for i,d in enumerate(raw):
            k=norm(d['text'])
            if not k or k in seen: removed+=1; continue
            seen.add(k)
            assert d['label'] in groups
            groups[d['label']].append({'text':d['text'],'label':LABELS.index(d['label']), 'source_row':i})
        selected=[]
        for k in LABELS:
            rng.shuffle(groups[k]); selected.extend(groups[k][:cap])
        rng.shuffle(selected); parts[name]=selected
        audit[name]={'source_sha256':sha(p),'original':len(raw),'removed':removed,'selected':len(selected),
                     'counts':{k:sum(x['label']==LABELS.index(k) for x in selected) for k in LABELS}}
    sets=[{norm(x['text']) for x in parts[k]} for k in parts]
    assert all(not (sets[i]&sets[j]) for i in range(3) for j in range(i))
    audit['normalized_cross_split_overlap']=0
    return parts,audit

class Classifier(nn.Module):
    """复用课堂的编码器、掩码池化、随机失活和线性分类头组织方式。"""
    def __init__(self, path, method):
        super().__init__()
        self.bert=AutoModel.from_pretrained(str(path),local_files_only=True)
        self.pool='cls' if method=='full_cls' else 'mean'
        self.dropout=nn.Dropout(0.1)
        self.classifier=nn.Linear(self.bert.config.hidden_size,len(LABELS))
        if method=='lora_mean':
            from peft import LoraConfig, get_peft_model
            self.bert=get_peft_model(self.bert,LoraConfig(r=8,lora_alpha=16,lora_dropout=.05,target_modules=['query','value']))
        self.bert.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})

    def forward(self, ids, mask):
        hs=self.bert(input_ids=ids,attention_mask=mask,return_dict=True).last_hidden_state
        m=mask.unsqueeze(-1).to(hs.dtype)
        v=hs[:,0] if self.pool=='cls' else (hs*m).sum(1)/m.sum(1).clamp(min=1)
        return self.classifier(self.dropout(v))

def encode(tok, items):
    e=tok([x['text'] for x in items],padding='max_length',truncation=True,max_length=64,return_tensors='pt')
    return TensorDataset(e['input_ids'],e['attention_mask'],torch.tensor([x['label'] for x in items]))

def metrics(y,p):
    pr,re,f,_=precision_recall_fscore_support(y,p,labels=list(range(12)),average='macro',zero_division=0)
    return dict(accuracy=float(accuracy_score(y,p)),macro_precision=float(pr),macro_recall=float(re),macro_f1=float(f),
                confusion_matrix=confusion_matrix(y,p,labels=list(range(12))).tolist())

@torch.no_grad()
def evaluate(model, ds, device):
    model.eval(); ys=[]; ps=[]; total=0.; scores=[]
    for ids,mask,y in DataLoader(ds,batch_size=8):
        out=model(ids.to(device),mask.to(device)).float()
        assert out.shape==(len(y),12) and torch.isfinite(out).all()
        total+=nn.functional.cross_entropy(out,y.to(device),reduction='sum').item()
        scores.append(out.cpu()); ys.extend(y.tolist()); ps.extend(out.argmax(-1).cpu().tolist())
    return dict(metrics(ys,ps),loss=total/len(ds)),ps,torch.cat(scores)

def run_neural(method, data, tok, args, out):
    seed_all(42); device='cuda' if torch.cuda.is_available() else 'cpu'
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    model=Classifier(args.model_dir,method).to(device)
    nall=sum(p.numel() for p in model.parameters()); ntrain=sum(p.numel() for p in model.parameters() if p.requires_grad)
    target=next(n for n,p in model.named_parameters() if ('lora_B' in n if method=='lora_mean' else 'encoder.layer.0.attention.self.query.weight' in n))
    before=dict(model.named_parameters())[target].detach().cpu().clone()
    head_before=model.classifier.weight.detach().cpu().clone()
    frozen=next(((n,p.detach().cpu().clone()) for n,p in model.named_parameters() if not p.requires_grad),None)
    opt=torch.optim.AdamW([{'params':[p for p in model.bert.parameters() if p.requires_grad],'lr':2e-4 if method=='lora_mean' else 2e-5},
                          {'params':model.classifier.parameters(),'lr':1e-4}],weight_decay=.01)
    ds={k:encode(tok,v) for k,v in data.items()}
    loader=DataLoader(ds['train'],batch_size=4,shuffle=True,generator=torch.Generator().manual_seed(42))
    updates_per_epoch=math.ceil(len(loader)/2); epochs=1 if args.smoke else args.epochs
    planned=3 if args.smoke else updates_per_epoch*epochs
    scheduler=get_linear_schedule_with_warmup(opt,max(1,int(planned*.1)),planned)
    history=[]; step=0; best=-1.; started=time.perf_counter(); train_seconds=0.
    progress=tqdm(total=planned,desc=method, mininterval=3)
    for ep in range(epochs):
        model.train(); opt.zero_grad(set_to_none=True); total=0.; count=0; t=time.perf_counter()
        for i,(ids,mask,y) in enumerate(loader):
            window=min(2,len(loader)-(i//2)*2)
            with torch.autocast(device_type=device,dtype=torch.bfloat16,enabled=device=='cuda' and torch.cuda.is_bf16_supported()):
                logits=model(ids.to(device),mask.to(device)); loss=nn.functional.cross_entropy(logits.float(),y.to(device))
            assert torch.isfinite(loss), 'nonfinite loss'
            (loss/window).backward(); total+=loss.item()*len(y); count+=len(y)
            if (i+1)%2==0 or i+1==len(loader):
                grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                assert math.isfinite(float(grad))
                opt.step(); scheduler.step(); opt.zero_grad(set_to_none=True); step+=1
                progress.update(); progress.set_postfix(loss=f'{total/count:.3f}',refresh=False)
                if args.smoke and step>=3: break
        torch.cuda.synchronize(); train_seconds+=time.perf_counter()-t
        val,_,_=evaluate(model,ds['dev'],device)
        changed=float((dict(model.named_parameters())[target].detach().cpu()-before).abs().max())
        head_changed=float((model.classifier.weight.detach().cpu()-head_before).abs().max())
        assert changed>0 and head_changed>0, 'parameters did not update'
        if frozen: assert torch.equal(dict(model.named_parameters())[frozen[0]].detach().cpu(),frozen[1])
        entry={'epoch':ep+1,'updates':step,'train_loss':total/count,'validation':val,'target_max_change':changed,'head_max_change':head_changed}
        history.append(entry); write(out/'history.json',history); print(json.dumps(entry),flush=True)
        if val['macro_f1']>best:
            best=val['macro_f1']; best_epoch=ep+1
            selected={n:t.detach().cpu().contiguous() for n,t in model.state_dict().items() if method!='lora_mean' or 'lora_' in n or n.startswith('classifier.')}
            save_file(selected,str(out/'best.safetensors')); del selected
    progress.close()
    peak=torch.cuda.max_memory_allocated()/2**20
    del opt,scheduler,model; gc.collect(); torch.cuda.empty_cache()
    seed_all(42); model=Classifier(args.model_dir,method)
    state=load_file(str(out/'best.safetensors')); missing,unexpected=model.load_state_dict(state,strict=method!='lora_mean')
    assert not unexpected and all('lora_' not in n and not n.startswith('classifier.') for n in missing)
    assert all(torch.equal(model.state_dict()[n],t) for n,t in state.items())
    del state; model=model.to(device)
    val,_,a=evaluate(model,ds['dev'],device)
    assert abs(val['macro_f1']-best)<1e-10
    _,_,b=evaluate(model,ds['dev'],device); torch.testing.assert_close(a,b,rtol=1e-4,atol=1e-5)
    key='dev' if args.smoke else 'test'
    test,pred,_=evaluate(model,ds[key],device)
    result={'method':method,'status':'passed','validation':val,'test':test,'evaluation_split':key,'best_epoch':best_epoch,
            'updates':step,'total_parameters':nall,'trainable_parameters':ntrain,'train_seconds':train_seconds,
            'fit_wall_seconds':time.perf_counter()-started,'peak_allocated_MiB':peak,'save_reload_passed':True,
            'target_parameter':target,'target_updated':True,'classifier_updated':True,'frozen_probe_unchanged':frozen is not None,
            'test_predictions':pred}
    write(out/'result.json',result); del model,ds; gc.collect(); torch.cuda.empty_cache()
    return result

def run_tfidf(data,out,smoke):
    import joblib
    t=time.perf_counter(); vec=TfidfVectorizer(analyzer='char',ngram_range=(1,2),max_features=300000)
    x=vec.fit_transform([d['text'] for d in data['train']]); clf=LinearSVC(C=1.,random_state=42).fit(x,[d['label'] for d in data['train']])
    seconds=time.perf_counter()-t
    joblib.dump((vec,clf),out/'model.joblib'); v,c=joblib.load(out/'model.joblib')
    k='dev' if smoke else 'test'; text=[d['text'] for d in data[k]]; p=clf.predict(vec.transform(text)).tolist()
    assert p==c.predict(v.transform(text)).tolist()
    val=metrics([d['label'] for d in data['dev']],clf.predict(vec.transform([d['text'] for d in data['dev']])))
    result={'method':'tfidf','status':'passed','validation':val,'test':metrics([d['label'] for d in data[k]],p),'evaluation_split':k,
            'train_seconds':seconds,'total_parameters':int(clf.coef_.size+clf.intercept_.size),'trainable_parameters':int(clf.coef_.size+clf.intercept_.size),
            'peak_allocated_MiB':0,'save_reload_passed':True,'test_predictions':p}
    write(out/'result.json',result); return result

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    p=argparse.ArgumentParser(); p.add_argument('--smoke',action='store_true'); p.add_argument('--epochs',type=int,default=3)
    p.add_argument('--data-dir',type=Path,default=ROOT.parent/'week03/data/langid/v12')
    p.add_argument('--model-dir',type=Path,default=ROOT.parent/'week04/bert-base-chinese')
    p.add_argument('--methods',nargs='+',choices=METHODS,default=METHODS); args=p.parse_args()
    assert 1<=args.epochs<=5
    from datetime import datetime
    out=ROOT/'experiments'/datetime.now().strftime(('smoke_' if args.smoke else 'formal_')+'%Y%m%d_%H%M%S')
    out.mkdir(parents=True,exist_ok=False); print('OUTPUT',out,flush=True)
    data,audit=prepare(args.data_dir,[4,2,2] if args.smoke else [200,50,100])
    write(out/'data_audit.json',audit); write(out/'split_manifest.json',data)
    tok=AutoTokenizer.from_pretrained(str(args.model_dir),local_files_only=True)
    coverage={}
    for k,items in data.items():
        coverage[k]={}
        for label in range(12):
            seq=tok([d['text'] for d in items if d['label']==label],add_special_tokens=False)['input_ids']
            ids=[i for s in seq for i in s]; coverage[k][LABELS[label]]={'tokens':len(ids),'unknown':ids.count(tok.unk_token_id),'unknown_ratio':ids.count(tok.unk_token_id)/max(1,len(ids))}
    token_sets={k:{tuple(v) for v in tok([d['text'] for d in items],truncation=True,max_length=64)['input_ids']} for k,items in data.items()}
    coverage['token_collision_train_test']=len(token_sets['train']&token_sets['test'])
    write(out/'token_coverage.json',coverage)
    import importlib.metadata as md
    write(out/'config.json',{'args':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},'seed':42,'labels':LABELS,
          'python':sys.executable,'packages':{k:md.version(k) for k in ['torch','transformers','peft','scikit-learn','matplotlib','numpy','safetensors','tqdm']},
          'gpu':torch.cuda.get_device_name(0),'base_weight_sha256':sha(args.model_dir/'pytorch_model.bin'),'source_sha256':sha(Path(__file__))})
    results=[]
    for method in args.methods:
        d=out/method; d.mkdir()
        try: r=run_tfidf(data,d,args.smoke) if method=='tfidf' else run_neural(method,data,tok,args,d)
        except Exception:
            import traceback
            err=traceback.format_exc(); print(err,flush=True); r={'method':method,'status':'failed','traceback':err}; write(d/'failure.json',r)
            gc.collect(); torch.cuda.empty_cache()
        results.append(r); write(out/'summary.json',results)
    print('FINISHED',out,flush=True)
    if any(r['status']!='passed' for r in results): sys.exit(1)

if __name__=='__main__': main()
