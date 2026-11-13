"""第五周：按课件因果化本地模型，实际进行回答段监督微调。

不调用收费接口，不修改原始资源。命令入口只在显式选择模式时运行。
"""
import argparse
from contextlib import nullcontext
from datetime import datetime
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['USE_TF'] = '0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import torch
import torch.nn.functional as F
from transformers import BertConfig, BertLMHeadModel, BertTokenizer
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = HERE.parent / 'week04' / 'bert-base-chinese'


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''): h.update(block)
    return h.hexdigest()


def state_digest(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(name.encode()); h.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def prompt_ids(tokenizer, instruction, cap=64):
    front = tokenizer.encode('用户：', add_special_tokens=False)
    back = tokenizer.encode('\n助手：', add_special_tokens=False)
    content = tokenizer.encode(instruction.strip(), add_special_tokens=False)
    room = cap - 1 - len(front) - len(back)
    if room < 1: raise ValueError('提示长度过短。')
    return [tokenizer.cls_token_id] + front + content[:room] + back


def prepare(tokenizer, length):
    """验证优先保留，按原文指令和实际输入编号剔除跨集合重复。"""
    sets, audit = {}, {'sources': {}, 'excluded': {}, 'truncated': {}}
    val_raw, train_raw = set(), set()
    val_encoded, train_encoded = set(), set()
    for split in ['val', 'train']:
        path = HERE / f'sft_{split}.jsonl'
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
        audit['sources'][split] = {'sha256': digest(path), 'rows': len(rows)}
        samples, excluded, truncated = [], [], 0
        for i, row in enumerate(rows):
            if not all(isinstance(row.get(k), str) and row[k].strip() for k in ['instruction','output']):
                excluded.append({'index':i,'reason':'empty_or_invalid'}); continue
            raw = ' '.join(row['instruction'].split())
            prompt = prompt_ids(tokenizer, row['instruction'], min(64, length//2))
            encoded = tuple(prompt)
            raw_seen = val_raw if split=='val' else val_raw | train_raw
            encoded_seen = val_encoded if split=='val' else val_encoded | train_encoded
            if raw in raw_seen or encoded in encoded_seen:
                excluded.append({'index':i,'reason':'duplicate_or_overlap'}); continue
            answer = tokenizer.encode(row['output'].strip(), add_special_tokens=False)
            if not answer: excluded.append({'index':i,'reason':'empty_encoded_answer'}); continue
            room = length-len(prompt)
            cut = len(answer)+1 > room
            target = answer[:room] if cut else answer+[tokenizer.sep_token_id]
            ids = prompt+target
            labels = [-100]*len(prompt)+target
            samples.append({'source_index':i,'instruction':row['instruction'],'reference':row['output'],
                            'input_ids':ids,'labels':labels,'prompt_ids':prompt,'truncated':cut})
            truncated += int(cut)
            (val_raw if split=='val' else train_raw).add(raw)
            (val_encoded if split=='val' else train_encoded).add(encoded)
        sets[split] = samples
        audit['excluded'][split] = excluded
        audit['truncated'][split] = truncated
        audit['sources'][split]['used_indices'] = [s['source_index'] for s in samples]
    audit['prompt_overlap'] = len(val_encoded & train_encoded)
    assert audit['prompt_overlap']==0 and sets['train'] and sets['val']
    audit['validation_signature'] = hashlib.sha256(json.dumps(sets['val'],ensure_ascii=False,sort_keys=True).encode()).hexdigest()
    return sets, audit


def load(path, device):
    tokenizer = BertTokenizer.from_pretrained(str(path), local_files_only=True)
    config = BertConfig.from_pretrained(str(path), local_files_only=True)
    config.is_decoder = True
    config.add_cross_attention = False
    config.use_cache = False
    config._attn_implementation = 'eager'
    model, info = BertLMHeadModel.from_pretrained(str(path), config=config,
                     local_files_only=True, output_loading_info=True)
    if info.get('missing_keys') or info.get('mismatched_keys') or info.get('error_msgs'):
        raise RuntimeError(str(info))
    return tokenizer, model.to(device), {k:sorted(info.get(k) or []) for k in ['missing_keys','unexpected_keys']}


def batch(sample, device):
    ids = torch.tensor([sample['input_ids']], device=device)
    labels = torch.tensor([sample['labels']], device=device)
    return {'input_ids':ids,'attention_mask':torch.ones_like(ids),'labels':labels}


def loss_sum(logits, labels):
    # 标签只错位一次：位置t的输出对应位置t+1的回答标签。
    return F.cross_entropy(logits[:,:-1].float().reshape(-1,logits.size(-1)),
                           labels[:,1:].reshape(-1),ignore_index=-100,reduction='sum')


@torch.inference_mode()
def evaluate(model, samples, device):
    model.eval(); total = correct = 0; summed = 0.
    for sample in tqdm(samples,desc='固定验证',leave=False):
        b = batch(sample,device); logits = model(**b,return_dict=True).logits
        targets = b['labels'][:,1:]; valid = targets != -100
        summed += loss_sum(logits,b['labels']).item(); total += valid.sum().item()
        correct += ((logits[:,:-1].argmax(-1)==targets)&valid).sum().item()
    return {'loss':summed/total,'token_accuracy':correct/total,'correct':correct,'targets':total,'samples':len(samples)}


@torch.inference_mode()
def generate(model, tokenizer, samples, device):
    model.eval(); results=[]
    for sample in samples[:5]:
        ids=torch.tensor([sample['prompt_ids']],device=device)
        result=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),do_sample=False,
                              max_new_tokens=48,eos_token_id=tokenizer.sep_token_id,
                              pad_token_id=tokenizer.pad_token_id,use_cache=True)
        results.append({'index':sample['source_index'],'instruction':sample['instruction'],
                        'reference':sample['reference'],'prediction':tokenizer.decode(result[0,ids.size(1):],skip_special_tokens=True),
                        'generated_ids':result[0,ids.size(1):].tolist()})
    return results


def correctness_checks(model, sample, device):
    model.eval(); b=batch(sample,device)
    with torch.no_grad():
        out=model(**b,return_dict=True)
        manual=loss_sum(out.logits,b['labels'])/(b['labels'][:,1:]!=-100).sum()
        torch.testing.assert_close(out.loss.float(),manual,atol=1e-5,rtol=1e-5)
        changed=b['input_ids'].clone(); pivot=changed.size(1)//2
        changed[:,pivot:]=(changed[:,pivot:]+1)%model.config.vocab_size
        other=model(input_ids=changed,attention_mask=b['attention_mask'],return_dict=True).logits
        torch.testing.assert_close(out.logits[:,:pivot],other[:,:pivot],atol=1e-5,rtol=1e-5)
    return {'manual_shifted_loss':'passed','causal_future_invariance':'passed',
            'input_shape':list(b['input_ids'].shape),'logits_shape':list(out.logits.shape)}


def run(args):
    if Path(sys.prefix).resolve()!=Path('D:/Anaconda3/envs/py312').resolve():
        raise RuntimeError('本项目仅使用 py312 环境。')
    random.seed(42); torch.manual_seed(42); torch.set_num_threads(4)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type!='cuda' and args.mode=='train': raise RuntimeError('正式实验要求现有显卡可用。')
    run_dir=HERE/'outputs'/(args.mode+'_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    run_dir.mkdir(parents=True)
    config={'mode':args.mode,'model_source':str(args.model.resolve()),'length':args.length,
            'batch_size':1,'accumulation':8,'epochs':1,'learning_rate':2e-5,'seed':42,
            'train_seconds_cap':2400,'updates_cap':400,'python':sys.executable,'script_sha256':digest(__file__)}
    write(run_dir/'config.json',config)
    try:
        tokenizer,model,loading=load(args.model,device)
        sets,audit=prepare(tokenizer,args.length)
        if args.mode=='smoke': sets={k:v[:16 if k=='train' else 8] for k,v in sets.items()}
        write(run_dir/'data_audit.json',audit)
        checks=correctness_checks(model,sets['train'][0],device);write(run_dir/'checks.json',checks)
        nparams=sum(p.numel() for p in model.parameters())
        if args.mode=='check':
            write(run_dir/'summary.json',{'status':'passed','checks':checks,'parameters':nparams,'training_executed':False});return
        baseline=evaluate(model,sets['val'],device)
        before_examples=generate(model,tokenizer,sets['val'],device)
        write(run_dir/'baseline.json',baseline);write(run_dir/'generation_before.json',before_examples)
        model.gradient_checkpointing_enable();model.config.use_cache=False
        optimizer=torch.optim.AdamW(model.parameters(),lr=2e-5,weight_decay=.01)
        tracked=model.cls.predictions.transform.dense.weight.detach().cpu().clone()
        initial_state=state_digest(model)
        rows=sets['train'].copy();random.Random(42).shuffle(rows)
        planned=min(400,math.ceil(len(rows)/8));warmup=max(1,math.ceil(planned*.05))
        bf16=device.type=='cuda' and torch.cuda.is_bf16_supported()
        config.update({'planned_updates':planned,'warmup_updates':warmup,'mixed_precision':'bfloat16' if bf16 else 'float32',
                       'parameters':nparams,'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad)})
        write(run_dir/'config.json',config)
        torch.cuda.reset_peak_memory_stats() if device.type=='cuda' else None
        start=time.monotonic(); records=[]; tokens=0; running=0.; model.train()
        for update in tqdm(range(planned),desc='监督微调更新'):
            if time.monotonic()-start>2400: break
            window=rows[update*8:(update+1)*8]
            denominator=sum(sum(x!=-100 for x in s['labels'][1:]) for s in window)
            optimizer.zero_grad(set_to_none=True)
            factor=(update+1)/warmup if update<warmup else max(0.,(planned-update)/(planned-warmup))
            for group in optimizer.param_groups:group['lr']=2e-5*factor
            window_loss=0.
            for sample in window:
                b=batch(sample,device)
                with torch.autocast(device_type=device.type,dtype=torch.bfloat16) if bf16 else nullcontext():
                    logits=model(**b,return_dict=True).logits
                    summed=loss_sum(logits,b['labels'])
                    if not torch.isfinite(summed):raise RuntimeError('训练损失非有限。')
                    loss=summed/denominator
                loss.backward();window_loss+=summed.item()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step();tokens+=denominator;running+=window_loss
            record={'update':update+1,'window_loss':window_loss/denominator,'cumulative_loss':running/tokens,
                    'targets':tokens,'gradient_norm':float(norm),'lr':optimizer.param_groups[0]['lr'],
                    'elapsed_seconds':time.monotonic()-start}
            records.append(record)
            with (run_dir/'metrics.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(record)+'\n')
        optimizer.zero_grad(set_to_none=True); del optimizer;gc.collect()
        torch.cuda.empty_cache() if device.type=='cuda' else None
        change=(tracked-model.cls.predictions.transform.dense.weight.detach().cpu()).abs().max().item()
        assert change>0 and all(torch.isfinite(p).all().item() for p in model.parameters())
        after=evaluate(model,sets['val'],device);examples=generate(model,tokenizer,sets['val'],device)
        saved_state=state_digest(model); assert initial_state!=saved_state
        write(run_dir/'post_training.json',{'after':after,'generation_after':examples,
              'state_sha256':saved_state,'parameter_max_change':change})
        probe=batch(sets['val'][0],device)
        with torch.no_grad(): reference=model(**probe,return_dict=True).logits.detach().cpu()
        model.save_pretrained(run_dir/'model');tokenizer.save_pretrained(run_dir/'model')
        del model;gc.collect()
        torch.cuda.empty_cache() if device.type=='cuda' else None
        _,reloaded,reload_info=load(run_dir/'model',device);reloaded.eval()
        assert state_digest(reloaded)==saved_state
        with torch.no_grad(): restored=reloaded(**probe,return_dict=True).logits.cpu()
        # 完整权重必须逐字节一致；近零分数使用2e-5绝对容差容纳单精度舍入差异。
        torch.testing.assert_close(reference,restored,rtol=1e-4,atol=2e-5)
        reloaded_metrics=evaluate(reloaded,sets['val'],device)
        assert reloaded_metrics['correct']==after['correct'] and abs(reloaded_metrics['loss']-after['loss'])<1e-5
        summary={'status':'passed','training_executed':True,'full_planned_epoch_completed':len(records)==planned,
                 'config':config,'loading':loading,'reload_loading':reload_info,'checks':checks,
                 'baseline':baseline,'after':after,'reloaded':reloaded_metrics,'updates':len(records),
                 'train_samples':len(sets['train']),'training_loss':running/tokens,'trained_targets':tokens,
                 'parameter_max_change':change,'state_sha256':saved_state,'reload_state_identical':True,
                 'reload_max_logit_difference':float((reference-restored).abs().max()),
                 'generation_before':before_examples,'generation_after':examples,
                 'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20 if device.type=='cuda' else None,
                 'seconds_training_and_acceptance':time.monotonic()-start}
        write(run_dir/'summary.json',summary)
        print('RESULT '+str(run_dir),flush=True)
        print(json.dumps({k:v for k,v in summary.items() if k not in ['generation_before','generation_after']},ensure_ascii=False),flush=True)
    except Exception as error:
        import traceback
        write(run_dir/'failure.json',{'status':'failed','error':str(error),'traceback':traceback.format_exc()})
        raise


if __name__=='__main__':
    for stream in [sys.stdout,sys.stderr]:stream.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['check','smoke','train'],required=True)
    parser.add_argument('--model',type=Path,default=DEFAULT_MODEL)
    parser.add_argument('--length',type=int,choices=[64,96,128],default=128)
    run(parser.parse_args())
