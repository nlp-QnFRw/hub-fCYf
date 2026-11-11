"""有限预算续训：固定验证、提前停止、训练探针和最佳权重保护。"""
import argparse
from datetime import datetime
import gc
import json
import math
from pathlib import Path
import random
import runpy
import sys
import time

HERE = Path(__file__).resolve().parent
DEFAULT_SOURCE = HERE/'outputs/train_20261111_235809_349617/model'


class Monitor:
    """按最低验证损失选模型，按实质改善监测平台；准确率只据实报告。"""
    def __init__(self, baseline, patience=3, delta=.01):
        self.best_loss = baseline['loss']
        self.significant_loss = baseline['loss']
        self.highest_accuracy = baseline['token_accuracy']
        self.patience, self.delta, self.stale = patience, delta, 0

    def observe(self, metrics):
        if not all(math.isfinite(metrics[k]) for k in ['loss','token_accuracy']):
            raise ValueError('验证指标非有限。')
        improved = metrics['loss'] < self.best_loss
        if improved: self.best_loss = metrics['loss']
        if metrics['loss'] < self.significant_loss-self.delta:
            self.significant_loss, self.stale = metrics['loss'], 0
        else: self.stale += 1
        self.highest_accuracy = max(self.highest_accuracy, metrics['token_accuracy'])
        return improved, self.stale >= self.patience


def main(args):
    if Path(sys.prefix).resolve()!=Path('D:/Anaconda3/envs/py312').resolve():
        raise RuntimeError('仅使用py312。')
    ns=runpy.run_path(str(HERE/'sft_bert.py'));torch=ns['torch'];write=ns['write']
    if not torch.cuda.is_available():raise RuntimeError('本轮需要已检查的显卡。')
    device=torch.device('cuda');torch.set_num_threads(4);torch.manual_seed(2026)
    run=HERE/'outputs'/('continue_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    run.mkdir(parents=True)
    config={'source':str(args.model.resolve()),'epochs_cap':args.epochs,'learning_rate':args.learning_rate,
            'seconds_cap':args.seconds,'eval_every':args.eval_every,'patience':args.patience,'min_delta':.01,
            'length':128,'accumulation':8,'batch_size':1,'optimizer_restored':False,'scheduler_restored':False,
            'target_token_accuracy':.8,'python':sys.executable,'script_sha256':ns['digest'](__file__)}
    write(run/'config.json',config)
    try:
        tokenizer,model,loading=ns['load'](args.model,device)
        source_weight_hash=ns['digest'](args.model/'model.safetensors')
        initial_state=ns['state_digest'](model)
        config['source_state_sha256']=initial_state
        sets,audit=ns['prepare'](tokenizer,128);write(run/'data_audit.json',audit)
        original=json.loads((HERE/'outputs/train_20261111_235809_349617/data_audit.json').read_text(encoding='utf-8'))
        if audit!=original:raise RuntimeError('数据或固定验证划分发生变化。')
        probe_indices=random.Random(17).sample(range(len(sets['train'])),128)
        train_probe=[sets['train'][i] for i in probe_indices]
        write(run/'train_probe_indices.json',probe_indices)
        baseline=ns['evaluate'](model,sets['val'],device)
        baseline_probe=ns['evaluate'](model,train_probe,device)
        write(run/'baseline.json',{'validation':baseline,'train_probe':baseline_probe})
        checks=ns['correctness_checks'](model,sets['val'][0],device);write(run/'checks.json',checks)
        monitor=Monitor(baseline,args.patience)
        best={'update':0,'epoch':0,'directory':str(args.model.resolve()),'validation':baseline,
              'train_probe':baseline_probe,'state_sha256':initial_state}
        write(run/'best.json',best)
        optimizer=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=.01)
        model.gradient_checkpointing_enable();model.config.use_cache=False
        updates_per_epoch=math.ceil(len(sets['train'])/8)
        planned=updates_per_epoch*args.epochs;warmup=max(1,math.ceil(planned*.05))
        bf16=torch.cuda.is_bf16_supported()
        config.update({'planned_updates':planned,'warmup':warmup,'precision':'bfloat16' if bf16 else 'float32',
                       'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad)})
        write(run/'config.json',config);torch.cuda.reset_peak_memory_stats()
        history=[];update=0;last_eval=0;reason='epoch_budget';start=time.monotonic()
        total_loss=0.;total_targets=0

        def validate(epoch):
            nonlocal best,last_eval
            val=ns['evaluate'](model,sets['val'],device)
            probe=ns['evaluate'](model,train_probe,device)
            improved,stop=monitor.observe(val)
            row={'update':update,'epoch':epoch,'validation':val,'train_probe':probe,
                 'loss_gap':val['loss']-probe['loss'],'stale_checks':monitor.stale,
                 'overfit_signal':val['loss']>baseline['loss']+.05 and probe['loss']<baseline_probe['loss']-.05,
                 'elapsed_seconds':time.monotonic()-start}
            if improved:
                directory=run/'checkpoints'/f'update_{update:06d}'/'model'
                state=ns['state_digest'](model)
                model.save_pretrained(directory);tokenizer.save_pretrained(directory)
                best={'update':update,'epoch':epoch,'directory':str(directory),'validation':val,
                      'train_probe':probe,'state_sha256':state}
                write(directory.parent/'metrics.json',row);write(run/'best.json',best)
            history.append(row);write(run/'validation_history.json',history)
            print('VALIDATION '+json.dumps(row,ensure_ascii=False),flush=True)
            last_eval=update;model.train()
            return stop

        progress=ns['tqdm'](total=planned,desc='有限续训')
        try:
            stop=False
            for epoch in range(1,args.epochs+1):
                rows=sets['train'].copy();random.Random(2026+epoch).shuffle(rows)
                model.train()
                for offset in range(0,len(rows),8):
                    if time.monotonic()-start>=args.seconds:
                        reason='time_budget';stop=True;break
                    window=rows[offset:offset+8]
                    count=sum(sum(t!=-100 for t in s['labels'][1:]) for s in window)
                    optimizer.zero_grad(set_to_none=True)
                    factor=(update+1)/warmup if update<warmup else max(0.,(planned-update)/(planned-warmup))
                    for group in optimizer.param_groups:group['lr']=args.learning_rate*factor
                    summed=0.
                    for sample in window:
                        b=ns['batch'](sample,device)
                        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=bf16):
                            logits=model(**b,return_dict=True).logits
                            value=ns['loss_sum'](logits,b['labels'])
                        if not torch.isfinite(value):raise RuntimeError('非有限训练损失。')
                        (value/count).backward();summed+=value.item()
                    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                    optimizer.step();update+=1;total_loss+=summed;total_targets+=count
                    record={'update':update,'epoch':epoch,'window_loss':summed/count,
                            'cumulative_loss':total_loss/total_targets,'gradient_norm':float(norm),
                            'lr':optimizer.param_groups[0]['lr'],'elapsed_seconds':time.monotonic()-start}
                    with (run/'metrics.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(record)+'\n')
                    progress.update(1)
                    if update%args.eval_every==0 or offset+8>=len(rows):
                        if validate(epoch):reason='early_stopping';stop=True;break
                if stop:break
            if update!=last_eval and update:validate(epoch)
        finally:progress.close()
        optimizer.zero_grad(set_to_none=True);del optimizer;del model;gc.collect();torch.cuda.empty_cache()
        _,model,reload_info=ns['load'](Path(best['directory']),device);model.eval()
        if ns['state_digest'](model)!=best['state_sha256']:raise RuntimeError('最佳模型状态不一致。')
        checked=ns['evaluate'](model,sets['val'],device)
        if checked['correct']!=best['validation']['correct'] or abs(checked['loss']-best['validation']['loss'])>1e-5:
            raise RuntimeError('重载验证不一致。')
        examples=ns['generate'](model,tokenizer,sets['val'],device)
        if ns['digest'](args.model/'model.safetensors')!=source_weight_hash:
            raise RuntimeError('来源权重文件发生改变。')
        summary={'status':'passed','training_executed':update>0,'updates':update,'stop_reason':reason,
                 'config':config,'baseline':baseline,'baseline_train_probe':baseline_probe,'best':best,
                 'reloaded_validation':checked,'best_state_identical_after_reload':True,
                 'highest_observed_accuracy':monitor.highest_accuracy,'target_80_reached':monitor.highest_accuracy>=.8,
                 'validation_history':history,'generation_after':examples,'reload_loading':reload_info,
                 'source_loading':loading,'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
                 'total_seconds':time.monotonic()-start,'source_files_unmodified':True,
                 'note':'Best within this finite run; not proof of global optimum or absence of overfitting. Validation has been reused; no new independent test.'}
        write(run/'summary.json',summary)
        print('RESULT '+str(run),flush=True)
        print(json.dumps({k:v for k,v in summary.items() if k not in ['generation_after','validation_history']},ensure_ascii=False),flush=True)
    except Exception as error:
        import traceback
        write(run/'failure.json',{'error':str(error),'traceback':traceback.format_exc()});raise


if __name__=='__main__':
    for stream in [sys.stdout,sys.stderr]:stream.reconfigure(encoding='utf-8')
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,default=DEFAULT_SOURCE)
    p.add_argument('--epochs',type=int,default=4)
    p.add_argument('--learning-rate',type=float,default=1e-5)
    p.add_argument('--eval-every',type=int,default=125)
    p.add_argument('--patience',type=int,default=3)
    p.add_argument('--seconds',type=int,default=2400)
    a=p.parse_args()
    if not 1<=a.epochs<=4 or not 1<=a.seconds<=2400 or not 1<=a.eval_every<=375 or not 1<=a.patience<=3:
        p.error('参数超过本次预算上限。')
    if not math.isfinite(a.learning_rate) or not 0<a.learning_rate<=2e-5:p.error('学习率需大于0且不超过2e-5。')
    main(a)
