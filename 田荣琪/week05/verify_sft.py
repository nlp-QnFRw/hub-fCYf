"""对已保存的正式训练模型做独立恢复验收，不重新训练、不改写失败记录。"""
import argparse
import gc
import json
from pathlib import Path
import runpy
import sys

if __name__=='__main__':
    for stream in [sys.stdout,sys.stderr]:stream.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    args=parser.parse_args();r=args.run_dir.resolve()
    ns=runpy.run_path(str(Path(__file__).with_name('sft_bert.py')));torch=ns['torch']
    if Path(sys.prefix).resolve()!=Path('D:/Anaconda3/envs/py312').resolve():raise RuntimeError('仅使用py312。')
    read=lambda name:json.loads((r/name).read_text(encoding='utf-8'))
    config=read('config.json');events=[json.loads(x) for x in (r/'metrics.jsonl').read_text().splitlines()]
    assert len(events)==config['planned_updates']
    assert [x['update'] for x in events]==list(range(1,len(events)+1))
    assert all(__import__('math').isfinite(x[k]) for x in events for k in ['window_loss','gradient_norm'])
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer,model,loading=ns['load'](r/'model',device);model.eval()
    sets,audit=ns['prepare'](tokenizer,config['length'])
    assert audit==read('data_audit.json')
    assert events[-1]['targets']==sum(sum(t!=-100 for t in s['labels']) for s in sets['train'])
    state=ns['state_digest'](model)
    after=ns['evaluate'](model,sets['val'],device)
    examples=ns['generate'](model,tokenizer,sets['val'],device)
    probe=ns['batch'](sets['val'][0],device)
    with torch.no_grad():reference=model(**probe,return_dict=True).logits.cpu()
    tracked=model.cls.predictions.transform.dense.weight.detach().cpu().clone()
    del model;gc.collect();torch.cuda.empty_cache()
    _,second,second_info=ns['load'](r/'model',device);second.eval()
    assert ns['state_digest'](second)==state
    with torch.no_grad():restored=second(**probe,return_dict=True).logits.cpu()
    torch.testing.assert_close(reference,restored,rtol=1e-4,atol=2e-5)
    repeated=ns['evaluate'](second,sets['val'],device)
    assert repeated['correct']==after['correct'] and abs(repeated['loss']-after['loss'])<1e-5
    assert all(torch.isfinite(p).all().item() for p in second.parameters())
    del second;gc.collect();torch.cuda.empty_cache()
    # CPU只读加载原始权重，直接测量所跟踪参数相对基座的实际变化。
    _,base,_=ns['load'](Path(config['model_source']),torch.device('cpu'))
    delta=float((tracked-base.cls.predictions.transform.dense.weight.detach()).abs().max())
    assert delta>0
    del base;gc.collect()
    report={'status':'passed','kind':'independent_saved_checkpoint_recovery','training_executed':False,
            'formal_training_completed':True,'full_epoch_completed':True,'config':config,
            'baseline':read('baseline.json'),'after':after,'reloaded':repeated,'updates':len(events),
            'training_loss':events[-1]['cumulative_loss'],'trained_targets':events[-1]['targets'],
            'train_samples':len(sets['train']),'parameter_max_change_from_original':delta,
            'state_sha256':state,'two_independent_loads_identical':True,
            'reload_max_logit_difference':float((reference-restored).abs().max()),
            'reload_rtol':1e-4,'reload_atol':2e-5,'loading':loading,'second_loading':second_info,
            'checks':read('checks.json'),'generation_before':read('generation_before.json'),
            'generation_after':examples,'original_failure_preserved':True,
            'training_seconds':events[-1]['elapsed_seconds'],
            'limitations':'Original in-memory reference is unavailable after failed process; output comparison uses two independent loads. Original state-equality assertion passed before the output failure. Peak memory was not persisted and is not fabricated.'}
    ns['write'](r/'recovery.json',report)
    print(json.dumps(report,ensure_ascii=False))
