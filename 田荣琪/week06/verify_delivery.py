"""实际结果、原材料完整性和提交体积的最终验收。"""
import sys,argparse,ast,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import compare_classifiers as c
from safetensors import safe_open

FILES=['README.md','compare_classifiers.py','infer_classifier.py','report_results.py','test_comparison.py',
       'verify_delivery.py','requirements.txt','experiment_settings.json','TASK.md','PLANS.md','PROGRESS.md',
       'DECISIONS.md','SUBMISSION.md','.gitignore']

def main():
    sys.stdout.reconfigure(encoding='utf-8');c.torch.set_num_threads(2)
    p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);args=p.parse_args()
    root=Path(__file__).resolve().parent;run=args.run.resolve();out=root/'results'/run.name
    read=lambda n:json.loads((run/n).read_text(encoding='utf-8'))
    results=read('summary.json');data=read('split_manifest.json');cfg=read('config.json')
    assert len(results)==4 and all(r['status']=='passed' for r in results)
    assert not cfg['args']['smoke'] and cfg['seed']==42
    assert cfg['source_sha256']==c.sha(root/'compare_classifiers.py')
    teacher=json.loads((root/'materials_inventory.json').read_text(encoding='utf-8'))
    for f in teacher:assert c.sha(root/f['path'])==f['sha256'],f['path']
    for name in FILES:
        path=root/name; assert path.is_file() and path.stat().st_size>0,name
        if path.suffix=='.py':ast.parse(path.read_text(encoding='utf-8'))
    y=[x['label'] for x in data['test']]
    for r in results:
        assert r['evaluation_split']=='test' and len(r['test_predictions'])==1200
        actual=c.metrics(y,r['test_predictions'])
        assert actual=={k:r['test'][k] for k in actual},r['method']
        assert r['save_reload_passed']
        if r['method']!='tfidf':
            assert r['updates']==900 and r['target_updated'] and r['classifier_updated']
            with safe_open(str(run/r['method']/'best.safetensors'),framework='pt',device='cpu') as f:
                for key in f.keys():assert c.torch.isfinite(f.get_tensor(key)).all(),key
    splitsets=[{c.norm(x['text']) for x in data[k]} for k in ['train','dev','test']]
    assert all(not (splitsets[i]&splitsets[j]) for i in range(3) for j in range(i))
    checks={'status':'passed','teacher_files_unchanged':len(teacher),'source_matches_training_sha256':True,
            'all_four_test_metrics_independently_recomputed':True,'all_saved_neural_tensors_finite':True,
            'normalized_split_overlap':0,'all_training_and_reload_checks_passed':True,
            'syntax_checks_passed':True,'git_operations_executed':False,
            'cuda_runtime':c.torch.version.cuda,'device':c.torch.cuda.get_device_name(0),
            'bfloat16_supported':c.torch.cuda.is_bf16_supported()}
    c.write(out/'checks.json',checks)
    paths=[root/n for n in FILES]+[f for f in out.rglob('*') if f.is_file() and f.name!='submission_audit.json']
    assert all(f.suffix not in ['.safetensors','.bin','.pt','.pth','.joblib'] for f in paths)
    assert all(f.stat().st_size<100*2**20 for f in paths)
    entries=[{'path':str(f.relative_to(root)),'bytes':f.stat().st_size,'sha256':c.sha(f)} for f in paths]
    audit={'status':'passed','file_count':len(entries),'total_bytes':sum(x['bytes'] for x in entries),
           'maximum_file_bytes':max(x['bytes'] for x in entries),'files_above_100_MiB':[],
           'scope':'允许清单与本实验轻量结果；本文件自身不参与内容摘要计算','files':entries,
           'excluded_large_weights':[{'path':str(f.relative_to(root)),'bytes':f.stat().st_size} for f in (root/'experiments').rglob('*.safetensors')]}
    c.write(out/'submission_audit.json',audit);print(json.dumps(checks,ensure_ascii=False));print('SUBMISSION_BYTES',audit['total_bytes'])

if __name__=='__main__':main()
