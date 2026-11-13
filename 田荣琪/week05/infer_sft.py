"""加载本地训练后模型进行贪心回答，不执行训练或写文件。"""
import argparse
from pathlib import Path
import runpy
import sys

if __name__ == '__main__':
    for stream in [sys.stdout,sys.stderr]: stream.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--text',required=True)
    parser.add_argument('--prompt-cap',type=int,default=64)
    args=parser.parse_args()
    if not args.text.strip():parser.error('问题不能为空。')
    if not 16<=args.prompt_cap<=64:parser.error('提示长度必须在16至64之间。')
    if Path(sys.prefix).resolve()!=Path('D:/Anaconda3/envs/py312').resolve():parser.error('只使用 py312 环境。')
    ns=runpy.run_path(str(Path(__file__).with_name('sft_bert.py')))
    torch=ns['torch'];device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer,model,info=ns['load'](args.model,device)
    sample={'source_index':0,'instruction':args.text,'reference':'',
            'prompt_ids':ns['prompt_ids'](tokenizer,args.text,args.prompt_cap)}
    print(ns['generate'](model,tokenizer,[sample],device)[0]['prediction'])
