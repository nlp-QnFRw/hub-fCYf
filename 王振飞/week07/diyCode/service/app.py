"""本地 HTTP 服务：python -m service.app，然后访问 http://127.0.0.1:8000。"""
import json, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from common.event_data import ROOT, schema_at
 # 配置只读取一次；BERT 模型首次使用时加载并缓存。
schema = schema_at()
models = {}
lock = threading.Lock()

class Handler(BaseHTTPRequestHandler):

    def send(self, code, body, content_type='application/json; charset=utf-8'):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        """首页返回静态 HTML。"""
        pages = {'/': 'index.html', '/bio': 'bio.html'}
        if self.path not in pages:
            return self.send(404, {'error': '页面不存在'})
        return self.send(200, (ROOT / 'service' / 'static' / pages[self.path]).read_bytes(), 'text/html; charset=utf-8')

    def do_POST(self):
        """统一入口：校验输入 -> 根据 method 调用模型 -> 返回 JSON。"""
        if self.path not in ('/api/extract', '/api/bio'):
            return self.send(404, {'error': '接口不存在'})
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 100000:
                raise ValueError('请求大小不合法')
            obj = json.loads(self.rfile.read(length))
            if self.path == '/api/bio':
                from bert.predict import BertPredictor
                from bert.bio_visualize import explain, find_sample
                mode = obj.get('mode')
                if mode == 'training':
                    index = obj.get('index', 0)
                    if not isinstance(index, int) or isinstance(index, bool) or index < 0:
                        raise ValueError('样本编号必须是非负整数')
                    row, total = find_sample(index)
                    text = row['text']
                elif mode == 'prediction':
                    text = obj.get('text')
                    if not isinstance(text, str) or not text.strip():
                        raise ValueError('请输入句子')
                    if len(text) > 500:
                        raise ValueError('教学演示最多输入 500 字')
                    row = None
                else:
                    raise ValueError('未知展示模式')
                with lock:
                    if 'linear' not in models:
                        models['linear'] = BertPredictor('linear')
                    result = explain(models['linear'], text, row)
                    if row is None:
                        # 复用首页的真实业务抽取流程，而不是将展示用字符投影当成事件。
                        events, warnings = models['linear'].extract(text, schema)
                        if not events:
                            warnings.append('未识别出事件，请人工确认')
                        result['structured_result'] = {
                            'method': 'bert_linear', 'events': events,
                            'needs_review': bool(warnings), 'review_reasons': warnings,
                        }
                if row:
                    result['sample_index'] = index % total
                    result['sample_count'] = total
                return self.send(200, result)
            text = obj.get('text')
            method = obj.get('method')
            if not isinstance(text, str) or not text.strip():
                raise ValueError('请输入文本')
            if len(text) > 5000:
                raise ValueError('文本不能超过 5000 字')
            # 模型加载较重，锁避免并发时重复加载和争用同一 GPU。
            if method in ('bert_linear', 'bert_crf'):
                from bert.predict import BertPredictor
                decoder = method[5:]
                with lock:
                    if decoder not in models:
                        models[decoder] = BertPredictor(decoder)
                    events, warnings = models[decoder].extract(text, schema)
            elif method in ('llm_zero', 'llm_few'):
                from llm_zero_shot.llm_extract import extract
                events, warnings = extract(text, schema, method[4:])
            else:
                raise ValueError('不支持的抽取方法')
            if not events:
                warnings.append('未识别出事件，请人工确认')
            self.send(200, {'method': method, 'events': events, 'needs_review': bool(warnings), 'review_reasons': warnings})
        except (ValueError, FileNotFoundError, json.JSONDecodeError) as e:
            self.send(400, {'error': str(e)})
        except Exception as e:
            print(f'请求失败: {type(e).__name__}: {e}', flush=True)
            self.send(500, {'error': str(e) if isinstance(e, RuntimeError) else '服务处理失败，请查看服务端日志'})
if __name__ == '__main__':
    print('打开 http://127.0.0.1:8000', flush=True)
    ThreadingHTTPServer(('127.0.0.1', 8000), Handler).serve_forever()
