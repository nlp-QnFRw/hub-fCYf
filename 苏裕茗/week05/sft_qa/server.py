from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import sys
import io
import os

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import torch
from transformers import EncoderDecoderModel, BertTokenizer

MODEL_PATH = "./bert-sft-qa"
MAX_LENGTH = 128

# ========== 加载模型 ==========
sft_model = None
sft_tokenizer = None
sft_device = None
sft_load_error = None

try:
    if os.path.exists(MODEL_PATH) and os.path.exists(os.path.join(MODEL_PATH, "config.json")):
        print("正在加载 SFT 问答模型...")
        sft_tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
        sft_model = EncoderDecoderModel.from_pretrained(MODEL_PATH)
        sft_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sft_model.to(sft_device)
        sft_model.eval()
        print("SFT 模型加载成功")
    else:
        sft_load_error = f"模型目录不存在: {MODEL_PATH}，请先运行 train.py 训练"
        print(sft_load_error)
except Exception as e:
    sft_load_error = f"模型加载失败: {e}"
    print(sft_load_error)


def generate_answer(question, max_length=128):
    if sft_model is None:
        raise RuntimeError(sft_load_error or "模型未加载")

    inputs = sft_tokenizer(
        question,
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(sft_device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = sft_model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
            decoder_start_token_id=sft_tokenizer.cls_token_id,
            pad_token_id=sft_tokenizer.pad_token_id,
            eos_token_id=sft_tokenizer.sep_token_id,
        )

    answer = sft_tokenizer.decode(output_ids[0], skip_special_tokens=True).replace(" ", "")
    return answer


HTML_PAGE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>BERT SFT 问答</title>
    <style>
        body { font-family: "Microsoft YaHei", sans-serif; max-width: 700px; margin: 60px auto; padding: 20px; background: #f5f5f5; }
        .box { background: #fff; padding: 30px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
        h1 { margin-top: 0; color: #333; }
        textarea { width: 100%; padding: 12px; font-size: 16px; border: 1px solid #ddd; border-radius: 4px; box-sizing: border-box; resize: vertical; min-height: 80px; }
        button { padding: 10px 24px; font-size: 16px; margin-top: 12px; cursor: pointer; background: #28a745; color: #fff; border: none; border-radius: 4px; }
        button:hover { background: #218838; }
        button:disabled { background: #ccc; cursor: not-allowed; }
        .result { margin-top: 24px; padding: 16px; background: #f8f9fa; border-radius: 4px; border-left: 4px solid #28a745; }
        .result-label { font-weight: bold; color: #555; margin-bottom: 8px; }
        .result-text { font-size: 16px; line-height: 1.6; color: #333; }
        .warn { padding: 12px; background: #fff3cd; border-radius: 4px; color: #856404; margin-bottom: 16px; }
    </style>
</head>
<body>
    <div class="box">
        <h1>BERT SFT 问答生成</h1>
        <p>输入问题，微调后的模型会自动生成答案。</p>

        <div id="warnBox" style="display:none" class="warn"></div>

        <textarea id="question" placeholder="例如：给定两个数字，请选择其中的较大数。数字1: 238 数字2: 564">给定两个数字，请选择其中的较大数。数字1: 238 数字2: 564</textarea>
        <button id="btn" onclick="ask()">生成答案</button>

        <div class="result" id="result" style="display:none">
            <div class="result-label">模型回答：</div>
            <div class="result-text" id="answer"></div>
        </div>
    </div>

    <script>
        fetch('/status')
            .then(r => r.json())
            .then(data => {
                if (!data.ready) {
                    const box = document.getElementById('warnBox');
                    box.style.display = 'block';
                    box.textContent = '提示：' + data.message;
                    document.getElementById('btn').disabled = true;
                }
            });

        async function ask() {
            const question = document.getElementById('question').value.trim();
            const resultDiv = document.getElementById('result');
            const answerDiv = document.getElementById('answer');
            const btn = document.getElementById('btn');

            if (!question) return;
            btn.disabled = true;
            answerDiv.textContent = '生成中，请稍候...';
            resultDiv.style.display = 'block';

            try {
                const resp = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question })
                });
                const data = await resp.json();
                if (data.error) {
                    answerDiv.textContent = '错误：' + data.error;
                } else {
                    answerDiv.textContent = data.answer;
                }
            } catch (e) {
                answerDiv.textContent = '请求失败：' + e;
            }
            btn.disabled = false;
        }
    </script>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        sys.stdout.write("%s - - [%s] %s\n" %
                         (self.address_string(), self.log_date_time_string(), fmt % args))

    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode('utf-8'))
        elif self.path == '/status':
            if sft_model:
                self.send_json(200, {"ready": True, "message": "模型已就绪"})
            else:
                self.send_json(200, {"ready": False, "message": sft_load_error or "模型未加载"})
        else:
            self.send_error(404)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_POST(self):
        if self.path == '/predict':
            length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(length)
            try:
                payload = json.loads(body.decode('utf-8'))
                question = payload.get('question', '')
                answer = generate_answer(question)
                self.send_json(200, {"answer": answer})
            except Exception as e:
                self.send_json(500, {"error": str(e)})
        else:
            self.send_error(404)

    def send_json(self, code, obj):
        self.send_response(code)
        self.send_header('Content-type', 'application/json; charset=utf-8')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(obj, ensure_ascii=False).encode('utf-8'))


if __name__ == '__main__':
    print("启动服务器 → http://localhost:8000")
    server = HTTPServer(('localhost', 8000), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n服务器已关闭")
