#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
miniGPT 对话台 —— 零依赖后端（标准库 http.server + SSE 流式输出）

    python chat.py                    # 默认 http://127.0.0.1:8000，自动开浏览器
    python chat.py --port 9000
    python chat.py --ckpt ckpt_novel/best.pt --device cpu    # 换权重 / 换设备

它做的事只有三件：
  1. 把 chat.html 发出去（前端是一个自包含的单文件）
  2. 收 /api/chat 的 POST，拼好上下文，跑自回归生成
  3. 每生成一个字符就 SSE 推一行给浏览器 —— 这就是"流式输出"

没有 Flask / FastAPI，只用了 PyTorch 和标准库。
"""

#region 0. 导入库   【模块:标准库 argparse/json/os/socket/sys/threading/time/webbrowser/http.server + torch】
import argparse
import json
import os
import socket
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch

HERE = os.path.dirname(os.path.abspath(__file__))  # 脚本所在目录
sys.path.insert(0, HERE)  # 保证从任何工作目录运行都能 import model

from model import EOS, load_ckpt  # noqa: E402
#endregion

#region 0.1 全局状态   【集合:STATE(模型/分词器/设备/元信息)，GPU_LOCK(串行锁)】
STATE = {}                      # model / tok / device / cfg
GPU_LOCK = threading.Lock()     # 单卡串行：两个人同时点发送就排队，不会 OOM
#endregion


#region 1. 中断信号   【类:_Abort】
class _Abort(Exception):
    """客户端中途关掉页面 / 点了停止时，用它从 on_token 里跳出来。

    为什么用异常而不是 return False：on_token 是 generate 内部回调，
    没有"提前结束"的返回约定，只能靠抛异常把控制流打断。
    """
#endregion


# ══════════════════════════════════════════════════════════════════════════════
# 2. 上下文组装
# ══════════════════════════════════════════════════════════════════════════════
#region 2. 上下文组装：多轮历史 → 一条纯文本   【函数:build_prompt(历史列表,本次输入,模式,最大长度),return 字符串】
def build_prompt(history, user_text, mode, max_ctx):
    """把多轮历史拼成一条纯文本 —— 因为 Base Model 只认"文风"，不认角色。

    mode="chat"     对话模式：用户：…\\n助手：…  （有对话的样子，但它其实只是在模仿这个格式）
    mode="continue" 续写模式：把历史直接连起来写   （它真正擅长的用法）

    最后按 max_ctx 从**末尾**截断：宁可丢掉最早的几轮，也要保住当前这一问。

    返回：拼好的字符串，长度 ≤ max_ctx。
    """
    if mode == "chat":
        # 对话模式：模拟"用户：/助手："的 few-shot 模板；模型会学着模仿这个格式
        parts = [f"用户：{h['u']}\n助手：{h['a']}\n" for h in history]  # 列表，每轮一段
        parts.append(f"用户：{user_text}\n助手：")  # 末尾留一个空的"助手："等它往下接
    else:
        # 续写模式：历史直接首尾相连，不加任何角色标记
        parts = [h["u"] + h["a"] + "\n" for h in history]
        parts.append(user_text)
    text = "".join(parts)  # 拼成一整段
    # 为什么从末尾截：上下文窗口只有 block_size，前面的是"老话题"，当前这一问才是重点
    return text[-max_ctx:] if len(text) > max_ctx else text
#endregion


# ══════════════════════════════════════════════════════════════════════════════
# 3. 生成（带流式回调 + 可中断）
# ══════════════════════════════════════════════════════════════════════════════
#region 3. 生成主函数：流式 + 可中断   【函数:run_generation(输入,参数dict,emit回调,aborted判断),return None】
def run_generation(prompt, p, emit, aborted):
    """跑一次自回归生成，每出一个字就通过 emit 推给前端。

    p 是前端传来的参数字典：temperature / top_k / top_p / rep_penalty / max_new /
    history / mode / ban_eos。
    """
    model, tok, device = STATE["model"], STATE["tok"], STATE["device"]
    cfg = model.cfg

    max_new = int(p.get("max_new", 160))  # 最多生成多少字
    # 预算：总窗口 - 要生成的字数 = 提示词能用多长；至少留 16 个字，否则模型没上下文可依
    text = build_prompt(p.get("history") or [], prompt, p.get("mode", "chat"),
                        max(16, cfg.block_size - max_new))

    ids = tok.encode(text)  # 列表，长度 = 提示词字符数
    if not ids:                                  # 空输入兜底，不然 forward 会崩
        ids = tok.encode("。")
    x = torch.tensor([ids], device=device)  # (1, len(ids))

    # 对话模式下让它在"用户："处收口，不然它会自问自答演完一场戏
    stops = ["\n用户：", "用户："] if p.get("mode") == "chat" else []
    ban = [EOS] if p.get("ban_eos", True) else None  # 禁 EOS：不然模型十几个字就收尾

    out_ids, st = [], time.time()  # out_ids 收集所有已生成 id，st 是计时起点
    sent = ""                                    # 已经推给前端的文本，用来算增量

    #region 3.1 每个 token 的回调：算增量 + 检查中断   【函数:on_token(id),return None】
    def on_token(tid):
        """generate 每生成一个 token 就调它一次。"""
        nonlocal sent  # 要改外层的 sent，必须声明
        if aborted():
            raise _Abort()  # 前端已经关了/点了停止，立刻中断生成
        out_ids.append(tid)
        full = tok.decode(out_ids)               # 字符级，全量解码最省心
        # 为什么不用"只解码新 token"：中文里一个字符可能由多个 id 组成（这里不会，但通用做法如此），
        # 全量解码再取差集，永远不会漏字也不会重复
        delta = full[len(sent):]  # 这次新增的那几个字
        sent = full
        if delta:
            emit({"type": "tok", "t": delta})  # 推给前端，这就是"流式"
        if any(s in full for s in stops):
            raise _Abort()  # 出现"用户："说明它在自问自答，主动掐断
    #endregion

    try:
        model.generate(x, max_new_tokens=max_new,
                       temperature=float(p.get("temperature", 0.85)),
                       top_k=int(p.get("top_k", 40)) or None,  # 前端传 0 表示"不用 top-k"
                       top_p=float(p.get("top_p", 0.9)),
                       rep_penalty=float(p.get("rep_penalty", 1.2)),
                       # 注意 ban_eos 与 stop_at_eos 是反的：禁了 EOS 就不能"遇 EOS 停"
                       stop_at_eos=not p.get("ban_eos", True),
                       ban_ids=ban, on_token=on_token)
    except _Abort:
        pass  # 主动中断，不是错误，静默退出

    #region 3.2 结束后推一条统计
    dt = max(1e-6, time.time() - st)  # 兜底 1e-6，防止除零
    emit({"type": "done", "stats": {
        "tokens": len(out_ids),  # 生成了多少个字
        "secs": round(dt, 2),
        "tps": round(len(out_ids) / dt, 1),  # 字/秒
        "ctx": len(ids),  # 本次用了多少上下文
        "ctx_max": cfg.block_size,  # 上下文上限，前端显示"352/512"用
        "prompt": text,  # 完整提示词，方便前端展开查看
        "stopped": aborted(),  # 是否被用户手动停止
    }})
    #endregion
#endregion


# ══════════════════════════════════════════════════════════════════════════════
# 4. HTTP 服务
# ══════════════════════════════════════════════════════════════════════════════
#region 4. HTTP 处理器：路由 + SSE   【类:Handler】
class Handler(BaseHTTPRequestHandler):
    """三个路由：GET / 发页面，GET /api/info 发模型信息，POST /api/chat 流式生成。"""
    protocol_version = "HTTP/1.1"     # SSE 需要长连接
    server_version = "miniGPT/1.0"

    #region 4.1 日志：只打印 API 调用
    def log_message(self, fmt, *a):   # 只打印 API 调用，静态请求太吵
        """覆盖父类的日志方法，过滤掉静态资源的请求，只留接口调用。"""
        try:
            msg = fmt % a          # 注意：send_error 传进来的 a[0] 可能是 HTTPStatus，不能再做字符串判断
        except Exception:
            msg = str(fmt)  # 格式化失败就退化成原始 fmt，别让日志把服务搞崩
        if "/api/" in msg or "/chat" in msg:
            sys.stderr.write("  %s\n" % msg)
    #endregion

    # ---------- 工具 ----------
    #region 4.2 发一个完整响应   【函数:_send(状态码,响应体,类型),return None】
    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        """统一出口：写状态码、三个头、再写 body。"""
        if isinstance(body, str):
            body = body.encode("utf-8")  # 字符串统一编码成 bytes
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))  # HTTP/1.1 必须给长度，否则浏览器一直等
        self.send_header("Cache-Control", "no-store")  # 页面和接口都不缓存，改代码立刻生效
        self.end_headers()
        self.wfile.write(body)
    #endregion

    #region 4.3 发一个 JSON 响应   【函数:_json(对象,状态码),return None】
    def _json(self, obj, code=200):
        """ensure_ascii=False：中文直接以 UTF-8 输出，不转成 \\uXXXX。"""
        self._send(code, json.dumps(obj, ensure_ascii=False))
    #endregion

    # ---------- GET ----------
    #region 4.4 GET 路由   【函数:do_GET(),return None】
    def do_GET(self):
        """处理页面请求和模型信息请求。"""
        path = self.path.split("?")[0]  # 去掉查询串，只看路径
        if path in ("/", "/index.html", "/chat.html"):
            fp = os.path.join(HERE, "chat.html")  # 前端单文件
            if not os.path.exists(fp):
                return self._send(500, "找不到 chat.html，请把它放在 chat.py 同目录", "text/plain; charset=utf-8")
            with open(fp, "rb") as f:  # 用二进制读，避免编码转换
                return self._send(200, f.read(), "text/html; charset=utf-8")
        if path == "/api/info":
            m, cfg = STATE["model"], STATE["model"].cfg
            # 前端启动时拉一次，用来显示"16.14M 参数 / 上下文 512"这些信息
            return self._json({
                "params": m.num_params(),
                "layers": cfg.n_layer, "d_model": cfg.d_model, "heads": cfg.n_head,
                "block_size": cfg.block_size, "vocab": cfg.vocab_size,
                "device": STATE["device_name"],
                "ckpt": STATE["ckpt_name"],
                "val_ppl": STATE.get("val_ppl"),  # 可能读不到，前端要能处理 None
            })
        return self._send(404, "not found", "text/plain; charset=utf-8")
    #endregion

    # ---------- HEAD（浏览器/探活工具会先来一发）----------
    #region 4.5 HEAD 路由：只回头不回体   【函数:do_HEAD(),return None】
    def do_HEAD(self):
        """HEAD 和 GET 一样的头、但没有 body。浏览器预检和 curl -I 会走这条路。"""
        path = self.path.split("?")[0]
        fp = os.path.join(HERE, "chat.html")
        if path in ("/", "/index.html", "/chat.html") and os.path.exists(fp):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(os.path.getsize(fp)))  # 只报大小，不写内容
            self.end_headers()
            return
        self.send_response(200 if path == "/api/info" else 404)
        self.send_header("Content-Length", "0")  # HEAD 必须显式声明 0 长度
        self.end_headers()
    #endregion

    # ---------- POST /api/chat ----------
    #region 4.6 POST /api/chat：流式生成入口   【函数:do_POST(),return None】
    def do_POST(self):
        """收 JSON 参数 → 开 SSE → 边生成边推 → 最后推 [DONE]。"""
        if self.path.split("?")[0] != "/api/chat":
            return self._send(404, "not found", "text/plain; charset=utf-8")

        #region 4.6.1 读参数 + 校验
        n = int(self.headers.get("Content-Length") or 0)  # 没有头就当 0，别崩
        try:
            p = json.loads(self.rfile.read(n) or b"{}")  # body 为空时给个空 dict
        except Exception:
            return self._json({"error": "bad json"}, 400)

        prompt = (p.get("prompt") or "").strip()
        if not prompt:
            return self._json({"error": "请输入内容"}, 400)  # 空输入直接拒掉，省得白跑一次
        #endregion

        #region 4.6.2 开 SSE 通道
        # SSE（Server-Sent Events）：响应头声明 text/event-stream，之后一直不关连接，
        # 服务端每写一段 "data: xxx\n\n" 浏览器就触发一次 onmessage
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")  # 一条连接只服务一次生成
        self.send_header("X-Accel-Buffering", "no")  # 关掉 nginx 一类反代的缓冲，否则流会攒成一坨
        self.end_headers()
        #endregion

        #region 4.6.3 断开检测 + emit 实现
        closed = threading.Event()  # 事件对象，用来在回调里传递"客户端已断开"

        def aborted():
            """给生成函数用的判断：客户端走了吗？"""
            return closed.is_set()

        def emit(obj):
            """往 SSE 通道写一条 JSON。写 socket 失败 = 浏览器关了页面。"""
            try:
                self.wfile.write(("data: " + json.dumps(obj, ensure_ascii=False) + "\n\n").encode("utf-8"))
                self.wfile.flush()  # 必须 flush，否则会攒在缓冲区，前端看不到"流"
            except (BrokenPipeError, ConnectionResetError, OSError):
                closed.set()          # 浏览器关了页面 → 下一个 token 就停
                raise _Abort()  # 顺便把生成循环也打断，别白烧 GPU
        #endregion

        #region 4.6.4 跑生成 + 收尾
        emit({"type": "start"})  # 先告诉前端"开始了"，它可以在这里把光标打开
        try:
            with GPU_LOCK:  # 串行：同一时刻只允许一个请求用 GPU
                run_generation(prompt, p, emit, aborted)
        except _Abort:
            pass  # 用户主动停，正常流程
        except Exception as e:        # 生成里炸了也要让前端知道，不然它一直转圈
            try:
                emit({"type": "err", "msg": f"{type(e).__name__}: {e}"})
            except _Abort:
                pass  # 前端已经断了，这条错误推不出去也无所谓
        try:
            self.wfile.write(b"data: [DONE]\n\n")  # 结束标记，前端收到后关流
            self.wfile.flush()
        except OSError:
            pass  # 连接已经断了，忽略
        #endregion
    #endregion
#endregion

#region 5. 工具：找一个空闲端口   【函数:free_port(起始端口,尝试次数),return int】
def free_port(start=8000, tries=40):
    """从 start 开始往上试，返回第一个能连上的"空"端口。

    connect_ex 返回 0 表示"有人占着"，非 0 表示没人监听、也就是可用。
    """
    for p in range(start, start + tries):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", p)) != 0:
                return p
    return start  # 全被占了就退回原端口，让绑定报错，而不是悄悄换到奇怪端口
#endregion

#region 6. 主流程：加载模型 + 起服务   【函数:main(),return None】
def main():
    #region 6.1 命令行参数   【集合:args】
    ap = argparse.ArgumentParser(description="miniGPT 对话台")
    ap.add_argument("--ckpt", default=os.path.join(HERE, "ckpt_novel", "best.pt"))
    ap.add_argument("--port", type=int, default=0, help="0 = 从 8000 起自动找空闲端口")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-browser", action="store_true")  # 脚本化启动时别弹浏览器
    args = ap.parse_args()
    #endregion

    #region 6.2 加载权重
    if not os.path.exists(args.ckpt):
        sys.exit(f"[x] 找不到权重 {args.ckpt}\n    先训练：python train.py --corpus data/corpus_novel.txt --min_freq 50 --minutes 70 --out ckpt_novel")

    t0 = time.time()
    print(f"[*] 加载权重 {os.path.basename(args.ckpt)} → {args.device} ...")
    model, tok, extra = load_ckpt(args.ckpt, args.device)
    #endregion

    #region 6.3 顺手把训练时的最佳 PPL 捞出来
    # 优先用 ckpt 里存的；没有再退回去读同级目录的 metrics.jsonl
    ppl = extra.get("val_ppl")
    mp = os.path.join(os.path.dirname(os.path.abspath(args.ckpt)), "metrics.jsonl")
    if ppl is None and os.path.exists(mp):
        try:
            rows = [json.loads(l) for l in open(mp, encoding="utf-8") if l.strip()]
            if rows:
                ppl = round(min(r["val_ppl"] for r in rows), 1)  # 取历史最小值 = 最佳 PPL
        except Exception:
            pass  # 指标文件坏了不该影响服务启动，吞掉异常
    #endregion

    #region 6.4 写入全局状态
    STATE.update({
        "model": model, "tok": tok, "device": args.device,
        "device_name": (torch.cuda.get_device_name(0) if args.device == "cuda" else "CPU"),
        "ckpt_name": os.path.basename(args.ckpt),
        "val_ppl": ppl,
    })
    #endregion

    #region 6.5 起 HTTP 服务
    port = args.port or free_port()  # 没指定端口就自动找
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)  # 只绑本地回环，外网访问不到
    srv.daemon_threads = True  # 主线程退出时子线程跟着走，Ctrl+C 能立刻生效
    url = f"http://127.0.0.1:{port}"

    print(f"[+] 模型 {model.num_params()/1e6:.2f}M 参数 | {model.cfg.n_layer} 层 "
          f"d_model={model.cfg.d_model} | 上下文 {model.cfg.block_size} 字符 | "
          f"词表 {model.cfg.vocab_size}  | 加载耗时 {time.time()-t0:.1f}s")
    print(f"[+] 打开 {url}\n    按 Ctrl+C 退出\n")

    if not args.no_browser:
        # 延迟 0.8 秒再开浏览器：给 serve_forever 一点时间把端口监听起来
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()  # 阻塞在这里，直到 Ctrl+C
    except KeyboardInterrupt:
        print("\n[*] 拜拜")
        srv.shutdown()
    #endregion
#endregion


#region 7. 入口
if __name__ == "__main__":
    main()
#endregion
