#region 0. 导入库
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen
#endregion

#region 1. 常量   【常量:FILES 需下载的 7 个文件】
MODEL_DIR = Path(__file__).resolve().parent / "models" / "Qwen3-0.6B-Base"       # 模型落地目录
BASE_URL = "https://modelscope.cn/models/Qwen/Qwen3-0.6B-Base/resolve/master/"   # 静态直链前缀
FILES = ["config.json", "generation_config.json", "merges.txt", "model.safetensors",
         "tokenizer.json", "tokenizer_config.json", "vocab.json"]                 # 7 个文件，缺一不可
EXPECT = {"model.safetensors": 1_192_135_096}                                    # 大文件字节数，判断是否完整
#endregion

#region 2. 函数：断点续传下载单个文件   【函数:fetch(name),return 是否成功】
def fetch(name):
    """下载单个文件：已完整则跳过，残缺则用 Range 从上次位置续传，最多重试 3 次。返回：bool。"""
    dst = MODEL_DIR / name
    for attempt in range(1, 4):
        have = dst.stat().st_size if dst.exists() else 0            # 本地已有字节数
        if EXPECT.get(name) == have:                                # 大小已对上，直接跳过（幂等）
            print(f"[skip] {name} {have / 1e6:.1f}MB")
            return True
        headers = {"User-Agent": "Mozilla/5.0"}                     # 部分 CDN 拒绝空 UA
        if have:
            headers["Range"] = f"bytes={have}-"                     # 告诉服务器：从 have 之后开始发
        try:
            t0 = time.time()
            with urlopen(Request(BASE_URL + name, headers=headers), timeout=60) as r, \
                    open(dst, "ab" if have else "wb") as f:         # 续传用 append，首下用 wb
                total = have + int(r.headers.get("Content-Length") or 0)   # 目标总字节数
                done, last = have, 0.0
                while chunk := r.read(1 << 20):                     # 1MB/块，边下边写，不占内存
                    f.write(chunk)
                    done += len(chunk)
                    if time.time() - last > 1.0:                    # 每秒刷一次进度，别刷屏
                        last = time.time()
                        spd = (done - have) / max(1e-6, last - t0) / 1e6
                        print(f"\r  {name} {done / 1e6:.1f}/{total / 1e6:.1f}MB {spd:.1f}MB/s", end="")
            print()
            if EXPECT.get(name) in (None, dst.stat().st_size):       # 小文件没有期望值，下完即算成功
                return True
            print(f"[warn] {name} 大小不符，重试")
        except (URLError, TimeoutError, OSError) as e:
            print(f"\n[retry {attempt}/3] {name}: {e}")
            time.sleep(2)                                           # 网络抖动，等一下再来
    return False
#endregion

#region 3. 主流程：依次拉 7 个文件并汇总
def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"目标目录: {MODEL_DIR}")
    t0 = time.time()
    for name in FILES:
        if not fetch(name):
            print(f'\n[FAIL] {name}\n手工兜底: curl -L -C - --retry 5 -o "{MODEL_DIR / name}" "{BASE_URL + name}"')
            return 1
    total = sum((MODEL_DIR / f).stat().st_size for f in FILES)       # 全部文件总大小
    print(f"\n完成，共 {total / 1e6:.1f}MB，耗时 {time.time() - t0:.1f}s")
    for f in FILES:
        print(f"  {f:24s}{(MODEL_DIR / f).stat().st_size:>14,d} bytes")
    return 0
#endregion

if __name__ == "__main__":
    sys.exit(main())
