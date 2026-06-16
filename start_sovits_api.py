"""
GPT-SoVITS 推理 API 启动脚本 (对应双层套娃 + v2Pro 整合包)
=========================================================
功能: 启动 api_v2.py + 自动加载你训好的 GPT/SoVITS 模型
用法: python start_sovits_api.py

启动后 API 监听 http://127.0.0.1:9880, app.py 直接调即可。
Ctrl+C 关闭。
"""

import subprocess
import sys
import time
from pathlib import Path

import requests   # pip install requests

# ====================== 配置区 ======================
# GPT-SoVITS 项目根目录 (最内层那个 GPT-SoVITS-v2pro-...)
GPT_SOVITS_ROOT = Path(r"D:\GPT-SoVITS\GPT-SoVITS-v2pro-20250604\GPT-SoVITS-v2pro-20250604")

# 训练完的模型路径 (老王 buddy, 用 e15 训练最充分的那个)
GPT_MODEL_PATH    = GPT_SOVITS_ROOT / "GPT_weights_v2Pro"    / "buddy-e15.ckpt"
SOVITS_MODEL_PATH = GPT_SOVITS_ROOT / "SoVITS_weights_v2Pro" / "buddy_e8_s392.pth"

API_PORT = 9880
API_URL  = f"http://127.0.0.1:{API_PORT}"


# ====================== 主流程 ======================
def main():
    # 1) 路径检查
    if not GPT_SOVITS_ROOT.exists():
        print(f"❌ GPT-SoVITS 目录不存在: {GPT_SOVITS_ROOT}")
        sys.exit(1)

    api_script = GPT_SOVITS_ROOT / "api_v2.py"
    if not api_script.exists():
        print(f"❌ 没找到 api_v2.py: {api_script}")
        print(f"   GPT_SOVITS_ROOT 路径不对, 改顶部那一行")
        sys.exit(1)

    for label, path in [("GPT 模型", GPT_MODEL_PATH), ("SoVITS 模型", SOVITS_MODEL_PATH)]:
        if not path.exists():
            print(f"❌ {label}不存在: {path}")
            print(f"   去 GPT_weights_v2Pro / SoVITS_weights_v2Pro 看实际文件名, 改顶部路径")
            # 顺便列出实际有哪些, 方便改
            if "GPT" in label:
                d = GPT_SOVITS_ROOT / "GPT_weights_v2Pro"
            else:
                d = GPT_SOVITS_ROOT / "SoVITS_weights_v2Pro"
            if d.exists():
                print(f"   {d} 下的 buddy 模型:")
                for f in sorted(d.glob("buddy*")):
                    print(f"      {f.name}")
            sys.exit(1)

    # 2) 选 Python: 整合包用 runtime/python.exe
    python_exe = GPT_SOVITS_ROOT / "runtime" / "python.exe"
    if not python_exe.exists():
        python_exe = Path(sys.executable)

    # 3) 启动 api_v2.py
    print(f"启动 GPT-SoVITS API ...")
    print(f"  Python : {python_exe}")
    print(f"  cwd    : {GPT_SOVITS_ROOT}")
    print(f"  GPT    : {GPT_MODEL_PATH.name}")
    print(f"  SoVITS : {SOVITS_MODEL_PATH.name}")
    proc = subprocess.Popen(
        [str(python_exe), "api_v2.py", "-p", str(API_PORT)],
        cwd=str(GPT_SOVITS_ROOT),
    )

    # 4) 等 API 就绪 (api_v2 首次启动加载模型要 30~60 秒)
    print(f"\n等 API 启动 (最多 90 秒) ...")
    ready = False
    for i in range(90):
        if proc.poll() is not None:
            print(f"\n❌ API 进程退出了, 返回码 {proc.returncode}")
            print(f"   去 {GPT_SOVITS_ROOT} 手动跑 runtime\\python.exe api_v2.py 看完整报错")
            sys.exit(1)
        try:
            r = requests.get(f"{API_URL}/control?command=ping", timeout=1)
            ready = True
            break
        except requests.RequestException:
            pass
        time.sleep(1)
        if (i + 1) % 5 == 0:
            print(f"   仍在等待... ({i+1}s)")

    if not ready:
        print(f"\n❌ API 超时未就绪")
        proc.terminate()
        sys.exit(1)
    print(f"✅ API 已就绪\n")

    # 5) 加载训好的模型 (api_v2 这两个端点用 GET, 不是 POST!)
    print(f"加载 GPT 模型 ...")
    r = requests.get(
        f"{API_URL}/set_gpt_weights",
        params={"weights_path": str(GPT_MODEL_PATH)},
    )
    print(f"   {r.status_code}  {r.text[:200]}")
    if r.status_code != 200:
        print(f"   ⚠️ GPT 模型加载失败! 当前用的还是默认底模, 声音不是克隆的")

    print(f"加载 SoVITS 模型 ...")
    r = requests.get(
        f"{API_URL}/set_sovits_weights",
        params={"weights_path": str(SOVITS_MODEL_PATH)},
    )
    print(f"   {r.status_code}  {r.text[:200]}")
    if r.status_code != 200:
        print(f"   ⚠️ SoVITS 模型加载失败! 当前用的还是默认底模, 声音不是克隆的")
        print(f"      如果报维度/版本错, 可能是 v2Pro 模型在 v2 模式 api 下不兼容, 把报错发我")

    print(f"\n✅ 模型加载完毕, API 运行中: {API_URL}")
    print(f"   app.py 把 USE_LOCAL_TTS 改 True 就能用了")
    print(f"   按 Ctrl+C 关闭 API\n")

    # 6) 主线程等 api_v2 进程
    try:
        proc.wait()
    except KeyboardInterrupt:
        print("\n关闭 API ...")
        proc.terminate()
        proc.wait(timeout=10)
        print("✅ 已关闭")


if __name__ == "__main__":
    main()
