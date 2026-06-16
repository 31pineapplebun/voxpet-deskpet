"""
桌宠硬件端 - PC WebSocket 服务器 (本地 SoVITS 音色版)
=====================================================
和网页版 app.py 共用同一套逻辑 (friends 花名册 / RAG / LLM / SoVITS),
区别是用 WebSocket 和 ESP32 通信, 且把 SoVITS 的 32k 音频重采样到 16k 发给 ESP32。

协议 (和之前打通的一致):
  ESP32 -> PC:  "REC_START" -> [PCM 16k 二进制流] -> "REC_END"
  PC -> ESP32:  "TTS_START" -> [PCM 16k 二进制流] -> "TTS_END"

启动前提:
  1) 先启动 SoVITS API:  python start_sovits_api.py
  2) 设好环境变量:       set OPENAI_API_KEY=你的key
  3) PC 开热点, ESP32 连热点 (WS_HOST=192.168.137.1)
  4) python desk_pet_server.py

依赖: pip install websockets numpy faster-whisper openai chromadb pydub sentence-transformers
"""
import asyncio
import io
import os
import re
import sys
from pathlib import Path

import numpy as np
import requests
import websockets

import chromadb
from chromadb.utils import embedding_functions
from faster_whisper import WhisperModel
from openai import OpenAI
from pydub import AudioSegment

# 复用网页版的花名册和 prompt 逻辑
from friends import (
    FRIENDS, get_friend, default_key, build_system_prompt,
    GPT_WEIGHTS_DIR, SOVITS_WEIGHTS_DIR,
)

# ============ 配置 ============
WS_HOST = "0.0.0.0"          # 监听所有网卡 (ESP32 通过热点 192.168.137.1 连进来)
WS_PORT = 8765

MIC_RATE        = 16000      # ESP32 麦克风采样率 (上传的 PCM)
ESP32_SPK_RATE  = 16000      # ESP32 喇叭采样率 (要发给它的 PCM, SoVITS 32k 重采样到这)

RAG_DB_ROOT    = "./friend_db"
RAG_COLLECTION = "friend_chat"
EMBED_MODEL    = "paraphrase-multilingual-MiniLM-L12-v2"
RAG_TOP_K      = 10

LLM_MODEL       = "gpt-4o"
LLM_MAX_TOKENS  = 80
LLM_TEMPERATURE = 1.0

SOVITS_API      = "http://127.0.0.1:9880"
SOVITS_REF_LANG = "zh"

# 当前用哪个好友 (硬件端默认用花名册第一个; 想换人改这里, 或扩展按键切换)
CURRENT_KEY = default_key()

# ============ 初始化 ============
if not os.getenv("OPENAI_API_KEY"):
    print("❌ 请先设置 OPENAI_API_KEY")
    sys.exit(1)

print("加载 Whisper ...")
whisper = WhisperModel("small", device="cpu", compute_type="int8")

print("加载 embedding ...")
embedder = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)

# 打开各好友的向量库
collections = {}
for key in FRIENDS:
    db_path = Path(RAG_DB_ROOT) / key
    if not db_path.exists():
        continue
    try:
        client = chromadb.PersistentClient(path=str(db_path))
        collections[key] = client.get_collection(RAG_COLLECTION, embedding_function=embedder)
        print(f"  ✓ {FRIENDS[key]['display_name']}({key}): {collections[key].count()} 条")
    except Exception as e:
        print(f"  ⚠️ {key} 库打开失败: {e}")

if CURRENT_KEY not in collections:
    CURRENT_KEY = next(iter(collections)) if collections else None
if not CURRENT_KEY:
    print("❌ 没有可用向量库, 先跑 build_rag.py")
    sys.exit(1)
print(f"当前好友: {get_friend(CURRENT_KEY)['display_name']}\n")

llm = OpenAI()
_sovits_loaded_key = None


# ============ SoVITS 模型切换 ============
def switch_sovits_model(key):
    global _sovits_loaded_key
    if _sovits_loaded_key == key:
        return
    f = get_friend(key)
    gpt_path    = GPT_WEIGHTS_DIR / f["gpt_model"]
    sovits_path = SOVITS_WEIGHTS_DIR / f["sovits_model"]
    print(f"切换音色 -> {f['display_name']} ...")
    r1 = requests.get(f"{SOVITS_API}/set_gpt_weights", params={"weights_path": str(gpt_path)}, timeout=60)
    r2 = requests.get(f"{SOVITS_API}/set_sovits_weights", params={"weights_path": str(sovits_path)}, timeout=60)
    if r1.status_code != 200 or r2.status_code != 200:
        raise RuntimeError(f"切换模型失败 GPT={r1.status_code} SoVITS={r2.status_code}")
    _sovits_loaded_key = key
    print(f"  ✓ 已加载 {f['display_name']} 音色")


# ============ STT ============
def stt(pcm16k: bytes) -> str:
    audio = np.frombuffer(pcm16k, dtype=np.int16).astype(np.float32) / 32768.0
    segments, _ = whisper.transcribe(audio, language="zh", beam_size=1)
    return "".join(s.text for s in segments).strip()


# ============ RAG + LLM (复用 app.py 逻辑) ============
def retrieve(key, query):
    res = collections[key].query(query_texts=[query], n_results=RAG_TOP_K)
    docs = res["documents"][0] if res["documents"] else []
    return docs


def parse_doc_to_messages(doc, friend_name):
    msgs = []
    for line in doc.strip().split("\n"):
        m = re.match(r"^([^:：]+)[:：]\s*(.+)$", line)
        if not m:
            continue
        speaker, content = m.group(1).strip(), m.group(2).strip()
        if not content:
            continue
        role = "assistant" if speaker == friend_name else "user"
        msgs.append((role, content))
    return msgs


def call_llm(key, user_text):
    f = get_friend(key)
    friend_name = f["display_name"]
    docs = retrieve(key, user_text)
    messages = [{"role": "system", "content": build_system_prompt(key)}]
    for doc in docs:
        pairs = parse_doc_to_messages(doc, friend_name)
        if not pairs:
            continue
        merged = []
        for role, content in pairs:
            if merged and merged[-1][0] == role:
                merged[-1] = (role, merged[-1][1] + "\n" + content)
            else:
                merged.append((role, content))
        for role, content in merged:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user_text})
    resp = llm.chat.completions.create(
        model=LLM_MODEL, messages=messages,
        max_tokens=LLM_MAX_TOKENS, temperature=LLM_TEMPERATURE,
    )
    return resp.choices[0].message.content.strip()


# ============ 文本预处理 (复用 app.py 的 prep_tts_text) ============
def prep_tts_text(text):
    text = text.lstrip("，。！？、~,.!?…—-\"' \t\n（）()【】[]")
    core = re.sub(r"[，。！？、~,.!?\s]", "", text)
    n = len(core)
    if n == 0:
        return "你说啥。"
    text = text.strip()
    if n <= 2:
        body = text.rstrip("，。！？、~ ,.!?")
        body = f"{body}，{body}"
    else:
        body = text
    if not body.endswith(("。", "！", "？", "~", ".", "!", "?")):
        body = body + "。"
    return body


# ============ TTS + 重采样到 16k ============
def normalize_audio(seg, target_dBFS=-16.0):
    if seg.dBFS == float("-inf"):
        return seg
    gain = max(-6.0, min(target_dBFS - seg.dBFS, 30.0))
    return seg.apply_gain(gain)


def tts_to_pcm16k_sync(key, text) -> bytes:
    """SoVITS 合成 -> 归一化 -> 重采样到 16k 单声道 -> 返回裸 PCM (int16)"""
    switch_sovits_model(key)
    f = get_friend(key)
    synth_text = prep_tts_text(text)
    payload = {
        "text": synth_text,
        "text_lang": "zh",
        "ref_audio_path": f["ref_audio"],
        "prompt_text": f["ref_text"],
        "prompt_lang": SOVITS_REF_LANG,
        "media_type": "wav",
        "streaming_mode": False,
    }
    r = requests.post(f"{SOVITS_API}/tts", json=payload, timeout=120)
    if r.status_code != 200:
        raise RuntimeError(f"SoVITS API {r.status_code}: {r.text[:200]}")

    # 解码 SoVITS 返回的 wav (它是 32k), 归一化, 重采样到 16k 单声道 16bit
    seg = AudioSegment.from_file(io.BytesIO(r.content))
    seg = normalize_audio(seg)
    seg = seg.set_frame_rate(ESP32_SPK_RATE).set_channels(1).set_sample_width(2)
    return seg.raw_data   # 裸 PCM int16 @ 16k


async def tts_to_pcm(key, text):
    return await asyncio.to_thread(tts_to_pcm16k_sync, key, text)


# ============ 单轮对话 ============
async def process_turn(ws, pcm):
    # 录音太短跳过 (< 0.3 秒)
    if len(pcm) < MIC_RATE * 2 * 0.3:
        print("⚠️  录音太短, 跳过\n")
        return

    print(f"⏹  收到 {len(pcm)/2/MIC_RATE:.1f}s 音频")

    user_text = await asyncio.to_thread(stt, pcm)
    if not user_text:
        print("⚠️  没识别到内容\n")
        return
    print(f"👤 用户: {user_text}")

    try:
        reply = await asyncio.to_thread(call_llm, CURRENT_KEY, user_text)
    except Exception as e:
        print(f"❌ LLM 出错: {e}")
        reply = "等下啊"
    print(f"🤖 {get_friend(CURRENT_KEY)['display_name']}: {reply}")

    try:
        out_pcm = await tts_to_pcm(CURRENT_KEY, reply)
    except Exception as e:
        print(f"❌ TTS 出错: {e}")
        return

    print(f"📤 发送 {len(out_pcm)/2/ESP32_SPK_RATE:.1f}s 音频 @ {ESP32_SPK_RATE}Hz")
    await ws.send("TTS_START")
    for i in range(0, len(out_pcm), 1024):
        await ws.send(out_pcm[i:i+1024])
        await asyncio.sleep(0.005)   # 给 ESP32 喘息, 别冲垮它的缓冲
    await ws.send("TTS_END")
    print("✅ 完成\n")


# ============ WebSocket 处理 ============
async def handler(ws):
    print(f"🔌 ESP32 连接: {ws.remote_address}")
    buf = bytearray()
    try:
        async for msg in ws:
            if isinstance(msg, bytes):
                buf.extend(msg)
            else:
                cmd = msg.strip()
                if cmd == "REC_START":
                    buf = bytearray()
                    print("🎤 接收语音 ...")
                elif cmd == "REC_END":
                    await process_turn(ws, bytes(buf))
                    buf = bytearray()
    except websockets.ConnectionClosed:
        print("🔌 ESP32 断开\n")
    except Exception as e:
        print(f"❌ handler 异常: {e}\n")


async def main():
    # 预加载当前好友的 SoVITS 模型 (避免第一句话等很久)
    try:
        await asyncio.to_thread(switch_sovits_model, CURRENT_KEY)
    except Exception as e:
        print(f"⚠️ 预加载音色失败(SoVITS API 没开?): {e}")
        print(f"   确认先跑了 start_sovits_api.py")

    print(f"\n🌐 WebSocket 服务器启动: ws://{WS_HOST}:{WS_PORT}")
    print(f"   ESP32 端 WS_HOST 填 192.168.137.1 (PC 热点网关)")
    print(f"   等待 ESP32 连接...\n")
    async with websockets.serve(handler, WS_HOST, WS_PORT, max_size=None):
        await asyncio.Future()   # 永久运行


if __name__ == "__main__":
    asyncio.run(main())