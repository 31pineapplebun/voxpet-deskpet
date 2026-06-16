"""
桌宠 web 版 - FastAPI 后端 (v4: 多好友版)
==========================================
v4 vs v3:
  - 读 friends.py 花名册, 支持多个好友
  - 网页可下拉切换好友, 音色 + 风格 + 向量库 一起换
  - 每个好友独立向量库 friend_db/<key>/
  - 切好友时动态切换 SoVITS 模型 (按需加载, 约几秒)

启动前提:
  - 先跑 start_sovits_api.py 启动 SoVITS API (本地音色时)
  - 设好 OPENAI_API_KEY
"""
import asyncio
import io
import os
import re
import sys
import uuid
from pathlib import Path

import numpy as np
import requests
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import chromadb
from chromadb.utils import embedding_functions
from faster_whisper import WhisperModel
from openai import OpenAI
from pydub import AudioSegment

from friends import (
    FRIENDS, get_friend, default_key, list_friends, build_system_prompt,
)

# ============ 配置 ============
RAG_DB_ROOT    = "./friend_db"        # 每人的库在 friend_db/<key>/
RAG_COLLECTION = "friend_chat"
EMBED_MODEL    = "paraphrase-multilingual-MiniLM-L12-v2"
RAG_TOP_K      = 10

LLM_MODEL       = "gpt-4o"
LLM_MAX_TOKENS  = 80
LLM_TEMPERATURE = 1.0

USE_LOCAL_TTS  = True                 # True=本地SoVITS音色, False=OpenAI

OPENAI_TTS_MODEL = "gpt-4o-mini-tts"
OPENAI_TTS_VOICE = "coral"
OPENAI_TTS_INSTR = (
    "声音特征:年轻女孩的声音,音调偏高、明亮、有活力。"
    "语气:活泼、亲切。语速:稍快,有自然的停顿和情绪起伏。"
)

SOVITS_API      = "http://127.0.0.1:9880"
SOVITS_REF_LANG = "zh"

# ============ 初始化 ============
if not os.getenv("OPENAI_API_KEY"):
    print("❌ 请先设置 OPENAI_API_KEY")
    sys.exit(1)

AUDIO_CACHE = Path("./audio_cache")
AUDIO_CACHE.mkdir(exist_ok=True)

print("加载 Whisper 模型 ...")
whisper = WhisperModel("small", device="cpu", compute_type="int8")

print("加载 embedding 模型 ...")
embedder = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)

# 为每个好友打开各自的向量库 (缺库的人跳过, 不影响别人)
collections = {}
for key in FRIENDS:
    db_path = Path(RAG_DB_ROOT) / key
    if not db_path.exists():
        print(f"  ⚠️  {FRIENDS[key]['display_name']}({key}) 向量库不存在, 跳过")
        print(f"      跑 python build_rag.py {key} 建库")
        continue
    try:
        client = chromadb.PersistentClient(path=str(db_path))
        col = client.get_collection(RAG_COLLECTION, embedding_function=embedder)
        collections[key] = col
        print(f"  ✓ {FRIENDS[key]['display_name']}({key}): {col.count()} 条")
    except Exception as e:
        print(f"  ⚠️  {key} 库打开失败: {e}")

if not collections:
    print("❌ 没有任何可用的向量库, 先跑 python build_rag.py")
    sys.exit(1)

# 当前选中的好友 (默认第一个有库的)
CURRENT_KEY = default_key() if default_key() in collections else next(iter(collections))
print(f"\n默认好友: {get_friend(CURRENT_KEY)['display_name']}")

llm = OpenAI()

# SoVITS 当前已加载的模型 (避免重复切换)
_sovits_loaded_key = None


# ============ SoVITS 模型切换 ============
def switch_sovits_model(key: str):
    """切到某个好友的 SoVITS+GPT 模型 (按需加载, 已是当前模型则跳过)"""
    global _sovits_loaded_key
    if not USE_LOCAL_TTS:
        return
    if _sovits_loaded_key == key:
        return
    f = get_friend(key)
    from friends import GPT_WEIGHTS_DIR, SOVITS_WEIGHTS_DIR
    gpt_path    = GPT_WEIGHTS_DIR / f["gpt_model"]
    sovits_path = SOVITS_WEIGHTS_DIR / f["sovits_model"]

    print(f"切换 SoVITS 模型 -> {f['display_name']} ...")
    r1 = requests.get(f"{SOVITS_API}/set_gpt_weights",
                      params={"weights_path": str(gpt_path)}, timeout=60)
    r2 = requests.get(f"{SOVITS_API}/set_sovits_weights",
                      params={"weights_path": str(sovits_path)}, timeout=60)
    if r1.status_code != 200 or r2.status_code != 200:
        raise RuntimeError(
            f"切换模型失败: GPT {r1.status_code}({r1.text[:100]}), "
            f"SoVITS {r2.status_code}({r2.text[:100]})"
        )
    _sovits_loaded_key = key
    print(f"  ✓ 已加载 {f['display_name']} 的音色")


# ============ 业务函数 ============
def stt_from_webm(webm_bytes: bytes) -> str:
    audio = AudioSegment.from_file(io.BytesIO(webm_bytes))
    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    pcm = np.frombuffer(audio.raw_data, dtype=np.int16).astype(np.float32) / 32768.0
    segments, _ = whisper.transcribe(pcm, language="zh", beam_size=1)
    return "".join(s.text for s in segments).strip()


def retrieve(key: str, query: str):
    col = collections[key]
    res = col.query(query_texts=[query], n_results=RAG_TOP_K)
    docs = res["documents"][0] if res["documents"] else []
    metas = res["metadatas"][0] if res["metadatas"] else []
    return list(zip(docs, metas))


def parse_doc_to_messages(doc: str, friend_name: str):
    """doc: '其他人: xxx\\n好友名: xxx' -> [(role, content), ...]"""
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


def call_llm(key: str, user_text: str):
    f = get_friend(key)
    friend_name = f["display_name"]
    refs = retrieve(key, user_text)
    messages = [{"role": "system", "content": build_system_prompt(key)}]

    for doc, _ in refs:
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
        model=LLM_MODEL,
        messages=messages,
        max_tokens=LLM_MAX_TOKENS,
        temperature=LLM_TEMPERATURE,
    )
    return resp.choices[0].message.content.strip(), refs


def tts_openai(text: str) -> bytes:
    response = llm.audio.speech.create(
        model=OPENAI_TTS_MODEL,
        voice=OPENAI_TTS_VOICE,
        input=text,
        instructions=OPENAI_TTS_INSTR,
        response_format="mp3",
    )
    return response.content


def prep_tts_text(text: str) -> str:
    """文本预处理。主要靠 tts_sovits 里的采样参数解决 EOS 提前截断,
    这里只做最基本的: 去前导标点(避免开头空音) + 极短补长 + 末尾句号。"""
    # 1) 去前导标点 (开头必须汉字, 否则合成开头是空音)
    text = text.lstrip("，。！？、~,.!?…—-\"' \t\n（）()【】[]")
    core = re.sub(r"[，。！？、~,.!?\s]", "", text)
    n = len(core)
    if n == 0:
        return "你说啥。"
    text = text.strip()

    # 2) 只有极短(1-2字)才重复, 其余原样 (重复多了听着怪, 短句主要靠参数救)
    if n <= 2:
        body = text.rstrip("，。！？、~ ,.!?")
        body = f"{body}，{body}"
    else:
        body = text

    # 3) 末尾补句号
    if not body.endswith(("。", "！", "？", "~", ".", "!", "?")):
        body = body + "。"
    return body



def tts_sovits(key: str, text: str) -> bytes:
    switch_sovits_model(key)            # 确保当前是这个人的音色
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
        # 用 api 默认采样参数 (激进参数反而让短文本崩)
        # 不传 text_split_method, 用 api 默认。
    }
    r = requests.post(f"{SOVITS_API}/tts", json=payload, timeout=120)
    if r.status_code != 200:
        raise RuntimeError(f"SoVITS API {r.status_code}: {r.text[:300]}")
    return r.content


def synthesize(key: str, text: str):
    if USE_LOCAL_TTS:
        return tts_sovits(key, text), "wav"
    return tts_openai(text), "mp3"


def normalize_audio(audio_bytes: bytes, ext: str, target_dBFS: float = -16.0) -> bytes:
    """音量归一化: GPT-SoVITS 短文本偶尔输出音量极小, 统一拉到正常响度。
    target_dBFS 越大越响 (-16 约等于正常语音, -10 更响, 0 是最大但可能爆音)。"""
    try:
        seg = AudioSegment.from_file(io.BytesIO(audio_bytes))
        if seg.dBFS == float("-inf"):   # 纯静音, 没法归一化
            return audio_bytes
        gain = target_dBFS - seg.dBFS
        # 限制增益范围, 避免把噪音放大到爆 / 或削得太狠
        gain = max(-6.0, min(gain, 30.0))
        seg = seg.apply_gain(gain)
        buf = io.BytesIO()
        seg.export(buf, format="wav" if ext == "wav" else ext)
        return buf.getvalue()
    except Exception as e:
        print(f"  ⚠️ 归一化失败, 用原音频: {e}")
        return audio_bytes


def save_audio_checked(audio_bytes: bytes, ext: str) -> tuple:
    """写音频文件, 归一化音量 + 解码检查有效性。
    返回 (文件名, 警告信息或None)"""
    warn = None
    # 先归一化音量 (解决短文本偶尔音量极小听不见的问题)
    audio_bytes = normalize_audio(audio_bytes, ext)
    size = len(audio_bytes)
    fname = f"{uuid.uuid4().hex}.{ext}"
    path = AUDIO_CACHE / fname
    path.write_bytes(audio_bytes)
    try:
        with open(path, "rb") as fp:
            os.fsync(fp.fileno())
    except Exception:
        pass

    # 真正解码, 看时长和峰值 (光看字节数不够, 坏WAV也可能字节很多)
    try:
        seg = AudioSegment.from_file(io.BytesIO(audio_bytes))
        dur_ms = len(seg)
        peak = seg.max          # 峰值振幅, 接近0说明是静音
        rms = seg.rms           # 响度, 0=纯静音
        print(f"  🎵 音频: {size}字节 时长{dur_ms}ms 峰值{peak} rms{rms}")
        if dur_ms < 200:
            warn = f"音频过短 {dur_ms}ms"
        elif rms < 50:
            warn = f"音频近乎静音 rms={rms}"
        if warn:
            # 坏音频存档, 供事后分析
            bad_dir = AUDIO_CACHE / "_bad"
            bad_dir.mkdir(exist_ok=True)
            (bad_dir / fname).write_bytes(audio_bytes)
            print(f"  ⚠️  {warn} -> 已存档到 audio_cache/_bad/{fname}")
    except Exception as e:
        warn = f"音频解码失败: {e}"
        print(f"  ⚠️  {warn} ({size}字节)")
        bad_dir = AUDIO_CACHE / "_bad"
        bad_dir.mkdir(exist_ok=True)
        (bad_dir / fname).write_bytes(audio_bytes)

    return fname, warn


# ============ FastAPI ============
app = FastAPI()
app.mount("/audio", StaticFiles(directory=str(AUDIO_CACHE)), name="audio")


@app.get("/", response_class=HTMLResponse)
async def index():
    return Path("index.html").read_text(encoding="utf-8")


@app.get("/api/info")
async def info():
    f = get_friend(CURRENT_KEY)
    return {
        "friend_name": f["display_name"],
        "friend_key": CURRENT_KEY,
        "rag_count": collections[CURRENT_KEY].count(),
        "use_local_tts": USE_LOCAL_TTS,
        "llm_model": LLM_MODEL,
        # 给下拉菜单: 只列出有库的好友
        "friends": [x for x in list_friends() if x["key"] in collections],
    }


@app.post("/api/switch")
async def switch_friend(friend: str = Form(...)):
    """切换当前好友"""
    global CURRENT_KEY
    if friend not in collections:
        return JSONResponse({"error": f"好友 {friend} 不可用"}, status_code=400)
    CURRENT_KEY = friend
    f = get_friend(friend)
    # 本地音色: 提前切好模型 (这样切换的等待发生在点选时, 不是第一句话时)
    if USE_LOCAL_TTS:
        try:
            await asyncio.to_thread(switch_sovits_model, friend)
        except Exception as e:
            return JSONResponse({"error": f"切换音色失败: {e}"}, status_code=500)
    return {
        "friend_name": f["display_name"],
        "friend_key": friend,
        "rag_count": collections[friend].count(),
    }


@app.post("/api/chat")
async def chat(text: str = Form(...), friend: str = Form(None)):
    key = friend if (friend and friend in collections) else CURRENT_KEY
    text = text.strip()
    if not text:
        return JSONResponse({"error": "空消息"}, status_code=400)
    try:
        reply, refs = await asyncio.to_thread(call_llm, key, text)
        audio_bytes, ext = await asyncio.to_thread(synthesize, key, reply)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    fname, warn = save_audio_checked(audio_bytes, ext)
    return {
        "user_text": text,
        "reply": reply,
        "audio_url": f"/audio/{fname}",
        "audio_warn": warn,
        "refs": [{"text": d, "source": m.get("source", "")} for d, m in refs],
    }
    return {
        "user_text": text,
        "reply": reply,
        "audio_url": f"/audio/{fname}",
        "refs": [{"text": d, "source": m.get("source", "")} for d, m in refs],
    }


@app.post("/api/voice")
async def voice(audio: UploadFile = File(...), friend: str = Form(None)):
    key = friend if (friend and friend in collections) else CURRENT_KEY
    audio_bytes = await audio.read()
    try:
        user_text = await asyncio.to_thread(stt_from_webm, audio_bytes)
    except Exception as e:
        return JSONResponse({"error": f"STT 失败: {e}"}, status_code=500)
    if not user_text:
        return JSONResponse({"error": "没识别到内容"}, status_code=400)
    try:
        reply, refs = await asyncio.to_thread(call_llm, key, user_text)
        out_bytes, ext = await asyncio.to_thread(synthesize, key, reply)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    fname, warn = save_audio_checked(out_bytes, ext)
    return {
        "user_text": user_text,
        "reply": reply,
        "audio_url": f"/audio/{fname}",
        "audio_warn": warn,
        "refs": [{"text": d, "source": m.get("source", "")} for d, m in refs],
    }


if __name__ == "__main__":
    import uvicorn
    print("🌐 启动: http://localhost:8000\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)