"""
微信语音文件归类脚本 - 扫描所有 CSV,按 talker 把对应的 WAV 复制到不同文件夹
用法: 改下面 4 个配置, python sort_voices.py
"""
import pandas as pd
from pathlib import Path
import re
import shutil
from collections import defaultdict

# ============ 配置 ============
CSV_DIR      = "./chats"                       # CSV 文件夹(和 build_rag 用同一个)
VOICE_DIR    = r"D:\微信聊天记录\voices"        # WAV 原文件夹
OUTPUT_DIR   = "./by_speaker"                  # 归类后输出到这里
INCLUDE_SELF = False                            # True = 也归类你自己的语音(is_sender=1)
COPY_OR_MOVE = "copy"                          # "copy" 复制(原文件保留), "move" 移动(原文件消失)

# ============ 工具函数 ============
INVALID_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')

def safe_folder_name(name: str) -> str:
    """清掉 Windows 文件夹名不允许的字符"""
    name = INVALID_CHARS.sub("_", str(name).strip())
    return name[:50] or "unknown"

def extract_wav_filename(msg: str) -> str | None:
    """从 msg 列字符串里抽出 WAV 文件名, 比如 voice_xxx.wav"""
    m = re.search(r'voice_[\w-]+\.wav', str(msg))
    return m.group(0) if m else None

# ============ 扫描 ============
voice_dir = Path(VOICE_DIR)
if not voice_dir.exists():
    print(f"❌ 语音文件夹不存在: {VOICE_DIR}")
    exit(1)

csv_files = sorted(Path(CSV_DIR).glob("*.csv"))
if not csv_files:
    print(f"❌ {CSV_DIR} 里没有 CSV")
    exit(1)

print(f"扫描 {len(csv_files)} 个 CSV ...\n")

# 收集 (talker, wav_filename) 对
records = []
for csv_file in csv_files:
    try:
        df = pd.read_csv(csv_file, encoding="utf-8")
    except Exception as e:
        print(f"  ⚠️  {csv_file.name} 读取失败: {e}")
        continue

    voice_df = df[df["type_name"] == "voice"].copy()
    if not INCLUDE_SELF:
        voice_df = voice_df[voice_df["is_sender"] == 0]

    n = 0
    for _, row in voice_df.iterrows():
        wav = extract_wav_filename(row["msg"])
        if wav:
            records.append((row["talker"], wav))
            n += 1
    print(f"  {csv_file.name}: 找到 {n} 条语音")

# ============ 按人分组 ============
by_speaker = defaultdict(set)   # 用 set 自动去重
for talker, wav in records:
    by_speaker[talker].add(wav)

total = sum(len(v) for v in by_speaker.values())
print(f"\n汇总: {total} 条语音 (已去重), {len(by_speaker)} 个发言者:")
print(f"  {'条数':>6}   昵称")
print(f"  {'-'*6}   {'-'*30}")
for talker, wavs in sorted(by_speaker.items(), key=lambda x: -len(x[1])):
    print(f"  {len(wavs):>6}   {talker}")

# ============ 复制/移动文件 ============
output_dir = Path(OUTPUT_DIR)
output_dir.mkdir(exist_ok=True)

action_name = "复制" if COPY_OR_MOVE == "copy" else "移动"
print(f"\n开始{action_name}文件 -> {output_dir.absolute()}\n")

stats = defaultdict(lambda: {"ok": 0, "missing": 0, "exists": 0})

for talker, wavs in by_speaker.items():
    target_dir = output_dir / safe_folder_name(talker)
    target_dir.mkdir(exist_ok=True)

    for wav in wavs:
        src = voice_dir / wav
        dst = target_dir / wav

        if dst.exists():
            stats[talker]["exists"] += 1
            continue
        if not src.exists():
            stats[talker]["missing"] += 1
            continue

        try:
            if COPY_OR_MOVE == "copy":
                shutil.copy2(src, dst)
            else:
                shutil.move(str(src), str(dst))
            stats[talker]["ok"] += 1
        except Exception as e:
            print(f"  ⚠️  {wav} 失败: {e}")

print(f"完成! 各发言者{action_name}结果:")
print(f"  {'新增':>6} {'已存在':>6} {'源缺失':>6}   昵称")
print(f"  {'-'*6} {'-'*6} {'-'*6}   {'-'*30}")
for talker in sorted(stats.keys(), key=lambda t: -stats[t]["ok"]):
    s = stats[talker]
    print(f"  {s['ok']:>6} {s['exists']:>6} {s['missing']:>6}   {talker}")

print(f"\n✅ 全部完成, 输出在: {output_dir.absolute()}")