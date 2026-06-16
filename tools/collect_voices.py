"""
从微信导出语音里自动挑选某好友的训练素材 (读 CSV 的 src 列)
============================================================
原理 (终于搞对了):
  CSV 的 src 列 = 语音文件完整路径, 形如:
    ../voices/voice_47974879924_chatroom_19489_1700498094_2919296508501361765.wav
  从 src 提取文件名, 去 voices 文件夹找, 按时长筛选, 复制到 raw/<key>/

关键: 用 talker 列区分说话人 (群聊里多人, 靠 talker=老王 筛出他的)
      不能靠文件名 (群聊语音文件名都是同一个群号前缀, 不含说话人信息)

用法:
  python collect_voices.py                  默认收集老王
  python collect_voices.py 小鹿 sweetie   收集别人 (中文名 英文key)
"""
import sys
import shutil
import re
from pathlib import Path
from collections import Counter

import pandas as pd
from pydub import AudioSegment

# ====================== 配置 ======================
VOICES_DIR = r"D:\微信聊天记录\voices"        # 所有 wav 在这
CHATS_DIR  = r"D:\微信聊天记录\texts"          # 所有 CSV 在这
RAW_ROOT   = r"D:\GPT-SoVITS\raw"            # 训练素材根目录

# 默认目标 (中文昵称 = CSV里talker, 英文key = raw文件夹名)
TARGET_DISPLAY = "老王"
TARGET_KEY     = "buddy"

MIN_SEC = 3.0       # 太短跳过
MAX_SEC = 12.0      # 太长跳过

if len(sys.argv) >= 3:
    TARGET_DISPLAY = sys.argv[1]
    TARGET_KEY     = sys.argv[2]


def extract_wav_name(path_str):
    """从 src 路径提取文件名: '../voices/voice_xxx.wav' -> 'voice_xxx.wav'"""
    if not isinstance(path_str, str):
        return ""
    m = re.search(r"(voice_[^/\\]+\.wav)", path_str)
    if m:
        return m.group(1)
    m = re.search(r"([^/\\]+\.wav)", path_str)
    return m.group(1) if m else ""


def main():
    voices_dir = Path(VOICES_DIR)
    chats_dir  = Path(CHATS_DIR)
    dst_dir    = Path(RAW_ROOT) / TARGET_KEY

    print(f"目标好友 : {TARGET_DISPLAY} (key={TARGET_KEY})")
    print(f"voices   : {voices_dir}")
    print(f"chats    : {chats_dir}")
    print(f"输出到   : {dst_dir}")
    print(f"时长筛选 : {MIN_SEC}~{MAX_SEC} 秒\n")

    if not voices_dir.exists():
        print(f"❌ voices 不存在: {voices_dir}"); sys.exit(1)
    if not chats_dir.exists():
        print(f"❌ chats 不存在: {chats_dir}"); sys.exit(1)

    csv_files = sorted(chats_dir.glob("*.csv"))
    print(f"找到 {len(csv_files)} 个 CSV")

    # 从所有 CSV 的 src 列收集目标好友的语音文件名
    wav_names = set()
    per_csv = {}
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file, encoding="utf-8")
        except Exception as e:
            print(f"  ⚠️  {csv_file.name} 读取失败: {e}"); continue
        sub = df[(df["talker"] == TARGET_DISPLAY) & (df["type_name"] == "voice")]
        cnt = 0
        for _, row in sub.iterrows():
            # src 是路径列; 没有再试 msg
            name = ""
            for col in ["src", "msg", "content"]:
                if col in row and isinstance(row[col], str):
                    name = extract_wav_name(row[col])
                    if name:
                        break
            if name:
                wav_names.add(name)
                cnt += 1
        if cnt:
            per_csv[csv_file.name] = cnt

    print(f"\n各 CSV 里 {TARGET_DISPLAY} 的语音:")
    for name, cnt in per_csv.items():
        print(f"  {name}: {cnt} 条")
    print(f"合计去重后: {len(wav_names)} 条语音记录")

    if not wav_names:
        print(f"\n❌ 没找到 {TARGET_DISPLAY} 的语音。检查昵称是否正确(CSV talker列)")
        sys.exit(1)

    # 去 voices 找文件, 测时长, 筛选, 复制
    dst_dir.mkdir(parents=True, exist_ok=True)
    existing = {f.name for f in dst_dir.glob("*.wav")}

    found, missing, too_short, too_long, copied, dup, err = 0,0,0,0,0,0,0
    durations = []

    for name in sorted(wav_names):
        src = voices_dir / name
        if not src.exists():
            missing += 1
            continue
        found += 1
        try:
            seg = AudioSegment.from_file(src)
            sec = len(seg) / 1000.0
        except Exception:
            err += 1
            continue
        durations.append(sec)
        if sec < MIN_SEC:
            too_short += 1; continue
        if sec > MAX_SEC:
            too_long += 1; continue
        if name in existing:
            dup += 1; continue
        shutil.copy2(src, dst_dir / name)
        copied += 1

    print(f"\n{'='*50}")
    print(f"  voices 文件夹里: 找到 {found}, 缺失 {missing}")
    print(f"  时长筛选: 太短 {too_short}, 太长 {too_long}, 读取失败 {err}")
    print(f"  合适的: {found - too_short - too_long - err}")
    print(f"  复制: 新增 {copied}, 已存在跳过 {dup}")
    print(f"{'='*50}")

    if durations:
        durations.sort()
        total = sum(durations)
        ok_dur = sum(d for d in durations if MIN_SEC <= d <= MAX_SEC)
        print(f"\n时长分布: {len(durations)}条 共{total:.0f}秒({total/60:.1f}分钟)")
        print(f"  ⭐ 合适区间有效时长: {ok_dur:.0f}秒 ({ok_dur/60:.1f}分钟) ← 实际训练用")
        buckets = Counter()
        for d in durations:
            if d < 1: buckets["0-1s"] += 1
            elif d < 3: buckets["1-3s"] += 1
            elif d < 5: buckets["3-5s"] += 1
            elif d < 10: buckets["5-10s"] += 1
            else: buckets[">10s"] += 1
        for b in ["0-1s","1-3s","3-5s","5-10s",">10s"]:
            if buckets[b]:
                print(f"    {b:>6}: {buckets[b]} 条")

    total_in_raw = len(list(dst_dir.glob("*.wav")))
    print(f"\n现在 {dst_dir} 共 {total_in_raw} 条 wav")
    if total_in_raw >= 40:
        print(f"\n✅ 数据充足! 重新训练 (这次质量会明显提升):")
    else:
        print(f"\n数据still偏少但可训:")
    print(f"  cd D:\\GPT-SoVITS\\GPT-SoVITS-v2pro-20250604\\GPT-SoVITS-v2pro-20250604")
    print(f"  runtime\\python.exe train_pipeline.py {TARGET_KEY}")


if __name__ == "__main__":
    main()
