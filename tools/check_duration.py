"""快速看某个 raw 文件夹里 wav 的时长分布, 决定要不要筛短的"""
import sys
from pathlib import Path
from collections import Counter
from pydub import AudioSegment

KEY = sys.argv[1] if len(sys.argv) > 1 else "sweetie"
d = Path(r"D:\GPT-SoVITS\raw") / KEY
wavs = list(d.glob("*.wav"))
print(f"{d}\n共 {len(wavs)} 条\n")

durations = []
for w in wavs:
    try:
        durations.append(len(AudioSegment.from_file(w)) / 1000.0)
    except Exception:
        pass

durations.sort()
total = sum(durations)
ok = sum(d for d in durations if 3 <= d <= 12)
buckets = Counter()
for x in durations:
    if x < 1: buckets["0-1s"] += 1
    elif x < 3: buckets["1-3s"] += 1
    elif x < 5: buckets["3-5s"] += 1
    elif x < 10: buckets["5-10s"] += 1
    else: buckets[">10s"] += 1

print(f"总时长: {total:.0f}秒 ({total/60:.1f}分钟)")
print(f"3-12秒有效时长: {ok:.0f}秒 ({ok/60:.1f}分钟)\n")
print("时长分布:")
for b in ["0-1s","1-3s","3-5s","5-10s",">10s"]:
    if buckets[b]:
        print(f"  {b:>6}: {buckets[b]} 条")

short = buckets["0-1s"] + buckets["1-3s"]
print(f"\n<3秒的碎语音: {short} 条 ({short*100//len(durations)}%)")
if short > len(durations) * 0.3:
    print("→ 碎语音较多, 建议筛掉 (跑 filter_short.py)")
else:
    print("→ 碎语音不多, 可以直接训")
