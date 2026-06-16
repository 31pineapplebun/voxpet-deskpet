"""
从 raw/<key>/ 里挑选最适合做参考音频的候选
==========================================
GPT-SoVITS 参考音频的理想条件:
  - 时长 6-10 秒 (太短不稳定, 太长也不好)
  - 音质清晰、响度适中
  - 最好是陈述句 (这个脚本判断不了内容, 需要你听)

用法: python pick_ref.py [英文key]   默认 buddy
列出候选后, 你逐个听, 挑一条清晰平稳的, 记下文件名和它说的话
"""
import sys
from pathlib import Path
from pydub import AudioSegment

RAW_ROOT = r"D:\GPT-SoVITS\raw"
KEY = sys.argv[1] if len(sys.argv) > 1 else "buddy"

d = Path(RAW_ROOT) / KEY
wavs = list(d.glob("*.wav"))
print(f"{d} 共 {len(wavs)} 条\n")

# 算每条的时长和响度
items = []
for w in wavs:
    try:
        seg = AudioSegment.from_file(w)
        sec = len(seg) / 1000.0
        dbfs = seg.dBFS
        items.append((w.name, sec, dbfs))
    except Exception:
        continue

# 理想参考: 6-10秒优先, 其次 5-6秒和 10-12秒, 按响度排序(响度适中的好)
def score(item):
    name, sec, dbfs = item
    # 时长得分: 6-10秒最高
    if 6 <= sec <= 10:
        dur_score = 100
    elif 5 <= sec < 6 or 10 < sec <= 12:
        dur_score = 70
    else:
        dur_score = 30
    return dur_score

items.sort(key=score, reverse=True)

print("最适合做参考音频的候选 (按推荐度排序, 听前几条挑一条清晰的):\n")
print(f"  {'时长':>6} {'响度dBFS':>9}   文件名")
print(f"  {'-'*6} {'-'*9}   {'-'*40}")
for name, sec, dbfs in items[:15]:
    star = " ⭐" if 6 <= sec <= 10 else ""
    print(f"  {sec:>5.1f}s {dbfs:>8.1f}   {name}{star}")

print(f"\n挑选建议:")
print(f"  1) 优先听带 ⭐ 的 (6-10秒)")
print(f"  2) 在文件夹里双击播放, 挑一条: 发音清楚、不太吵、是陈述句(不是问句/单字)")
print(f"  3) 记下文件名 + 它说的话(一字不差)")
print(f"  4) 把这两个填到 friends.py 里 buddy 的 ref_audio 和 ref_text")
