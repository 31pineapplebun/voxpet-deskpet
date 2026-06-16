"""
筛选 raw/<key>/ 里的 wav, 只保留指定时长区间的, 其余移到备份文件夹
============================================================
不删除, 只是移到 raw/<key>_removed/, 后悔了能找回。

用法: python filter_short.py sweetie [min秒] [max秒]
  默认保留 3-10 秒 (训练最优区间)
"""
import sys
import shutil
from pathlib import Path
from pydub import AudioSegment

KEY = sys.argv[1] if len(sys.argv) > 1 else "sweetie"
MIN_SEC = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
MAX_SEC = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0

src_dir = Path(r"D:\GPT-SoVITS\raw") / KEY
removed_dir = Path(r"D:\GPT-SoVITS\raw") / f"{KEY}_removed"

print(f"筛选 {src_dir}")
print(f"保留区间: {MIN_SEC}~{MAX_SEC} 秒")
print(f"区间外的移到: {removed_dir}\n")

wavs = list(src_dir.glob("*.wav"))
print(f"当前 {len(wavs)} 条")

removed_dir.mkdir(parents=True, exist_ok=True)
kept, moved, err = 0, 0, 0

for w in wavs:
    try:
        sec = len(AudioSegment.from_file(w)) / 1000.0
    except Exception:
        err += 1
        continue
    if MIN_SEC <= sec <= MAX_SEC:
        kept += 1
    else:
        shutil.move(str(w), str(removed_dir / w.name))
        moved += 1

print(f"\n保留: {kept} 条")
print(f"移走: {moved} 条 (在 {removed_dir}, 后悔了能移回来)")
if err:
    print(f"读取失败: {err} 条")

remain = len(list(src_dir.glob("*.wav")))
total_sec = sum(len(AudioSegment.from_file(w))/1000.0 for w in src_dir.glob("*.wav"))
print(f"\n现在 {src_dir} 剩 {remain} 条, 共 {total_sec:.0f}秒 ({total_sec/60:.1f}分钟)")
print(f"\n可以训练了:")
print(f"  cd D:\\GPT-SoVITS\\GPT-SoVITS-v2pro-20250604\\GPT-SoVITS-v2pro-20250604")
print(f"  runtime\\python.exe train_pipeline.py {KEY}")
