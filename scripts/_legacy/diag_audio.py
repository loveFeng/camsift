"""诊断：为什么某个视频"明明有声音"却判 0% 人声。

步骤：抽 16kHz wav → ffmpeg 音量统计 → fsmn-vad 原始分段（不做过滤）
→ paraformer 直接整段转写。各步独立打印，一眼看出漏在哪一环。
用法: python scripts/diag_audio.py <视频路径>
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

VIDEO = Path(sys.argv[1] if len(sys.argv) > 1 else r"G:\change\2015\20151213\20151213211936_010.mp4")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config           # noqa: E402
from src.audioreview import AudioReviewer    # noqa: E402

cfg = load_config()
rev = AudioReviewer(cfg)
tmp = Path(tempfile.mkdtemp(prefix="diag_audio_"))
wav = rev.extract_wav(VIDEO, tmp)   # extract_wav 自己拼 .wav 后缀
print(f"[1/4] 抽音轨: {VIDEO}\n      wav = {wav}  size={wav.stat().st_size}")

print("[2/4] ffmpeg 音量统计 (volumedetect):")
r = subprocess.run([rev._ffmpeg(), "-i", str(wav), "-af", "volumedetect",
                    "-f", "null", "-"], capture_output=True)
for line in r.stderr.decode("utf-8", "ignore").splitlines():
    if any(k in line for k in ("mean_volume", "max_volume", "Duration")):
        print("      " + line.strip())

print("[3/4] fsmn-vad 原始分段（无任何过滤）:")
vad, asr, _ = rev._load_engine()
raw = vad.generate(input=str(wav))
print("      raw =", json.dumps(raw, ensure_ascii=False))

segs = []
for item in raw if isinstance(raw, (list, tuple)) else [raw]:
    v = item.get("value") if isinstance(item, dict) else None
    if isinstance(v, list):
        segs = [p for p in v if isinstance(p, (list, tuple)) and len(p) >= 2]
dur = rev._wav_duration(wav)
speech = sum((e - s) / 1000.0 for s, e in segs) if segs else 0.0
print(f"      总时长 {dur:.1f}s, VAD 语音合计 {speech:.1f}s, "
      f"占比 {speech / dur * 100:.1f}%" if dur else "")

print("[4/4] paraformer 整段直接转写（绕过 VAD）:")
raw_asr = asr.generate(input=str(wav))
for item in raw_asr if isinstance(raw_asr, (list, tuple)) else [raw_asr]:
    if isinstance(item, dict):
        print("      text =", repr(item.get("text", "")))
