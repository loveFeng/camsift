"""引擎冒烟回归：真实中文语音（模型自带 example）vs 纯噪声。

用法：
    python scripts/smoke_audio.py

前置：scripts/setup_audio_engine.py + scripts/download_audio_models.py 已执行；
模型目录由 config.audio.model_dir 决定。首次载入模型约需 1-3 分钟。
"""
import subprocess
import sys
import time
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.audioreview import AudioReviewer  # noqa: E402
from src.config import load_config  # noqa: E402

t0 = time.time()
print(f"[{time.time()-t0:.0f}s] 载入配置...")

cfg = load_config()                     # 读真实配置（audio.model_dir / engine）
cfg.setdefault("audio", {})
cfg["audio"].setdefault("enabled", True)
cfg["audio"].setdefault("min_speech_sec", 1.0)
cfg.setdefault("llm", {})["enabled"] = False   # 冒烟不依赖 ollama
cfg.setdefault("runtime", {})

ar = AudioReviewer(cfg)

mdir = Path(cfg["audio"].get("model_dir") or "data/models/audio")
speech_wav = mdir / "paraformer-zh" / "example" / "asr_example.wav"
noise_wav = Path.cwd() / "_noise_smoke.wav"
if not speech_wav.exists():
    print(f"缺少人声样本：{speech_wav}（先运行 download_audio_models.py）")
    sys.exit(1)

subprocess.run(
    ["ffmpeg", "-y", "-f", "lavfi", "-i",
     "anoisesrc=color=pink:duration=30:amplitude=0.4",
     "-ar", "16000", "-ac", "1", str(noise_wav)],
    capture_output=True, check=True)


def dur(p):
    with wave.open(str(p), "rb") as w:
        return w.getnframes() / w.getframerate()


print(f"[{time.time()-t0:.0f}s] 引擎懒加载（首次载入模型需 1-3 分钟）...")
r1 = ar.review_wav(speech_wav, dur(speech_wav))
print(f"[{time.time()-t0:.0f}s] 人声样本: speech_present={r1['speech_present']} "
      f"ratio={r1['speech_ratio']:.2f} 时长={dur(speech_wav):.0f}s")
print("  转写:", (r1["transcript_text"] or "(空)")[:150].replace("\n", " / "))

r2 = ar.review_wav(noise_wav, dur(noise_wav))
print(f"[{time.time()-t0:.0f}s] 噪声样本: speech_present={r2['speech_present']} "
      f"ratio={r2['speech_ratio']:.2f}")

ok1 = r1["speech_present"] == 1 and bool((r1["transcript_text"] or "").strip())
ok2 = r2["speech_present"] == 0
noise_wav.unlink(missing_ok=True)
print("=" * 50)
print(f"人声检出+转写: {'PASS' if ok1 else 'FAIL'}")
print(f"纯噪声判无人声: {'PASS' if ok2 else 'FAIL'}")
print(f"总耗时 {time.time()-t0:.0f}s")
sys.exit(0 if (ok1 and ok2) else 1)
