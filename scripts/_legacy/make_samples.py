"""
生成测试样本视频，用于跑通 扫描 → 分析 → 审核 → 反馈 全链路。

生成的样本用不同动静强度模拟监控场景：
  static   静止空场景（模拟无有效内容）
  motion   强运动（模拟有目标经过）
  gradient 缓慢光影变化（模拟环境干扰导致的误触发）

注意：这些是合成画面，YOLO 不会从中检出真实人物。
它们用于验证**流程与判定逻辑**，检测精度需要用你的真实录像验证。

用法：
    python scripts/make_samples.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config  # noqa: E402

# (文件名, lavfi 滤镜, 说明)
CLIPS = [
    ("static_空场景.mp4", "color=c=gray:s=640x360:d=8:r=15", "静止画面，几乎无动静"),
    ("motion_强运动.mp4", "testsrc=s=640x360:r=15:d=8", "持续变化，动静强烈"),
    ("gradient_光影渐变.mp4", "gradients=s=640x360:d=8:r=15:speed=0.2", "缓慢光影变化"),
]

YEARS = ["2023", "2024"]


def pick_encoder(ffmpeg: str) -> str:
    out = subprocess.run([ffmpeg, "-hide_banner", "-encoders"],
                         capture_output=True, text=True, errors="ignore")
    txt = out.stdout or ""
    if "libx265" in txt:
        return "libx265"
    if "libx264" in txt:
        return "libx264"
    return ""


def main() -> int:
    cfg = load_config()
    from shutil import which
    ffmpeg = cfg.get("runtime", {}).get("ffmpeg_path") or which("ffmpeg") or "ffmpeg"
    enc = pick_encoder(ffmpeg)
    if not enc:
        print("未找到 H.264/H.265 编码器，无法生成样本")
        return 1

    out_root = Path(__file__).resolve().parent.parent / "samples"
    print(f"编码器：{enc}")
    print(f"输出目录：{out_root}\n")

    made = 0
    for year in YEARS:
        for fname, lavfi, desc in CLIPS:
            outdir = out_root / year / "camera01"
            outdir.mkdir(parents=True, exist_ok=True)
            dest = outdir / fname
            if dest.exists():
                continue
            cmd = [ffmpeg, "-v", "error", "-f", "lavfi", "-i", lavfi,
                   "-c:v", enc, "-preset", "ultrafast", "-t", "8",
                   "-pix_fmt", "yuv420p", "-y", str(dest)]
            r = subprocess.run(cmd, capture_output=True, text=True, errors="ignore")
            if r.returncode == 0 and dest.exists():
                made += 1
                print(f"  [OK] {year}/{fname}  ({desc})")
            else:
                print(f"  [FAIL] {year}/{fname}: {(r.stderr or '')[:120]}")

    print(f"\n生成 {made} 个样本")
    print("下一步：在「设置与扫描」页把源目录指向 samples/ 即可试跑全链路。")
    print("注意：回收目录必须与该目录在同一物理盘，否则移动会被拒绝。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
