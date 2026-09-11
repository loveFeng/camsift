"""精华剪辑 30 样本验证 —— 用真实素材回答三个问题：

1. 命中率：1372 段里到底有多少能捞出东西？A/B/C 各多少？
2. 耗时：单视频平均多少秒？全量 1372 段要跑多久？
3. 两级检测值不值：精扫到底否决了多少粗扫误报、又捞回了多少真目标？

产物写到 E:\\video-highlights\\_sample30（独立于正式目录，验证完可整体删掉）。
不改数据库 —— 验证是干跑，正式跑才落库。

用法：
    .\\.venv\\Scripts\\python.exe scripts/sample_highlight.py [样本数]
"""
from __future__ import annotations

import json
import random
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config  # noqa: E402
from src.db import Database  # noqa: E402
from src import highlighter as hl  # noqa: E402


def main(n: int = 30, seed: int = 42):
    cfg = load_config()
    # 抽帧中间产物放系统临时目录，验证完顺手清掉，不碰工作盘
    cfg["work_dir"] = str(Path(tempfile.gettempdir()) / "hl-sample")
    out_dir = hl.highlight_dir_of(cfg) / "_sample30"
    out_dir.mkdir(parents=True, exist_ok=True)

    db = Database(cfg["runtime"]["sqlite_path"])
    rows = [dict(r) for r in db.highlight_queue(limit=5000)]
    random.Random(seed).shuffle(rows)
    rows = rows[:n]
    print(f"样本：{len(rows)} 段（随机种子 {seed}）")
    print(f"输出：{out_dir}")
    print("=" * 72)

    cut = hl.HighlightCutter(cfg)
    grades = Counter()
    hit_videos = 0
    empty_videos = 0
    failed = 0
    total_sec = 0.0
    coarse_sent_sum = coarse_frames_sum = 0
    fine_sent_sum = 0
    fine_rescues = 0      # 粗扫没戏但精扫确认的帧数（两级价值的直接证据）
    fine_rejects = 0      # 粗扫报了但精扫否决的窗数
    t_all = time.time()
    per_video_secs: list[float] = []
    examples: list[str] = []

    for i, row in enumerate(rows, 1):
        t0 = time.time()
        try:
            r = cut.cut_one(row, out_dir, datetime.now().isoformat(timespec="seconds"))
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"[{i:2d}] 失败 {row['filename']}: {type(e).__name__}: {e}")
            continue
        dt = time.time() - t0
        per_video_secs.append(dt)
        st = r.get("stats", {})
        clips = r.get("clips", [])
        cs = Counter(c["grade"] for c in clips)
        grades.update(cs)
        total_sec += r.get("sec", 0.0)
        coarse_frames_sum += st.get("coarse_frames", 0)
        coarse_sent_sum += st.get("coarse_sent", 0)
        fine_sent_sum += st.get("fine_sent", 0)
        fine_rescues += sum(1 for c in clips if c["level"] >= hl.LV_CONFIRMED)
        if st.get("windows", 0) and not st.get("windows_scanned"):
            pass
        if st.get("coarse_hits", 0) and not any(
                c["level"] >= hl.LV_CONFIRMED for c in clips):
            fine_rejects += 1
        tag = " ".join(f"{g}{cs[g]}" for g in ("A", "B", "C") if cs[g]) or "无"
        if clips:
            hit_videos += 1
            best = max(clips, key=lambda c: c["conf"])
            if best["conf"] >= float(cfg["highlight"].get("fine_conf", 0.25)) \
                    and len(examples) < 8:
                examples.append(f"{row['filename']} → {tag} (conf {best['conf']:.2f})")
        else:
            empty_videos += 1
        print(f"[{i:2d}] {row['filename']}  {st.get('duration', 0):5.1f}s "
              f"粗扫{st.get('coarse_sent', 0):3d}/{st.get('coarse_frames', 0):3d} "
              f"精扫{st.get('fine_sent', 0):3d}帧 "
              f"窗{st.get('windows_scanned', 0)} → {tag:<12} {dt:5.1f}s")

    n = len(per_video_secs) or 1
    avg = sum(per_video_secs) / n
    print("=" * 72)
    print(f"命中 {hit_videos} 段 / 空镜 {empty_videos} 段 / 失败 {failed} 段")
    print(f"分级：A={grades['A']}  B={grades['B']}  C={grades['C']}"
          f"　切出视频时长合计 {total_sec / 60:.1f} 分钟")
    print(f"帧差预筛：粗扫送检 {coarse_sent_sum}/{coarse_frames_sum} "
          f"（省 {100 - 100 * coarse_sent_sum / max(1, coarse_frames_sum):.0f}% 算力）")
    print(f"精扫送检 {fine_sent_sum} 帧 @1280 —— 高分辨率复核的成本大头")
    print(f"精扫确认目标（A/B）：{fine_rescues} 段；粗扫有动静但被精扫否决：{fine_rejects} 段")
    print(f"单视频平均 {avg:.1f}s → 全量 1372 段约 {avg * 1372 / 3600:.1f} 小时")
    if examples:
        print("-" * 72)
        print("高置信命中示例（可打开 _sample30 下的 mp4 直接核对）：")
        for e in examples:
            print(f"  {e}")
    print("=" * 72)
    print(f"总耗时 {time.time() - t_all:.0f}s")
    print(f"验证产物在 {out_dir}，确认无误后可整体删除")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 30)
