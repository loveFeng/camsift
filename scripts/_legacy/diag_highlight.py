"""精华剪辑诊断脚本 —— 看清单个视频在两级检测里到底发生了什么。

用法：
    .\\.venv\\Scripts\\python.exe scripts/diag_highlight.py [视频条数]

输出每级的抽帧数、送检数、每帧检出（类别+置信度）、动静比，
用来判断"精扫全否"究竟是正常过滤还是时间戳错位。
"""
from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402

from src.config import load_config  # noqa: E402
from src.db import Database  # noqa: E402
from src import highlighter as hl  # noqa: E402


def main(n: int = 2):
    cfg = load_config()
    cfg["work_dir"] = str(Path(tempfile.gettempdir()) / "hl-diag")
    db = Database(cfg["runtime"]["sqlite_path"])
    rows = db.highlight_queue(limit=n)
    cut = hl.HighlightCutter(cfg)
    h = cfg["highlight"]
    tmp = Path(cfg["work_dir"]) / "diag"

    for row in rows:
        row = dict(row)
        src = Path(row.get("trash_path") or row["path"])
        if not src.exists():
            print(f"[跳过] 源文件不存在 {src}")
            continue
        info = hl.probe_av(src, cut.ffprobe)
        duration = info["duration"]
        print("=" * 70)
        print(f"id={row['id']}  {src.name}  {duration:.1f}s  fps={info['fps']}")

        # ---------------- 第一级 ----------------
        c_fps = float(h.get("coarse_fps", 3.0))
        frames = cut._extract(src, tmp / "coarse", c_fps,
                              int(h.get("coarse_size", 640)), duration)
        print(f"粗扫：抽帧 {len(frames)} 张 @ {c_fps}fps")
        c_hits, c_diffs, c_sent = cut._scan(
            frames, c_fps, 0.0, duration, cut.coarse,
            float(h.get("coarse_conf", 0.15)),
            bool(h.get("motion_prefilter", True)),
            {"person"}, float(h.get("motion_fallback_threshold", 0.05)))
        print(f"      送检 {c_sent} 张，命中 {len(c_hits)} 帧")
        mthr = float(h.get("motion_threshold", 0.004))
        shown = 0
        for i, p in enumerate(frames):
            if c_diffs[i] <= mthr:
                continue
            img = cv2.imread(str(p))
            if img is None:
                continue
            final, raw = cut.coarse.detect(hl._enhance(img, True), raw_min=0.10)
            if not final:
                continue
            dets = ", ".join(f"{d['class']}:{d['conf']:.2f}" for d in final[:3])
            print(f"      t={i / c_fps:6.2f}s  动静={c_diffs[i]:.4f}  {dets}")
            shown += 1
            if shown >= 8:
                print("      ...")
                break

        if not c_hits:
            print("      （粗扫无任何候选，跳过精扫）")
            continue

        # ---------------- 第二级 ----------------
        seeds = [x for x in c_hits if not x.get("motion_only")
                 or c_diffs[min(int(x["t"] * c_fps), len(c_diffs) - 1)] > 0.05]
        wins = hl.build_windows(seeds, duration,
                                gap=float(h.get("gap_merge", 3.0)) + 1.0,
                                pad=float(h.get("fine_window_pad", 2.0)))
        print(f"精扫：候选窗 {len(wins)} 个 {[(round(a,1), round(b,1)) for a, b in wins]}")
        f_fps = float(h.get("fine_fps", 4.0))
        for i, (ws, we) in enumerate(wins[:3]):
            fr = cut._extract(src, tmp / f"w{i}", f_fps,
                              int(h.get("fine_size", 1280)), we - ws, ws)
            got, d2, sent = cut._scan(fr, f_fps, ws, duration, cut.fine,
                                      float(h.get("fine_conf", 0.25)), False,
                                      {"person"}, 0.05)
            print(f"      窗[{ws:.1f},{we:.1f}] 抽帧 {len(fr)} 张，"
                  f"送检 {sent}，精扫命中 {len(got)}")
            for k, p in enumerate(fr):
                img = cv2.imread(str(p))
                if img is None:
                    continue
                final, raw = cut.fine.detect(hl._enhance(img, True), raw_min=0.10)
                if final:
                    dets = ", ".join(f"{d['class']}:{d['conf']:.2f}" for d in final[:3])
                    print(f"         t={ws + k / f_fps:6.2f}s  {dets}")
                    break
            else:
                # 打印该窗内原始检出的最高分，看是"完全没看到"还是"看到但没过阈值"
                best = 0.0
                for p in fr:
                    img = cv2.imread(str(p))
                    if img is None:
                        continue
                    _f, raw = cut.fine.detect(hl._enhance(img, True), raw_min=0.05)
                    for d in raw:
                        if d["class"] == "person":
                            best = max(best, d["conf"])
                print(f"         窗内 person 原始最高置信度 {best:.3f}"
                      f"（精扫阈值 {h.get('fine_conf', 0.25)}）")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2)
