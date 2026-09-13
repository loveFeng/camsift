#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""模型对照实验：同一批已目视标注的产物，比较"换模型 / 换输入尺寸"到底有没有用。

为什么要有这个脚本（而不是随手写个临时的）
--------------------------------------------
"模型精度不够，换个大的"是个看起来显然、实测未必成立的判断。2026-09-12 实测：
yolov8n → yolov8s 之后误检被报出的从 3/6 变 4/6，误检的 conf 反而从 0.45 抬到 0.75，
真阳性一条没多。**换更大的同族 YOLO 不是这个问题的解**（详见
`docs/自动剪辑方案说明.md` 第十一节）。结论要能被复核，所以把对照固化成脚本。

两条设计约束（都踩过）
----------------------
1. **严格按标注时的 id 取行，绝不重新抽样。** 库一直被工作台改写，产物池的行序会变
   —— 按 `pool[::step][:20]` 重抽，抽出来的 id 和当初目视的不是同一批
   （曾因此 20 条里 19 条标签对不上，统计全 0/0，不可读）。
2. **每个模型显式带输入尺寸。** 官方导出的 ONNX 有的把 shape 写死在 640，
   统一喂 1280 会直接崩（见 `download_model.py` 的两道门）。

用法
----
    .venv/Scripts/python.exe scripts/compare_models.py                 # 有哪些模型就比哪些
    .venv/Scripts/python.exe scripts/compare_models.py --extra yolo11s # 再挂一个规格
    .venv/Scripts/python.exe scripts/compare_models.py --out D:\\tmp\\cmp.txt

只读：不写库、不动任何原片或产物。
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config as cfgmod  # noqa: E402
from src.analyzer import YoloDetector  # noqa: E402
from src.highlighter import _enhance, probe_av  # noqa: E402

# 逐条**目视**判定过的真值（不是模型给的，是我把 5 帧抓到原始分辨率看出来的）。
# 打分口径：误检 = 确认不是人却报了（最该被压下去）；真人 = 确认是人却漏了（最不该丢）。
# 扩展样本就往这里加 id，别改抽样逻辑 —— 见文件头约束 ①。
LABELS: Dict[int, str] = {
    # 误检（红袋子 / 提包 / 纸箱 / 绿植被稳定判成 person）
    2418: "误检", 9192: "误检", 10941: "误检", 11884: "误检", 7975: "误检", 23184: "误检",
    # 真人（含 23854：坐椅子上、画面占比 0.8%、离得远 —— 最该留的那类）
    5115: "真人", 6696: "真人", 18587: "真人", 23854: "真人", 26459: "真人",
    # 存疑
    21326: "存疑",
    # 抽 5 帧未见人（**不能**当"产物是空的"的证据：1–2 秒的窗口会被抽帧跳过）
    15: "无人", 13478: "无人", 14556: "无人", 15794: "无人",
    16744: "无人", 19834: "无人", 22245: "无人", 24875: "无人",
}

LABEL_ORDER = ("误检", "真人", "存疑", "无人")

# (标签, 规格, 输入尺寸)。同一规格跑多个尺寸，是为了把"换模型"和"只放大输入"分开。
DEFAULT_RUNS: List[Tuple[str, str, int]] = [
    ("yolov8n@640", "yolov8n", 640),
    ("yolov8s@640", "yolov8s", 640),
    ("yolov8m@640", "yolov8m", 640),
    ("yolo11n@640", "yolo11n", 640),
    ("yolo11s@640", "yolo11s", 640),
    ("yolo11m@640", "yolo11m", 640),
    ("yolov8n@1280", "yolov8n", 1280),
]

FRAMES_PER_CLIP = 5


def runs_for(extra: List[str]) -> List[Tuple[str, str, int]]:
    """默认清单 + `--extra` 里那些规格的 640 一档。"""
    runs = list(DEFAULT_RUNS)
    known = {spec for _tag, spec, _size in runs}
    for spec in extra:
        if spec not in known:
            runs.append((f"{spec}@640", spec, 640))
    return runs


def load_detectors(runs, models_dir: Path, log) -> Dict[str, YoloDetector]:
    dets: Dict[str, YoloDetector] = {}
    for tag, spec, size in runs:
        fp = models_dir / f"{spec}.onnx"
        if not fp.exists():
            log(f"  缺模型 {fp.name} → 跳过 {tag}")
            continue
        try:
            d = YoloDetector(str(fp), size, 0.25, 0.45)
            _ = d.net            # 提前触发装载：坏模型当场暴露，别等到推理
        except Exception as e:   # noqa: BLE001
            log(f"  {tag} 装载失败：{type(e).__name__}: {str(e)[:80]}")
            log("    （用 download_model.py 的 verify() 查两道门）")
            continue
        dets[tag] = d
        log(f"  装载 {tag:<14} {fp.stat().st_size/1048576:6.2f} MB  input={size}")
    return dets


def grab_frames(ff: str, clip: Path, vid: int, outdir: Path, log) -> List[Any]:
    """从产物 mp4 里均匀抽 `FRAMES_PER_CLIP` 帧，做与生产一致的 CLAHE 增强。"""
    import cv2

    try:
        dur = probe_av(clip, ff)["duration"] or 0
    except Exception:  # noqa: BLE001
        dur = 0
    frames = []
    for k in range(FRAMES_PER_CLIP):
        t = dur * (0.15 + 0.7 * k / 4) if dur > 0 else 1 + k
        tmp = outdir / f"{vid}_{k}.jpg"
        if not tmp.exists():
            subprocess.run([ff, "-v", "error", "-y", "-ss", f"{t:.2f}", "-i", str(clip),
                            "-frames:v", "1", "-q:v", "2", str(tmp)],
                           capture_output=True, timeout=120)
        img = cv2.imread(str(tmp))
        if img is not None:
            frames.append(_enhance(img, True))
    return frames


def best_person(det: YoloDetector, frames) -> Dict[str, Any] | None:
    """该模型在这条产物上报出的**最自信的 person**（没有则 None）。"""
    best = None
    for enh in frames:
        final, _raw = det.detect(enh, raw_min=0.10)
        for d in final:
            if d["class"] == "person" and (best is None or d["conf"] > best["conf"]):
                best = d
    return best


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="同批样本对比候选检测模型 / 输入尺寸")
    ap.add_argument("--extra", nargs="*", default=[],
                    help="额外要比的规格（如 yolo11s yolo11m），各自跑 640")
    ap.add_argument("--models-dir", default="",
                    help="去哪找 <规格>.onnx；默认生产模型所在目录。"
                         "候选模型要真换上去就得和生产模型放一起，"
                         "所以这里一般不用改（本项目有两处 data 根，别放错）")
    ap.add_argument("--out", default="", help="结果写到哪里（默认临时目录）")
    ap.add_argument("--frames-dir", default="", help="抽帧缓存目录（默认为临时目录）")
    args = ap.parse_args(argv)

    cfg = cfgmod.load_config()
    ff = cfgmod.bin_path(cfg, "ffmpeg_path", "ffmpeg")
    db_path = cfg["runtime"]["sqlite_path"]
    models_dir = (Path(args.models_dir) if args.models_dir
                  else Path(cfg["analysis"].get("yolo_model_path") or ".").parent)
    out_path = Path(args.out) if args.out else Path(tempfile.gettempdir()) / "compare_models.txt"
    fdir = Path(args.frames_dir) if args.frames_dir else Path(tempfile.gettempdir()) / "cmp_frames"
    fdir.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []

    def log(s: str = "") -> None:
        print(s)
        lines.append(s)

    log(f"模型目录 {models_dir}")
    dets = load_detectors(runs_for(args.extra), models_dir, log)
    if not dets:
        log("没有任何可用模型 —— 先跑 scripts/download_model.py")
        Path(out_path).write_text("\n".join(lines), encoding="utf-8")
        return 2
    log("")

    c = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    c.row_factory = sqlite3.Row

    results: Dict[int, Dict[str, Any]] = {}
    ids = sorted(LABELS, key=lambda i: (LABEL_ORDER.index(LABELS[i]), i))
    for i, vid in enumerate(ids, 1):
        lab = LABELS[vid]
        r = c.execute("SELECT id, highlight_clips, highlight_version FROM videos WHERE id=?",
                      (vid,)).fetchone()
        if r is None:
            log(f"[{i:2}] id={vid:<5} {lab:<3} 库里没有此行")
            continue
        try:
            clips = json.loads(r["highlight_clips"] or "[]")
        except Exception:  # noqa: BLE001
            clips = []
        ab = [x for x in clips if x.get("file") and str(x.get("grade")) in ("A", "B")]
        if not ab:
            log(f"[{i:2}] id={vid:<5} {lab:<3} 当前没有 A/B 产物"
                f"（clips={len(clips)}，version={r['highlight_version']}）")
            continue
        clip = max(ab, key=lambda x: x.get("score") or 0)
        f = Path(clip["file"])
        if not f.exists():
            log(f"[{i:2}] id={vid:<5} {lab:<3} 产物文件不存在：{f.name}")
            continue

        frames = grab_frames(ff, f, vid, fdir, log)
        row: Dict[str, Any] = {"label": lab, "frames": len(frames)}
        for tag, det in dets.items():
            try:
                row[tag] = best_person(det, frames)
            except Exception as e:  # noqa: BLE001
                row[tag] = ("ERR", str(e)[:50])
        results[vid] = row

        def fmt(v):
            if isinstance(v, tuple):
                return "  ERR  "
            if v is None:
                return "   —   "
            return "c%.2f/a%.1f%%" % (v["conf"], v["area_ratio"] * 100)

        log(f"[{i:2}] id={vid:<5} {lab:<3} {row['frames']}帧 | "
            + " | ".join(f"{t}:{fmt(row[t])}" for t in dets))

    log("")
    log("== 该条上模型「是否报出 person」==")
    hdr = f"{'标签':<4} {'条数':>4} " + " ".join(f"{t:>13}" for t in dets)
    log(hdr)
    log("-" * (len(hdr) + 4))
    for lab in LABEL_ORDER:
        sub = [v for v in results if results[v]["label"] == lab]
        if not sub:
            continue
        line = f"{lab:<4} {len(sub):>4} "
        for t in dets:
            k = sum(1 for v in sub
                    if results[v][t] is not None and not isinstance(results[v][t], tuple))
            line += f"{f'{k}/{len(sub)}':>13} "
        log(line)

    log("")
    log("== 判读（误检越少越好 / 真人越全越好）==")
    fids = [v for v in results if results[v]["label"] == "误检"]
    tids = [v for v in results if results[v]["label"] == "真人"]

    def hit(ids_, tag):
        """这批里有多少条被该模型**报出**了 person（None 与异常都算"没报出"）。"""
        return sum(1 for v in ids_
                   if results[v][tag] is not None
                   and not isinstance(results[v][tag], tuple))

    for t in dets:
        log(f"{t:<14} 误检仍被报出 {hit(fids, t)}/{len(fids)}   "
            f"真人被报出 {hit(tids, t)}/{len(tids)}")
    log("")
    log("== 误检组的最优 conf（越大越糟：模型越自信那是人）==")
    for t in dets:
        vals = [results[v][t]["conf"] for v in fids
                if results[v][t] is not None and not isinstance(results[v][t], tuple)]
        log(f"{t:<14} " + (" ".join("%.2f" % x for x in sorted(vals, reverse=True))
                           or "（一条都没报）")
            + (f"   max={max(vals):.2f}" if vals else ""))

    Path(out_path).write_text("\n".join(lines), encoding="utf-8")
    print(f"\n结果已写入 {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
