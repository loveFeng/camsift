#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""周期合辑编译器（原型 · 待评审）—— 把一段时间的画面精华拼成一条成片。

和 L2.8 的分工（两者不重叠）
---------------------------
    src/highlighter.py  = 检索 + 切割：找出"哪几秒有价值"，切成一个个 mp4
    本脚本              = 编排 + 拼接：把某周 / 某月的片段按时间顺序串成一条片子

highlighter 不做时间线，本脚本不碰检测。所以这不是"给精华剪辑加个按钮"，
而是一个独立的**下游模块**（项目术语表里 "成片/集锦" 被刻意排除在 "画面精华"
之外，正是为了不让这两件事混成一个概念）。

为什么不能一条 ffmpeg 命令拼完
------------------------------
一次 filter_complex 拼 300+ 段会撞 **Windows 命令行 32K 字符上限**
（2016-01 就有 327 段，光路径就 2.6 万字符）。所以走两遍：

    pass 1  逐段归一化转码：分辨率 / 帧率 / 像素格式 / 采样率 / 声道统一，
            无音轨的补静音轨（否则 concat 时流数量不一致会直接失败）
    pass 2  concat demuxer + -c copy：秒级完成

两遍法还有个附带好处 —— pass 1 的中间片段可复用，改顺序 / 换片头不必重新
解码源片。

安全边界
--------
- 只读源片与数据库；只写 --out 目录下的自创文件
- 中间片段落在 --out 的 _work/ 下，跑完即清（AUDIT-OK：自创临时目录）
- --out 为空 / 盘符根目录时拒绝执行；与源盘同盘只提示不阻止（沿用项目口径）

用法
----
    cd E:/code/video-search
    ./.venv/Scripts/python.exe scripts/compile_period.py --list
    ./.venv/Scripts/python.exe scripts/compile_period.py --by month --plan
    ./.venv/Scripts/python.exe scripts/compile_period.py --by month --value 2015-12
    ./.venv/Scripts/python.exe scripts/compile_period.py --by week  --value 2015-W52 --grades A
"""
from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.archiver import is_drive_root, same_drive_warnings  # noqa: E402
from src.config import bin_path, load_config  # noqa: E402

# 归一化目标：与现有产物同规格（源片就是 1280x720/15fps），不升帧、不放大
TARGET_W, TARGET_H, TARGET_FPS = 1280, 720, 15
TARGET_AC, TARGET_AR = 1, 48000
VF = (f"scale={TARGET_W}:{TARGET_H}:force_original_aspect_ratio=decrease,"
      f"pad={TARGET_W}:{TARGET_H}:(ow-iw)/2:(oh-ih)/2,setsar=1,"
      f"fps={TARGET_FPS},format=yuv420p")


# --------------------------------------------------------------------------- 基础设施
def _bin(cfg: Dict[str, Any], key: str, name: str) -> str:
    p = bin_path(cfg, key, name)
    if not p:
        raise SystemExit(f"未找到 {name}，请在配置里指定 runtime.{key}")
    return p


def _run(cmd: List[str], timeout: int = 900) -> Tuple[int, str]:
    r = subprocess.run(cmd, capture_output=True, timeout=timeout)
    return r.returncode, r.stderr.decode("utf-8", "ignore").strip()


def probe(path: Path, ffprobe: str) -> Dict[str, Any]:
    """只读 header：时长 / 音轨。"""
    out = {"duration": 0.0, "has_audio": False}
    cmd = [ffprobe, "-v", "quiet", "-print_format", "json",
           "-show_format", "-show_streams", str(path)]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=60)
        data = json.loads(r.stdout.decode("utf-8", "ignore") or "{}")
    except Exception:  # noqa: BLE001
        return out
    try:
        out["duration"] = float((data.get("format") or {}).get("duration") or 0)
    except (TypeError, ValueError):
        pass
    out["has_audio"] = any(s.get("codec_type") == "audio"
                           for s in (data.get("streams") or []))
    return out


# --------------------------------------------------------------------------- 数据
def load_clips(cfg: Dict[str, Any], grades: Optional[set]) -> Tuple[List[Dict], Dict[str, int]]:
    """从库里读出所有「有可播放 mp4」的片段，带上真实拍摄时间。"""
    db_path = cfg["runtime"]["sqlite_path"]
    if not Path(db_path).exists():
        raise SystemExit(f"数据库不存在：{db_path}")
    conn = sqlite3.connect(f"file:{Path(db_path).as_posix()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, filename, year, highlight_clips FROM videos "
        "WHERE highlight_clips IS NOT NULL AND highlight_clips NOT IN ('[]','')"
    ).fetchall()
    conn.close()

    clips: List[Dict[str, Any]] = []
    stats = Counter()
    for r in rows:
        try:
            items = json.loads(r["highlight_clips"])
        except (TypeError, ValueError):
            stats["json_bad"] += 1
            continue
        for cl in items:
            f = cl.get("file")
            if not f:                       # hlv1 的 C 级只有静帧，没有视频可拼
                stats["no_mp4"] += 1
                continue
            if grades and str(cl.get("grade")) not in grades:
                stats["grade_filtered"] += 1
                continue
            p = Path(f)
            if not p.exists():
                stats["file_missing"] += 1
                continue
            sa = cl.get("shot_at")
            if not sa:
                stats["no_shot_at"] += 1
                continue
            clips.append({
                "vid": r["id"], "src": str(p), "grade": cl.get("grade"),
                "dur": float(cl.get("dur") or 0), "score": float(cl.get("score") or 0),
                "conf": cl.get("conf"), "n_person": cl.get("n_person"),
                "shot_at": sa, "stem": Path(r["filename"] or p.name).stem,
            })
            stats["usable"] += 1
    return clips, dict(stats)


def period_key(shot_at: str, by: str) -> Optional[str]:
    """月份键 YYYY-MM；周键按 ISO 年-周（跨年周不会串到上一年）。"""
    try:
        dt = datetime.fromisoformat(shot_at)
    except (TypeError, ValueError):
        return None
    if by == "month":
        return dt.strftime("%Y-%m")
    y, w, _ = dt.isocalendar()
    return f"{y}-W{w:02d}"


def group(clips: List[Dict], by: str) -> Dict[str, List[Dict]]:
    out: Dict[str, List[Dict]] = defaultdict(list)
    for c in clips:
        k = period_key(c["shot_at"], by)
        if k:
            out[k].append(c)
    for k in out:
        out[k].sort(key=lambda c: c["shot_at"])
    return dict(sorted(out.items()))


def pick(period_clips: List[Dict], max_minutes: float, per_clip_cap: float) -> List[Dict]:
    """按时长预算挑片段：先按价值降序取，再按时间升序排回。

    单段时长上限（per_clip_cap）是为了不让一条 52 秒的长段独占成片 —— 它本身就是
    "整片都有内容"的候选（highlighter 的 keep_full_ratio 逻辑会把它改判整片保留）。
    """
    usable = [c for c in period_clips if 0 < c["dur"] <= per_clip_cap] or period_clips
    if max_minutes <= 0:
        return sorted(usable, key=lambda c: c["shot_at"])
    budget = max_minutes * 60.0
    ranked = sorted(usable, key=lambda c: (c["grade"] != "A", -c["score"], c["shot_at"]))
    taken: List[Dict] = []
    used = 0.0
    for c in ranked:
        if used + c["dur"] > budget:
            continue
        taken.append(c)
        used += c["dur"]
        if used >= budget * 0.995:
            break
    taken.sort(key=lambda c: c["shot_at"])
    return taken


# --------------------------------------------------------------------------- 拼接
def normalize_one(clip: Dict, dst: Path, ffmpeg: str, ffprobe: str,
                  crf: int, preset: str) -> Tuple[bool, str]:
    """pass 1：单段归一化。无音轨补静音，保证 concat 时流结构一致。"""
    src = clip["src"]
    info = probe(Path(src), ffprobe)
    base = [ffmpeg, "-v", "error", "-y", "-i", src]
    if info["has_audio"]:
        maps = ["-map", "0:v:0", "-map", "0:a:0"]
    else:
        base += ["-f", "lavfi", "-i", f"anullsrc=r={TARGET_AR}:cl=mono"]
        maps = ["-map", "0:v:0", "-map", "1:a:0", "-shortest"]
    cmd = base + maps + [
        "-vf", VF,
        "-c:v", "libx264", "-crf", str(crf), "-preset", preset,
        "-g", str(int(TARGET_FPS * 2)),
        "-c:a", "aac", "-b:a", "128k", "-ar", str(TARGET_AR), "-ac", str(TARGET_AC),
        "-movflags", "+faststart", str(dst),
    ]
    try:
        rc, err = _run(cmd)
    except subprocess.TimeoutExpired:
        return False, "转码超时"
    if rc != 0 or not dst.exists() or dst.stat().st_size == 0:
        return False, (err.splitlines() or ["未知错误"])[-1][:200]
    return True, ""


def concat(list_path: Path, out: Path, ffmpeg: str) -> Tuple[bool, str]:
    """pass 2：concat demuxer + 流复制（pass 1 已保证参数一致）。"""
    cmd = [ffmpeg, "-v", "error", "-y", "-f", "concat", "-safe", "0",
           "-fflags", "+genpts", "-i", str(list_path),
           "-c", "copy", "-movflags", "+faststart", str(out)]
    try:
        rc, err = _run(cmd, timeout=1800)
    except subprocess.TimeoutExpired:
        return False, "拼接超时"
    if rc != 0 or not out.exists() or out.stat().st_size == 0:
        return False, (err.splitlines() or ["未知错误"])[-1][:300]
    return True, ""


# --------------------------------------------------------------------------- 主流程
def build_one(key: str, clips: List[Dict], cfg: Dict[str, Any], out_root: Path,
              crf: int, preset: str, max_minutes: float, per_clip_cap: float,
              force: bool, dry: bool, ffmpeg: str, ffprobe: str,
              cleanup: bool = False) -> Dict[str, Any]:
    chosen = pick(clips, max_minutes, per_clip_cap)
    total_sec = sum(c["dur"] for c in chosen)
    grades = Counter(c["grade"] for c in chosen)
    plan = {
        "period": key, "clips": len(chosen), "sec": round(total_sec, 1),
        "grades": dict(grades),
        "first": chosen[0]["shot_at"] if chosen else None,
        "last": chosen[-1]["shot_at"] if chosen else None,
    }
    print(f"  {key}: {len(chosen)} 段 / {total_sec/60:.1f} 分钟  "
          f"[A {grades.get('A',0)} · B {grades.get('B',0)}]  "
          f"{plan['first']} → {plan['last']}")
    if dry or not chosen:
        return plan

    out = out_root / f"{key}.mp4"
    if out.exists() and not force:
        print(f"    已存在，跳过（--force 可重编）：{out.name}")
        plan["skipped"] = True
        return plan

    # 中间片段目录：**不预先清空** —— 片段名是确定的 00001.mp4…，重跑直接覆盖，
    # 没必要为了"干净"去 rmtree（项目一贯口径：删前保全，能不清就不清）。
    work = out_root / "_work" / key
    work.mkdir(parents=True, exist_ok=True)

    ok_cnt = 0
    failures: List[str] = []
    parts: List[Path] = []
    t0 = time.time()
    for i, c in enumerate(chosen, 1):
        part = work / f"{i:05d}.mp4"
        ok, err = normalize_one(c, part, ffmpeg, ffprobe, crf, preset)
        if ok:
            ok_cnt += 1
            parts.append(part)
        else:
            failures.append(f"{c['stem']}@{c['shot_at']}: {err}")
        if i % 20 == 0 or i == len(chosen):
            print(f"    pass1 {i}/{len(chosen)}  已用 {time.time()-t0:.0f}s")

    if not parts:
        plan["error"] = "全部片段转码失败"
        return plan

    lst = work / "list.txt"
    lst.write_text("".join(f"file '{p.as_posix()}'\n" for p in parts), encoding="utf-8")
    ok, err = concat(lst, out, ffmpeg)
    if not ok:
        plan["error"] = f"拼接失败：{err}"
        return plan

    info = probe(out, ffprobe)
    size = out.stat().st_size
    manifest = {
        "period": key, "file": str(out), "clips": ok_cnt,
        "failed": len(failures), "duration_sec": round(info["duration"], 1),
        "bytes": size, "crf": crf, "preset": preset,
        "target": f"{TARGET_W}x{TARGET_H}@{TARGET_FPS} {TARGET_AR}Hz mono",
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "items": [{"t": c["shot_at"], "grade": c["grade"], "dur": c["dur"],
                   "score": c["score"], "src": c["src"]} for c in chosen],
        "failures": failures,
    }
    (out_root / f"{key}.manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [f"周期合辑 节目单 — {key}",
             f"时长 {info['duration']/60:.1f} 分钟 / {size/1024**2:.0f} MB / "
             f"{ok_cnt} 段（失败 {len(failures)}）",
             f"目标规格 {TARGET_W}x{TARGET_H}@{TARGET_FPS}fps · "
             f"{TARGET_AR}Hz 单声道", ""]
    for i, c in enumerate(chosen, 1):
        lines.append(f"{i:>4}. {c['shot_at'][:19]}  {c['grade']}  "
                     f"{c['dur']:5.1f}s  conf={c['conf']}  {c['stem']}")
    (out_root / f"{key}.节目单.txt").write_text("\n".join(lines), encoding="utf-8")

    print(f"    ✅ {out.name}  {info['duration']/60:.1f} 分钟  {size/1024**2:.0f} MB  "
          f"（失败 {len(failures)}，耗时 {time.time()-t0:.0f}s）")
    if cleanup:
        shutil.rmtree(work, ignore_errors=True)   # AUDIT-OK: 自创临时目录，显式 --cleanup 才删
    else:
        print(f"    中间片段保留在 {work}（确认成片没问题后可手动删，或加 --cleanup）")
    plan.update({"file": str(out), "bytes": size,
                 "duration_sec": round(info["duration"], 1),
                 "failed": len(failures)})
    return plan


def main() -> int:
    ap = argparse.ArgumentParser(description="把画面精华按周/按月拼成一条成片")
    ap.add_argument("--by", choices=["month", "week"], default="month")
    ap.add_argument("--value", default="", help="期号，如 2015-12（月）或 2015-W52（周）；留空=全部")
    ap.add_argument("--grades", default="A,B", help="纳入的级别，默认 A,B（C 级是噪声档案）")
    ap.add_argument("--max-minutes", type=float, default=0, help="每期时长上限，0=不限")
    ap.add_argument("--per-clip-cap", type=float, default=20.0, help="单段时长上限（秒）")
    ap.add_argument("--crf", type=int, default=20)
    ap.add_argument("--preset", default="veryfast")
    ap.add_argument("--out", default="", help="输出目录（默认 work_dir/compilations）")
    ap.add_argument("--list", action="store_true", help="只列出可编的期")
    ap.add_argument("--plan", action="store_true", help="只算计划不产出（dry-run）")
    ap.add_argument("--force", action="store_true", help="已存在也重编")
    ap.add_argument("--cleanup", action="store_true",
                    help="成片成功后清掉 _work 下的中间片段（默认保留）")
    args = ap.parse_args()

    cfg = load_config()
    grades = {g.strip().upper() for g in args.grades.split(",") if g.strip()} or None
    out_root = Path(args.out) if args.out else Path(cfg.get("work_dir") or ".") / "compilations"
    if is_drive_root(out_root):
        raise SystemExit("输出目录不能为空或盘符根目录")
    for w in same_drive_warnings(out_root, cfg, "合辑输出目录"):
        print("⚠️ ", w)

    t0 = time.time()
    clips, stats = load_clips(cfg, grades)
    print(f"可用片段 {len(clips)} 段（{sum(c['dur'] for c in clips)/60:.1f} 分钟）"
          f"  过滤统计 {stats}")
    if not clips:
        return 1
    groups = group(clips, args.by)
    print(f"按{('月' if args.by=='month' else '周')}分组：{len(groups)} 期\n")

    targets = {args.value: groups[args.value]} if args.value else groups
    if args.value and args.value not in groups:
        print(f"没有这一期：{args.value}")
        return 1

    if args.list:
        for k, v in groups.items():
            g = Counter(c["grade"] for c in v)
            print(f"  {k}  {len(v):>4} 段  {sum(c['dur'] for c in v)/60:>6.1f} 分钟  "
                  f"A {g.get('A',0)} / B {g.get('B',0)}")
        return 0

    ffmpeg = _bin(cfg, "ffmpeg_path", "ffmpeg")
    ffprobe = _bin(cfg, "ffprobe_path", "ffprobe")
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"输出目录：{out_root}\n")

    results = [build_one(k, v, cfg, out_root, args.crf, args.preset,
                         args.max_minutes, args.per_clip_cap,
                         args.force, args.plan, ffmpeg, ffprobe,
                         cleanup=args.cleanup)
               for k, v in targets.items()]

    ok = [r for r in results if r.get("file")]
    print(f"\n完成 {len(ok)}/{len(results)} 期，产物 "
          f"{sum(r.get('bytes',0) for r in ok)/1024**2:.0f} MB，"
          f"总时长 {sum(r.get('duration_sec',0) for r in ok)/60:.1f} 分钟，"
          f"耗时 {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
