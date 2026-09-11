"""实机 A/B 试点：同一条视频，只切换 keep_coarse_suspect，对比产物。

目的：证明 Track C 的「降级保留」不是纸面推论 —— 对同一条被判「无精华」的
真实视频，旧口径交付 0 个产物，新口径交付可播放的 mp4。

只读源文件、只写 data/probe/ 下的临时目录，不碰回收目录、不改数据库。
用法：
    ./.venv/Scripts/python.exe scripts/pilot_salvage_ab.py --id 60
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config          # noqa: E402
from src.db import Database                 # noqa: E402
from src.highlighter import HighlightCutter  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def run_once(cfg, row, out_dir: Path, keep: bool) -> dict:
    cfg = json.loads(json.dumps(cfg))          # 深拷贝，避免两次跑互相污染
    cfg["highlight"]["dir"] = str(out_dir)
    cfg["highlight"]["keep_coarse_suspect"] = keep
    cfg["highlight"]["auto_dispose"] = False   # 试点只剪不处置
    cutter = HighlightCutter(cfg)
    res = cutter.cut_one(dict(row), out_dir, "pilot")
    return res


def summarize(tag: str, res: dict, out_dir: Path) -> None:
    clips = res.get("clips", [])
    st = res.get("stats", {}) or {}
    print(f"\n—— {tag} ——")
    print(f"  ok={res.get('ok')} 段数={len(clips)} note={res.get('note', '')}")
    print(f"  粗扫送检 {st.get('coarse_sent')} 帧 / 粗扫命中 {st.get('coarse_hits')} "
          f"/ 降级保留 {st.get('coarse_downgraded')} / 合并后 hits {st.get('hits')}")
    print(f"  分级 {st.get('grades')}  用时 {st.get('secs')}s")
    for c in clips:
        f = Path(c["file"]) if c.get("file") else None
        size = f.stat().st_size / 1024 if f and f.exists() else 0
        th = len([t for t in (c.get("thumbs") or []) if Path(t).exists()])
        print(f"    {c['grade']} 级 {c['dur']:.1f}s conf={c['conf']:.2f} "
              f"level={c['level']} | mp4={'有 %.0fKB' % size if f and f.exists() else '无'} "
              f"| 静帧 {th} 张")
    mf = out_dir / f"{res.get('year') or '_未分类'}"
    print(f"  产物目录：{mf}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", type=int, default=60)
    args = ap.parse_args()

    cfg = load_config()
    db = Database(cfg["runtime"]["sqlite_path"])
    with db.conn() as c:
        row = c.execute("SELECT * FROM videos WHERE id=?", (args.id,)).fetchone()
    if not row:
        print(f"库里没有 id={args.id}")
        return 1
    row = dict(row)
    print(f"id={args.id} {row.get('filename')}  时长 {row.get('duration_sec'):.1f}s")
    print(f"  库里记录：clips={len(json.loads(row.get('highlight_clips') or '[]'))} 段 "
          f"（sent={row.get('highlight_sent')} hits={row.get('highlight_coarse_hits')}）")

    base = ROOT / "data" / "probe" / "salvage_ab"
    shutil.rmtree(base, ignore_errors=True)   # AUDIT-OK: 清的是本脚本自建的试点输出目录
    old_dir = base / "A_old_behavior"
    new_dir = base / "B_new_behavior"

    r_old = run_once(cfg, row, old_dir, keep=False)
    summarize("A 旧口径（keep_coarse_suspect=False，今天线上的行为）", r_old, old_dir)
    r_new = run_once(cfg, row, new_dir, keep=True)
    summarize("B 新口径（keep_coarse_suspect=True，Track C）", r_new, new_dir)

    n_old = len([c for c in r_old.get("clips", []) if c.get("file")])
    n_new = len([c for c in r_new.get("clips", []) if c.get("file")])
    print("\n" + "=" * 60)
    print(f"旧口径产出可播放 mp4：{n_old} 段　新口径：{n_new} 段")
    if n_old == 0 and n_new > 0:
        print("✅ 证实：旧口径交付 0 产物，新口径把这条信号救回来了")
    elif n_new > n_old:
        print("✅ 新口径多留了内容")
    else:
        print("ℹ️ 这条样本两种口径一致（换一条试）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
