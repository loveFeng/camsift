"""一次性回填：把 manifest.jsonl 里的「粗扫送检帧数 / 粗扫命中数」灌进数据库。

为什么需要它
------------
`highlight_sent` / `highlight_coarse_hits` 是 Track 0 的尺子（见 db.HIGHLIGHT_COLUMNS）。
库里已有的 4,825 条是旧版本剪的，没记这两个数 —— 不回填的话，
「无精华」永远是糊涂账：分不清"真复核过没有"和"压根没跑检测"。

数据来源
--------
`<highlight.dir>/manifest.jsonl` 每行一条，`stats` 里有 coarse_sent / coarse_hits。
manifest 是追加写的，同一 id 可能有多行（重剪过）——**取最后一行**为准。

幂等
----
只写 `highlight_sent IS NULL` 的行，重复执行不会覆盖重剪后的新数。
`--force` 可强制覆盖（默认关闭）。

用法
----
    ./.venv/Scripts/python.exe scripts/backfill_sent.py --dry-run
    ./.venv/Scripts/python.exe scripts/backfill_sent.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config                      # noqa: E402
from src.db import Database                             # noqa: E402
from src.highlighter import MANIFEST_NAME, highlight_dir_of  # noqa: E402

LEGACY_VERSION = "hlv1"     # 旧口径：粗扫疑似被丢弃、C 级只留静帧


def load_manifest(path: Path) -> dict:
    """读 manifest，同 id 取最后一行（重剪过的以最新为准）。"""
    latest = {}
    if not path.exists():
        return latest
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("id") is not None:
                latest[int(r["id"])] = r
    return latest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="只看结果不写库")
    ap.add_argument("--force", action="store_true",
                    help="连已有 highlight_sent 的行也覆盖（默认只补空值）")
    args = ap.parse_args()

    cfg = load_config()
    db = Database(cfg["runtime"]["sqlite_path"])
    mf_path = highlight_dir_of(cfg) / MANIFEST_NAME
    mf = load_manifest(mf_path)
    print(f"manifest: {mf_path}")
    print(f"  条目 {len(mf)} 条（同 id 取最后一行）")

    with db.conn() as c:
        rows = c.execute(
            "SELECT id, highlight_sent, highlight_clips, highlight_version "
            "FROM videos WHERE highlight_at IS NOT NULL").fetchall()

    filled = skipped = no_mf = 0
    sent_sum = hits_sum = 0
    updates = []
    for r in rows:
        vid = r["id"]
        m = mf.get(vid)
        if not m:
            no_mf += 1
            continue
        if r["highlight_sent"] is not None and not args.force:
            skipped += 1
            continue
        st = (m.get("stats") or {})
        sent = int(st.get("coarse_sent") or 0)
        hits = int(st.get("coarse_hits") or 0)
        # 旧产物一律标 hlv1 —— 重剪队列靠它认人
        ver = r["highlight_version"] or LEGACY_VERSION
        updates.append((sent, hits, ver, vid))
        filled += 1
        sent_sum += sent
        hits_sum += hits

    print(f"库里已剪辑 {len(rows)} 条")
    print(f"  可回填 {filled} 条（粗扫送检合计 {sent_sum:,} 帧，命中合计 {hits_sum:,}）")
    print(f"  已有值跳过 {skipped} 条（--force 可覆盖）")
    print(f"  manifest 里找不到 {no_mf} 条（手动删过 manifest？保持 NULL）")

    if args.dry_run:
        print("\n--dry-run：未写入")
    elif updates:
        with db.conn() as c:
            c.executemany(
                "UPDATE videos SET highlight_sent=?, highlight_coarse_hits=?, "
                "highlight_version=? WHERE id=?", updates)
        print(f"\n已写入 {len(updates)} 条")
    else:
        print("\n没有需要写入的行")

    b = db.no_clip_breakdown()
    print("\n「无精华」三义（回填后）：")
    print(f"  已检测·确实没有 {b['detected_empty']}")
    print(f"  预筛跳过(已接受) {b['skipped']}")
    print(f"  看到过但被否决   {b['dropped']}   ← Track C 要救的就是这批")
    print(f"  未知(无 manifest) {b['unknown']}")
    print(f"  合计 {b['total']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
