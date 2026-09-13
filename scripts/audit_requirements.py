"""只读体检：把「删前保全」的四条诉求映射到现库实数。

用途：回答"当前代码是否符合预期"，并让报告里的数字**可复现**（数字真源 = 现库实测，
写进文档的数字会过期，所以文档只留结论与量级，精确值跑本脚本）。

只读：以 `mode=ro` 打开 sqlite，不写库、不移动/删除任何文件。
可选 `--fs` 顺便统计产出目录与音频缓存的占用（要遍历磁盘，慢）。

    ./.venv/Scripts/python.exe scripts/audit_requirements.py
    ./.venv/Scripts/python.exe scripts/audit_requirements.py --fs
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402  只为了拿到与工作台同一份配置

GB = 1024 ** 3


def q(c, sql, *args):
    return c.execute(sql, args).fetchone()[0]


def q1(c, sql, *args):
    row = c.execute(sql, args).fetchone()
    return row[0] if row else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fs", action="store_true", help="额外遍历磁盘统计产出目录占用（慢）")
    ap.add_argument("--out", default="", help="把结果同时写一份到该文件（UTF-8）")
    args = ap.parse_args()

    out = io.StringIO()

    def p(*a):
        print(*a)
        print(*a, file=out)

    cfg = load_config()
    db_path = cfg.get("runtime", {}).get("sqlite_path")
    if not db_path or not Path(db_path).exists():
        print(f"[错误] 找不到数据库：{db_path}（runtime.sqlite_path）", file=sys.stderr)
        return 2

    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    c = con.cursor()

    total = q(c, "SELECT COUNT(*) FROM videos")
    tot_gb = (q1(c, "SELECT SUM(size_bytes) FROM videos") or 0) / GB
    tot_h = (q1(c, "SELECT SUM(duration_sec) FROM videos") or 0) / 3600
    p(f"库：{db_path}")
    p(f"底数：{total} 条 / {tot_gb:.1f} GB / {tot_h:.1f} 小时")
    p("")

    # ---------------- ① 没有人 ----------------
    p("① 「没有人的画面」→ 是否被判可删")
    np_all = q(c, "SELECT COUNT(*) FROM videos WHERE has_person=0")
    np_trash = q(c, "SELECT COUNT(*) FROM videos WHERE has_person=0 AND trash_path IS NOT NULL")
    wp_trash = q(c, "SELECT COUNT(*) FROM videos WHERE has_person=1 AND trash_path IS NOT NULL")
    p(f"   has_person=0 {np_all} 条；其中已移入回收 {np_trash}"
      f"（覆盖率 {np_trash/np_all*100:.1f}%）" if np_all else "   无 has_person=0 的行")
    p(f"   反向：has_person=1 却已在回收 {wp_trash} 条（应来自人工处置或旧口径）")
    for v in ("keep", "trash", "review"):
        p(f"   verdict={v:<6} {q(c, 'SELECT COUNT(*) FROM videos WHERE verdict=?', v)}")
    p("   代码：analyzer.build_decision（人物权重最高）；放行需 verdict.release_require_no_person=true")
    p("")

    # ---------------- ② 明显不好 ----------------
    p("② 「明显不好的视频」→ 有没有独立判据")
    rows = c.execute("SELECT id, size_bytes, has_person, verdict, trash_path, highlight_at,"
                     " frame_stats FROM videos").fetchall()
    bad_total = bad_noperson = bad_noperson_recycled = bad_noperson_bare = 0
    bad_person = 0
    bad_gb = 0.0
    for r in rows:
        try:
            d = json.loads(r["frame_stats"] or "{}")
        except (TypeError, ValueError):
            d = {}
        bad = bool(d.get("is_blurry") or d.get("is_low_light"))
        if not bad:
            continue
        bad_total += 1
        bad_gb += r["size_bytes"] or 0
        if r["has_person"]:
            bad_person += 1
        else:
            bad_noperson += 1
            if r["trash_path"]:
                bad_noperson_recycled += 1
                if not r["highlight_at"]:
                    bad_noperson_bare += 1
    p(f"   画质差（is_blurry 或 is_low_light）共 {bad_total} 条 / {bad_gb/GB:.2f} GB")
    p(f"   其中 has_person=0 的 {bad_noperson} 条（已回收 {bad_noperson_recycled}）；"
      f"has_person=1 的 {bad_person} 条")
    p(f"   ⚠ 已回收 且 一次没剪过 = {bad_noperson_bare} 条 —— 删了就真没了（无画面产物）")
    p("   结论：画质指标算得出、落库了（frame_stats），但 **不参与打分** —— "
      "build_decision 只把它写进 trash 的解释文案")
    p("")

    # ---------------- ③ 精彩画面 ----------------
    p("③ 「保留精彩画面」")
    p(f"   highlight_at 非空        {q(c, 'SELECT COUNT(*) FROM videos WHERE highlight_at IS NOT NULL')}")
    p(f"   已在回收 且已剪辑        {q(c, 'SELECT COUNT(*) FROM videos WHERE trash_path IS NOT NULL AND highlight_at IS NOT NULL')}")
    p(f"   已在回收 但未剪辑        {q(c, 'SELECT COUNT(*) FROM videos WHERE trash_path IS NOT NULL AND highlight_at IS NULL')}")
    p(f"   highlight_error 非空     {q(c, 'SELECT COUNT(*) FROM videos WHERE highlight_error IS NOT NULL')}")
    p(f"   产物片段总数             {q(c, 'SELECT COALESCE(SUM(highlight_count),0) FROM videos')}")
    p(f"   产物时长                 {(q1(c, 'SELECT SUM(highlight_sec) FROM videos') or 0)/3600:.1f} 小时")
    p("   highlight_version 分布（目标 = 见 config.json highlight.version）：")
    for r in c.execute("SELECT highlight_version, COUNT(*) n FROM videos GROUP BY 1 ORDER BY n DESC"):
        p(f"      {str(r[0]):<8} {r[1]}")
    tgt = cfg.get("highlight", {}).get("version")
    behind = q(c, "SELECT COUNT(*) FROM videos WHERE highlight_version IS NOT NULL "
                  "AND highlight_version<>? ", tgt)
    p(f"   → 目标口径 {tgt}：口径不符的已剪行 {behind} 条（进重剪队列）")
    p("")

    # ---------------- ④ 所有可用音频 ----------------
    p("④ 「保留所有可用音频」→ 拿什么兜底")
    has_audio = q(c, "SELECT COUNT(*) FROM videos WHERE has_audio=1")
    sp1 = q(c, "SELECT COUNT(*) FROM videos WHERE speech_present=1")
    archived = q(c, "SELECT COUNT(*) FROM videos WHERE archived_at IS NOT NULL")
    gap_all = q(c, "SELECT COUNT(*) FROM videos WHERE has_audio=1 "
                   "AND (speech_present IS NULL OR speech_present=0)")
    gap_rec = q(c, "SELECT COUNT(*) FROM videos WHERE has_audio=1 "
                   "AND (speech_present IS NULL OR speech_present=0) AND trash_path IS NOT NULL")
    gap_gb = (q1(c, "SELECT SUM(size_bytes) FROM videos WHERE has_audio=1 "
                    "AND (speech_present IS NULL OR speech_present=0) AND trash_path IS NOT NULL") or 0) / GB
    p(f"   有音轨 {has_audio} 条；speech_present=1 仅 {sp1} 条（{sp1/has_audio*100:.1f}%）")
    p(f"   已归档 {archived} 条；已在回收目录但未归档 "
      f"{q(c, 'SELECT COUNT(*) FROM videos WHERE trash_path IS NOT NULL AND archived_at IS NULL')} 条")
    p(f"   ⚠ 有音轨 但 speech_present≠1 = {gap_all} 条 → 不进归档")
    p(f"   ⚠ 其中已在回收目录 = {gap_rec} 条 / {gap_gb:.2f} GB 源片 → "
      f"清空回收目录即音频永久消失")
    p(f"   有转写文本 却 speech_present≠1 = "
      f"{q(c, 'SELECT COUNT(*) FROM videos WHERE transcript_text IS NOT NULL AND '
              'transcript_text<>\'\' AND (speech_present IS NULL OR speech_present=0)')} 条")
    p("   代码：db.archive_queue 带 speech_present=1；归档物 = work_dir/audio/{id}.wav"
      "（16k 单声道 PCM，非原始音轨）")
    p("")

    # ---------------- ⑤ 减少存储 ----------------
    p("⑤ 「减少存储空间」")
    cand = ("(trash_path IS NOT NULL) OR (human_decision IS NULL AND "
            "verdict IN ('trash','review') AND meta_ok=1)")
    p(f"   处置候选（已回收 ∪ 未处置 trash/review）"
      f"{q(c, f'SELECT COUNT(*) FROM videos WHERE {cand}')} 条 / "
      f"{(q1(c, f'SELECT SUM(size_bytes) FROM videos WHERE {cand}') or 0)/GB:.2f} GB")
    p(f"   已移入回收目录 "
      f"{q(c, 'SELECT COUNT(*) FROM videos WHERE trash_path IS NOT NULL')} 条 / "
      f"{(q1(c, 'SELECT SUM(size_bytes) FROM videos WHERE trash_path IS NOT NULL') or 0)/GB:.2f} GB")
    p("   trash_tier：")
    for r in c.execute("SELECT trash_tier, COUNT(*) n FROM videos GROUP BY 1 ORDER BY n DESC"):
        p(f"      {str(r[0]):<8} {r[1]}")
    p("   注：程序只移动、不删除；真正腾出空间要靠机主手动清空回收目录")

    if args.fs:
        p("")
        p("⑥ 产出目录实际占用")
        for label, d in (("音频档案", cfg.get("archive", {}).get("dir")),
                         ("音频缓存", str(Path(cfg.get("work_dir") or ".") / "audio")),
                         ("画面精华", cfg.get("highlight", {}).get("dir"))):
            if not d:
                continue
            n = 0
            s = 0
            for root, _dirs, files in os.walk(d):
                for f in files:
                    try:
                        s += os.path.getsize(os.path.join(root, f))
                        n += 1
                    except OSError:
                        pass
            p(f"   {label:<10} {d} → {n} 个文件 / {s/GB:.2f} GB")

    con.close()
    if args.out:
        Path(args.out).write_text(out.getvalue(), encoding="utf-8")
        print(f"\n已写出：{args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
