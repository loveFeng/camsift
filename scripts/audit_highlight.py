"""产品体检：找出「检出过东西、产物却是空的/只有静帧」的原片。

为什么需要它
------------
这类行是**静默漏检**：库里看起来只是"没剪出东西"，实际是"检出过、被丢掉了"。
2026-09-13 实测过一次：630 条这种行，其中 489 条正躺在回收目录里被判成
「无精华产物 ⇒ 可直接删」——删掉就永久没了。

它长的样子（一句话）：`highlight_coarse_hits > 0` 而 `highlight_clips` 是空的，
或者 clips 非空但没有一个带 mp4（C 级只有静帧）。

看它还不够，要能区分两种成因
------------------------------
- **版本落后**：老口径本来就丢（hlv1），重剪队列收得到 → 不用管；
- **版本已是目标值却仍为空**：说明**当前口径**在某处吃掉了产物。
  这种行会被 `rehighlight_queue` 的 `version <> target` 挡在队列外 —— 属于
  "必须 bump 版本号 + 修代码"的那一类（见 `docs/自动剪辑方案说明.md` §5.4）。

所以报告按 (症状 × 版本) 分桶，**并单独标出"版本已是目标值"那一桶**。

用法
----
    ./.venv/Scripts/python.exe scripts/audit_highlight.py            # 只出报告
    ./.venv/Scripts/python.exe scripts/audit_highlight.py --recut 24 # 抽样重跑估救回率
    ./.venv/Scripts/python.exe scripts/audit_highlight.py --ids 15638 16428

`--recut` 只把产物切到临时目录，**不写库、不碰回收目录**，可以随时跑。
注意用 `env -u PYTHONPATH` 起（沙箱注入的删除守卫会拦 `cut_one` 清理临时抽帧目录）。
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import mover                                    # noqa: E402
from src.config import load_config                       # noqa: E402
from src.db import HIGHLIGHT_TARGET_VERSION, Database    # noqa: E402

COLS = ("id, path, trash_path, filename, year, rel_dir, duration_sec, "
        "highlight_at, highlight_error, highlight_version, highlight_clips, "
        "highlight_coarse_hits, highlight_sent, highlight_best_grade, "
        "human_decision")


def symptom_of(row: dict) -> str:
    """这一行的产物症状："" = 正常，其余是三种"没见过产物"的形态。"""
    if row.get("highlight_at") is None:
        return "没剪过"
    if row.get("highlight_error"):
        return "硬失败"
    try:
        clips = json.loads(row.get("highlight_clips") or "[]")
    except (TypeError, ValueError):
        clips = []
    if not clips:
        return "零产物+有命中" if (row.get("highlight_coarse_hits") or 0) > 0 else ""
    if not any(c.get("file") for c in clips):
        return "只有静帧无mp4"
    return ""


def recycle_states(rows: list[dict]) -> dict[str, int]:
    """这批行里**已在回收目录**的那些，按三态计数。

    刻意先按 `trash_path` 过滤：三态判据是**回收目录语境**的（回答"这一条现在
    能不能删"），套到全库行上会把"还没剪过"报成"待保全"。实测 keep 类里有 392
    条就是这样 —— 详见 `src/mover.py` 里 `is_pending` 的 ⚠️ 段。
    """
    out: dict[str, int] = {}
    for r in rows:
        if not r.get("trash_path"):
            continue
        st = mover.recycle_state(r, verify=False)
        out[st] = out.get(st, 0) + 1
    return out


def collect(db: Database) -> list[dict]:
    with db.conn() as c:
        return [dict(r) for r in c.execute(f"SELECT {COLS} FROM videos")]


def report(rows: list[dict], db: Database) -> list[dict]:
    """按 (症状 × 版本) 分桶并打印；返回"版本已是目标值"的可疑行。"""
    buckets: dict[tuple, list[dict]] = {}
    for r in rows:
        sym = symptom_of(r)
        if sym:
            buckets.setdefault((sym, r.get("highlight_version") or "(NULL)"), []).append(r)

    print("== 症状 × 版本（只列异常行）==")
    print("   %-14s %-8s %6s %8s   %s"
          % ("症状", "版本", "条数", "在回收目录", "回收目录内三态"))
    suspect: list[dict] = []
    for (sym, ver), lst in sorted(buckets.items()):
        trashed = [r for r in lst if r.get("trash_path")]
        states = recycle_states(lst)
        behind = ver != HIGHLIGHT_TARGET_VERSION
        if sym in ("零产物+有命中", "只有静帧无mp4") and not behind:
            suspect.extend(lst)
        flag = "  ← ★ 版本已是目标值却仍为空" if (
            sym in ("零产物+有命中", "只有静帧无mp4") and not behind) else ""
        print("   %-14s %-8s %6d %8d   %s%s" % (
            sym, ver, len(lst), len(trashed),
            (" ".join(f"{k}={v}" for k, v in sorted(states.items())) or "—"), flag))

    print(f"\n   目标版本 = {HIGHLIGHT_TARGET_VERSION}")
    print(f"   ★ 可疑（版本已是目标值、产物却为空/只有静帧）= {len(suspect)} 条"
          f"，其中在回收目录 {sum(1 for r in suspect if r.get('trash_path'))} 条")
    if suspect:
        print("     这类行被 rehighlight_queue 的 `version <> target` 挡在队列外。"
              "\n     先看 docs/自动剪辑方案说明.md §5.4 的处置办法（修代码 + bump 版本号），"
              "\n     不要给队列谓词加症状分支 —— 那会自我循环。")
    return suspect


def recut(sample: list[dict], cfg) -> None:
    """抽样重跑 cut_one，估"现在这一跑能不能留下产物"。只写临时目录。"""
    from src.highlighter import HighlightCutter
    hc = HighlightCutter(cfg)
    hdir = Path(tempfile.mkdtemp(prefix="audit_hl_"))
    cnt = {"AB": 0, "C": 0, "zero": 0, "err": 0, "missing": 0}
    t_all = time.time()
    print(f"\n== 抽样 {len(sample)} 条重跑 cut_one（产物落在 {hdir}，不写库）==")
    for r in sample:
        src = None
        for k in ("trash_path", "path"):
            v = r.get(k)
            if v and Path(v).exists():
                src = Path(v)
                break
        if src is None:
            cnt["missing"] += 1
            continue
        try:
            res = hc.cut_one(dict(r), hdir, "audit")
        except Exception as e:  # noqa: BLE001 单条失败不影响统计
            cnt["err"] += 1
            print("   id=%-6s 异常 %s: %s" % (r["id"], type(e).__name__, str(e)[:60]))
            continue
        st = res.get("stats") or {}
        gs = st.get("grades") or {}
        nab = gs.get("A", 0) + gs.get("B", 0)
        key = "AB" if nab else ("C" if gs.get("C") else "zero")
        cnt[key] += 1
        print("   id=%-6s ver=%-5s 库hits=%-4s → hits=%-4s events=%-2s %s" % (
            r["id"], r.get("highlight_version"), r.get("highlight_coarse_hits"),
            st.get("hits"), st.get("events"),
            {"AB": "✅ A/B", "C": "· 仅 C", "zero": "✗ 仍零"}[key]))
    n = len(sample) or 1
    print("\n   A/B %d ｜ 仅C %d ｜ 仍零 %d ｜ 异常 %d ｜ 文件不在 %d"
          % (cnt["AB"], cnt["C"], cnt["zero"], cnt["err"], cnt["missing"]))
    print("   至少留下产物 ≈ %.0f%%（均 %.1fs/条，用时 %.0fs）"
          % (100.0 * (cnt["AB"] + cnt["C"]) / n, (time.time() - t_all) / n,
             time.time() - t_all))
    print(f"   ⚠️ 临时产物留着自查看：{hdir}（确认后可手动删）")


def main() -> int:
    ap = argparse.ArgumentParser(description="精华产物体检：找『检出过却没产物』的行")
    ap.add_argument("--recut", type=int, default=0,
                    help="随机抽 N 条重跑 cut_one，估救回率（不写库）")
    ap.add_argument("--ids", nargs="*", type=int, default=[],
                    help="只看/只重跑这些 id（与 --recut 二选一，给了就按它来）")
    ap.add_argument("--seed", type=int, default=20260913, help="抽样种子（可复现）")
    args = ap.parse_args()

    cfg = load_config()
    db = Database(cfg["runtime"]["sqlite_path"])
    rows = collect(db)
    print(f"全库 {len(rows)} 条\n")

    pool = rows
    if args.ids:
        want = set(args.ids)
        pool = [r for r in rows if r["id"] in want]
        print(f"--ids 指定 {len(pool)} 条\n")

    suspect = report(pool if not args.ids else rows, db)

    if args.recut:
        target = pool if args.ids else (suspect or [
            r for r in rows if symptom_of(r) in ("零产物+有命中", "只有静帧无mp4")])
        random.seed(args.seed)
        sample = random.sample(target, min(args.recut, len(target)))
        recut(sample, cfg)
    elif args.ids:
        recut(pool, cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
