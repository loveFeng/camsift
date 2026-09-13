#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""存量对齐：把回收目录里的原片按**保全状态**归入对应子档。

为什么需要这一步（不是"顺手改今后的行为"就完事）
------------------------------------------------
回收目录（`G:\\need-del`）里现在混着三类东西，而机主是**手动删除**的 ——
打开文件夹时唯一要回答的问题是「这一条我现在能删吗」：

    ┌ 已剪辑        有 A/B 产物        → 验完产物即可删
    ├ 待保全        ★ 精华还没落袋    → 一个都别删
    └ （档位目录下） 剪过、无 A/B 产物  → 可直接删

不按"已剪辑 / 未剪辑"两分，是因为"没剪出 A/B"的那一大堆里**六成其实可以闭眼删**
（机器跑完确认无命中），把危险的那两千多条淹在里面，等于没有保护。

判据只有一处（`mover.recycle_state`），本脚本不重复实现：
    clipped  剪出 A/B 级 mp4 且产物此刻仍在
    pending  highlight_at 为空（含硬失败）或版本落后且命中重剪症状
    plain    其余

安全边界
--------
- **只 move，不删除**（程序对源数据的唯一写操作，与 src/mover.py 同口径）
- 同盘搬迁 = 改目录项，不复制数据（跨盘直接拒绝）
- 默认 **dry-run**：只打印计划，不动任何文件；确认后加 `--apply`
- **幂等**：已在正确子档的直接跳过，可反复重跑
- **先核对再动手**：`align_subdirs` 会拿「行内判断」和「队列口径」对一遍，
  不一致的行直接跳过并报出来 —— 宁可停下，也不照着错的状态搬文件

推荐用法（顺序很重要）
----------------------
    # ① 先跑完保全（工作台：画面精华剪辑 → 按新口径重剪）
    # ② 再对齐子档：
    cd E:/code/video-search
    ./.venv/Scripts/python.exe scripts/align_subdirs.py            # 只看计划
    ./.venv/Scripts/python.exe scripts/align_subdirs.py --sample 12
    ./.venv/Scripts/python.exe scripts/align_subdirs.py --apply    # 真搬

对齐完可以在工作台「设置与扫描」页的「回收目录三堆」卡里核对（同一套判据）。
`待保全` 归零 = 可以放心手动清空回收目录。
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import mover  # noqa: E402
from src.config import load_config  # noqa: E402
from src.db import Database  # noqa: E402


def _collect(db, cfg: Dict[str, Any], limit: int) -> List[Dict[str, Any]]:
    """挑出「已在回收目录 + 没待在正确子档」的行，算出新旧路径。"""
    trash_root = cfg.get("trash_root", "")
    out: List[Dict[str, Any]] = []
    for row in db.recycle_scan_rows(limit=limit):
        tp = row["trash_path"]
        if not (row["human_decision"] == "trash" and tp):
            continue
        state = mover.recycle_state(row, verify=True)
        want = mover.SUBDIR_BY_STATE[state]
        have = mover.current_subdir(tp, trash_root)
        if have == want:
            continue                      # 已在正确子档 → 幂等跳过
        out.append({
            "id": row["id"],
            "state": state,
            "from": have or "(档位目录)",
            "to": want or "(档位目录)",
            "src": tp,
            "dst": str(mover.trash_target_path(
                trash_root, row["trash_tier"] or mover.TIER_B, row, subdir=want)),
            "tier": row["trash_tier"] or "?",
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="把回收目录里的原片对齐到保全状态对应的子档")
    ap.add_argument("--apply", action="store_true", help="真执行搬迁（默认只打印计划）")
    ap.add_argument("--sample", type=int, default=8, help="打印多少条新旧路径样例（默认 8）")
    ap.add_argument("--limit", type=int, default=200_000, help="扫描上限")
    args = ap.parse_args()

    cfg = load_config()
    trash_root = (cfg.get("trash_root") or "").strip()
    if not trash_root:
        print("✗ 配置里没有 trash_root，无法搬迁。")
        return 2

    db = Database(cfg["runtime"]["sqlite_path"])
    stats = mover.recycle_stats(db, cfg, verify=True, limit=args.limit)

    print("=" * 72)
    print("回收目录 · 保全状态对齐")
    print("=" * 72)
    print(f"回收目录      {trash_root}")
    print(f"子档          {mover.CLIPPED_SUBDIR} / {mover.PENDING_SUBDIR}"
          f"  （plain 不进子档，直接落在档位目录下）")
    print()
    print(f"{'状态':<12}{'回收目录内':>10}{'待归位':>10}   说明")
    print("-" * 72)
    for st in (mover.STATE_CLIPPED, mover.STATE_PENDING, mover.STATE_PLAIN):
        print(f"{st:<12}{stats['in_trash'][st]:>10}{stats['to_align'][st]:>10}   "
              f"{mover.STATE_LABELS[st]}")
    print("-" * 72)
    n_hold = stats["in_trash"][mover.STATE_PENDING]
    print(f"{'合计':<12}{stats['total']:>10}{sum(stats['to_align'].values()):>10}")
    print()
    if n_hold:
        print(f"⚠️  待保全 {n_hold} 条 —— 精华还没落袋。**先把剪辑 / 重剪跑完再清空回收目录。**")
    else:
        print("✅ 待保全为 0 —— 回收目录里已经没有「精华没落袋」的原片了。")

    plan = _collect(db, cfg, args.limit)
    if not plan:
        print("\n没有需要搬迁的文件 —— 已是目标状态（幂等）。")
        return 0

    by_state = Counter(p["state"] for p in plan)
    by_move = Counter((p["from"], p["to"]) for p in plan)
    print(f"\n本次要搬 {len(plan)} 条")
    print("  按状态   " + "  ".join(f"{k}:{v}" for k, v in sorted(by_state.items())))
    print("  按走向   " + "  ".join(f"{a}→{b}:{v}" for (a, b), v in sorted(by_move.items())))

    if args.sample:
        print(f"\n新旧路径样例（{min(args.sample, len(plan))} 条）：")
        for p in plan[:args.sample]:
            print(f"  id={p['id']}  {p['state']}  [{p['tier']}]  {p['from']} → {p['to']}")
            print(f"    旧 {p['src']}")
            print(f"    新 {p['dst']}")

    if not args.apply:
        print("\n" + "-" * 72)
        print("这是**计划**，没有动任何文件。确认无误后加 --apply 执行：")
        print("  ./.venv/Scripts/python.exe scripts/align_subdirs.py --apply")
        return 0

    print("\n开始搬迁（同盘 move，只改目录项）…")

    def _log(msg: str) -> None:
        print("  " + msg)

    res = mover.align_subdirs(db, cfg, log=_log, limit=args.limit)
    print("-" * 72)
    print(f"归位 {res['moved']} 条 / 跳过 {res['skipped']} 条 / 失败 {res['failed']} 条")
    bs = res["by_state"]
    print(f"  落点：{mover.CLIPPED_SUBDIR} {bs[mover.STATE_CLIPPED]} / "
          f"{mover.PENDING_SUBDIR} {bs[mover.STATE_PENDING]} / "
          f"无产物 {bs[mover.STATE_PLAIN]}")
    if res["errors"]:
        print("失败明细（前 20 条）：")
        for e in res["errors"]:
            print("  " + e)
    after = mover.recycle_stats(db, cfg, verify=True, limit=args.limit)
    print(f"复核：待归位 {sum(after['to_align'].values())} 条（应为 0）")
    return 0 if res["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
