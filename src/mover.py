"""
移动执行 —— 程序对源数据的唯一写操作

职责边界：**程序永不删除**。这里只做 move，且必须同盘（跨盘 move 会退化成复制）。

分档规则（见 config.verdict）
----------------------------
A_高置信  score <= trash_threshold 且无目标
B_低置信  其余经人工确认的（存在接近阈值的疑似目标等“模型没敢确定”的情形）

不移动：解不开或分析出错的片段 —— 默认保留，绝不因技术失败而误删。
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .archiver import same_drive

TIER_A = "A"
TIER_B = "B"
TIER_NAMES = {TIER_A: "A_高置信_可批量删除", TIER_B: "B_低置信_建议抽查"}


def release_params(cfg: Dict) -> tuple[float, bool]:
    """放行门槛 + 「必须无人物信号」开关 —— 一键执行与审核台共用，避免两处口径漂移。

    返回 (release_threshold, release_require_no_person)。
    """
    vd = cfg.get("verdict", {}) or {}
    return (float(vd.get("release_threshold", 30)),
            bool(vd.get("release_require_no_person", True)))


def decide_tier(row, cfg: Dict) -> str:
    """根据 AI 结论与两级一致性决定回收档位。"""
    trace_score = row["retention_score"] or 0
    near_miss = False
    try:
        import json
        trace = json.loads(row["decision_trace"] or "{}")
        near_miss = bool(trace.get("near_miss"))
    except Exception:  # noqa: BLE001
        pass

    # 存在未达阈值的疑似目标 → 降为低置信（漏检高发区，不进可批量删除档）
    if near_miss:
        return TIER_B
    # 无目标且分数为 0 → 高置信
    if trace_score <= cfg.get("verdict", {}).get("trash_threshold", 0):
        return TIER_A
    return TIER_B


def trash_target_path(trash_root: str, tier: str, row) -> Path:
    """构造回收路径，**保留原目录结构**，杜绝重名覆盖。"""
    year = row["year"] or "_未分类"
    rel_dir = row["rel_dir"] or ""
    base = Path(trash_root) / TIER_NAMES[tier] / str(year) / rel_dir
    return base / (row["filename"] or Path(row["path"]).name)


def move_one(db, row, cfg: Dict, tier: str,
             log: Optional[Callable[[str], None]] = None) -> Tuple[bool, str]:
    """移动单个文件到分级回收目录。返回 (成功, 消息)。"""
    src = Path(row["path"])
    trash_root = cfg.get("trash_root", "")
    if not trash_root:
        return False, "未配置 trash_root"
    if not src.exists():
        return False, f"源文件不存在：{src}"
    if not same_drive(src, trash_root):
        return False, ("回收目录与源不在同一物理盘，跨盘 move 等于复制 10TB。"
                       "请把 trash_root 设置到源盘上。")

    dst = trash_target_path(trash_root, tier, row)
    if dst.exists():
        # 极端情况：同名文件已存在，加时间戳后缀，绝不覆盖
        stem, suffix = dst.stem, dst.suffix
        dst = dst.with_name(f"{stem}_{int(time.time())}{suffix}")

    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
    except Exception as e:  # noqa: BLE001
        return False, f"移动失败：{type(e).__name__}: {e}"

    db.set_human_decision(row["id"], "trash", trash_path=str(dst), trash_tier=tier)
    db.log_action(row["id"], "move", str(src), str(dst), row["batch_id"])
    if log:
        log(f"已移动 [{tier}] {src.name}")
    return True, str(dst)


def undo_move(db, row, log: Optional[Callable[[str], None]] = None) -> Tuple[bool, str]:
    """撤销一次移动，把文件移回原位。"""
    tp = row["trash_path"]
    if not tp or not Path(tp).exists():
        return False, "回收路径无效或文件已不存在"
    src = Path(row["path"])
    src.parent.mkdir(parents=True, exist_ok=True)
    if src.exists():
        return False, f"原位置已存在同名文件，为避免覆盖未执行撤销：{src}"
    try:
        shutil.move(tp, str(src))
    except Exception as e:  # noqa: BLE001
        return False, f"撤销失败：{type(e).__name__}: {e}"
    db.set_human_decision(row["id"], "keep")
    db.log_action(row["id"], "undo", tp, str(src), row["batch_id"])
    if log:
        log(f"已撤销 {src.name}")
    return True, str(src)


def _reports_dir(cfg: Dict) -> Path:
    """清单与统计的落盘位置：优先回收目录下的 _reports（人工删除时顺手就能看到）。"""
    base = cfg.get("trash_root") or cfg.get("work_dir") or "."
    d = Path(base) / "_reports"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _gb(n: float) -> str:
    return f"{(n or 0) / (1024 ** 3):.2f}"


def write_reports(db, cfg: Dict, batch_id: Optional[int] = None) -> Dict[str, Any]:
    """生成「移动清单 CSV」+「空间统计 TXT」。

    人工删除前要看的两样东西都在这里：删哪些文件、能释放多少空间。
    """
    ts = time.strftime("%Y%m%d_%H%M%S")
    d = _reports_dir(cfg)
    csv_path = d / f"移动清单_{ts}.csv"
    db.export_csv(csv_path, batch_id)

    rep = db.space_report(batch_id)
    total = rep["total"]
    lines = [
        "监控视频清理 — 空间统计",
        f"生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"批次：{('#' + str(batch_id)) if batch_id else '全部'}",
        f"回收目录：{cfg.get('trash_root') or '(未配置)'}",
        "",
        f"{'年份':<10}{'文件数':>8}{'A档':>7}{'B档':>7}{'容量(GB)':>12}{'时长(小时)':>12}",
        "-" * 56,
    ]
    for y in rep["by_year"]:
        lines.append(
            f"{y['year']:<10}{y['n']:>8}{y.get('A', 0):>7}{y.get('B', 0):>7}"
            f"{_gb(y['bytes']):>12}{y['secs'] / 3600:>12.1f}"
        )
    lines += [
        "-" * 56,
        f"{'合计':<10}{total['n']:>8}{total.get('A', 0):>7}{total.get('B', 0):>7}"
        f"{_gb(total['bytes']):>12}{total['secs'] / 3600:>12.1f}",
        "",
        f"A_高置信_可批量删除：{total.get('A', 0)} 个 —— 可直接全选删除",
        f"B_低置信_建议抽查：{total.get('B', 0)} 个 —— 建议抽查后再删",
        "",
        "说明：文件只是被移动，尚未释放空间。删除本程序不做，请你手动完成。",
        "      移动可撤销（审核工作台 → 撤销移动 / 执行进度 → 按批次撤销）。",
    ]
    txt_path = d / f"空间统计_{ts}.txt"
    txt_path.write_text("\n".join(lines), encoding="utf-8")
    return {"csv": str(csv_path), "txt": str(txt_path), "report": rep}


def move_rows(db, rows: List, cfg: Dict,
              log: Optional[Callable[[str], None]] = None,
              limit: Optional[int] = None) -> Dict:
    """批量移动。

    limit=None  → 受 runtime.batch_size_limit 护栏（batch_size_limit，防误操作波及全库）
    limit<=0    → 不切片，整批走完（调用方已显式指定候选，如「无人」候选一键放行）
    """
    if limit is None:
        limit = int(cfg.get("runtime", {}).get("batch_size_limit", 500))
    ok = fail = 0
    errors: List[str] = []
    for row in (rows if limit <= 0 else rows[:limit]):
        tier = decide_tier(row, cfg)
        success, msg = move_one(db, row, cfg, tier, log)
        if success:
            ok += 1
        else:
            fail += 1
            errors.append(msg)
    if limit > 0 and len(rows) > limit:
        errors.append(f"本次仅处理前 {limit} 条，剩余 {len(rows) - limit} 条需再次执行")

    reports: Dict[str, str] = {}
    if ok:
        # 以第一条记录所属批次为准；跨批次时退化为全量清单
        bid = rows[0]["batch_id"] if rows and rows[0]["batch_id"] else None
        try:
            r = write_reports(db, cfg, bid)
            reports = {"csv": r["csv"], "txt": r["txt"]}
        except Exception as e:  # noqa: BLE001
            errors.append(f"清单生成失败（文件已成功移动，不影响结果）：{e}")
    return {"moved": ok, "failed": fail, "errors": errors[:20], "reports": reports}


def move_all(db, rows: List, cfg: Dict,
             log: Optional[Callable[[str], None]] = None) -> Dict:
    """一次性放行整批（不受 runtime.batch_size_limit 的 500 切片限制）。

    调用方已经显式指定了要放行的整批候选（如「无人」候选一键放行），逐条 move
    不再切片，全部完成后统一生成一份移动清单 + 空间统计。

    关于**音频守卫**：本函数刻意不拦"有音轨但尚未复核"的片段 —— 音频队列已放宽为
    「处置候选」，放行后仍能补复核、补归档（review_one 有 trash_path 回落）。
    与 highlighter._dispose_one（剪辑后自动移入）保持同一口径：都是"事后补"，
    不是"拦住"。唯一的代价是回收目录**手动清空**前必须把音频复核+归档跑完。
    """
    return move_rows(db, rows, cfg, log=log, limit=0)


def undo_batch(db, rows: List, log: Optional[Callable[[str], None]] = None) -> Dict:
    """按批次/文件集合批量撤销移动。"""
    ok = fail = 0
    errors: List[str] = []
    for row in rows:
        success, msg = undo_move(db, row, log)
        if success:
            ok += 1
        else:
            fail += 1
            errors.append(msg)
    return {"undone": ok, "failed": fail, "errors": errors[:20]}
