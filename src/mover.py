r"""
移动执行 —— 程序对源数据的唯一写操作

职责边界：**程序永不删除**。这里只做 move，且必须同盘（跨盘 move 会退化成复制）。

分档规则（见 config.verdict）
----------------------------
A_高置信  score <= trash_threshold 且无目标
B_低置信  其余经人工确认的（存在接近阈值的疑似目标等“模型没敢确定”的情形）

不移动：解不开或分析出错的片段 —— 默认保留，绝不因技术失败而误删。

回收目录内的子档：回答「这一条我现在能删吗」
--------------------------------------------
回收档位（A_高置信 / B_低置信）说的是「AI 判不判得出能不能整批删」，与下面这
一层**正交**。真正回答"能不能删"的是原片自己的**保全状态**，共三态：

    clipped  有 A/B 产物           → <回收目录>\已剪辑\<档位>\<年>\<原目录>\<文件>
                                    精华已落袋，验完产物即可删
    pending  **还没剪完**          → <回收目录>\待保全\<档位>\<年>\<原目录>\<文件>
                                    别删！删了精华就真没了
    plain    剪过、没有 A/B 产物    → <回收目录>\<档位>\<年>\<原目录>\<文件>
                                    可直接删（不进任何子档）

子档名**刻意不带字母** —— 带字母就会出现
「已剪辑_AB / A_高置信_可批量删除 / ...」这种两个 A 两个 B 套在一起、
半年后自己也读不出来的目录。

两个判据都必须**现算**，不能看字段：
- `has_ab_clip()` 现算 highlight_clips JSON（highlight_best_grade 只有新写入的
  行才有值，老行全 NULL）
- `is_pending()` 用的就是 db 两条剪辑队列的谓词（见 db.pending_highlight_ids），
  **不另立口径** —— 计数方法与队列方法必须同口径，本项目踩过这个坑
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .archiver import same_drive
from .db import HIGHLIGHT_TARGET_VERSION

TIER_A = "A"
TIER_B = "B"
TIER_NAMES = {TIER_A: "A_高置信_可批量删除", TIER_B: "B_低置信_建议抽查"}

# 已剪出 A/B 产物的原片在回收目录内的顶层子档（人工清理时一眼可辨）。
# 不带字母是刻意的，理由见模块 docstring —— 别改成「已剪辑_AB」。
CLIPPED_SUBDIR = "已剪辑"

# 「还没剪完」的原片子档。存在的唯一理由：**别让机主在手动删除时，
# 把精华还没落袋的原片一起删掉**。跑完保全（剪辑/重剪）后它会自然清空。
PENDING_SUBDIR = "待保全"

# 构成「已剪辑」标记的精华级别。C 级是**档案**不是待办（不进人工审核队列），
# 所以只有 A/B 才算「有待你验的产物」。
AB_GRADES: Tuple[str, ...] = ("A", "B")

# ---------------------------------------------------------------- 保全状态三态
STATE_CLIPPED = "clipped"   # 有 A/B 产物
STATE_PENDING = "pending"   # 还没剪完 → 不许删
STATE_PLAIN = "plain"       # 剪过、无 A/B 产物 → 可删

# plain 的空串 = 不进子档，直接落在档位目录下（存量 8 千个文件已在那儿，
# 没必要为了"对称"把它们全搬一层）。
SUBDIR_BY_STATE: Dict[str, str] = {
    STATE_CLIPPED: CLIPPED_SUBDIR,
    STATE_PENDING: PENDING_SUBDIR,
    STATE_PLAIN: "",
}

STATE_LABELS: Dict[str, str] = {
    STATE_CLIPPED: "已剪辑（有 A/B 产物，验完即可删）",
    STATE_PENDING: "待保全（还没剪完，别删）",
    STATE_PLAIN: "无精华产物（可直接删）",
}

# 归属于回收目录顶层的全部子档名（搬迁/统计用）。顺序 = 展示顺序。
ALL_SUBDIRS: Tuple[str, ...] = (CLIPPED_SUBDIR, PENDING_SUBDIR)


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


def clips_of(row) -> List[Dict[str, Any]]:
    """从行里取精华产物清单。JSON 解析失败一律当空 —— 坏的 clips 不该拦住移动。"""
    try:
        raw = row["highlight_clips"]
    except (KeyError, IndexError, TypeError):
        return []
    if not raw:
        return []
    try:
        clips = json.loads(raw)
    except (TypeError, ValueError):
        return []
    return clips if isinstance(clips, list) else []


def ab_clip_files(clips: List[Dict[str, Any]]) -> List[Path]:
    """clips 里 A/B 级、且真切出了 mp4 的产物路径。

    `file` 为 None 表示当时切段失败（只剩代表帧），不算可验的产物。
    """
    out: List[Path] = []
    for c in clips or []:
        if not isinstance(c, dict):
            continue
        if str(c.get("grade") or "") in AB_GRADES and c.get("file"):
            out.append(Path(str(c["file"])))
    return out


def has_ab_clip(clips: List[Dict[str, Any]], verify: bool = True) -> bool:
    """该原片是否已剪出 A/B 级产物 —— 这就是「已剪辑」标记的判据。

    verify=True（默认）：还要求产物文件**此刻仍在**。这个标记对用户的承诺是
    "你在这堆里能验产物"，指向一个已经不存在的文件等于给一个无法兑现的承诺。
    代价是每条各一次 Path.exists()，所以它只用在真要做决定的地方
    （move_one / rehome_clipped）。

    verify=False：只认库里的记录，**不碰文件系统**。给 UI 现状统计用 ——
    Streamlit 会把所有页签都渲染一遍，设置页每次交互都扫 1600+ 个 G: 上的文件
    纯属白等。两个分支共用 ab_clip_files()，差异只有"查不查文件"这一条。
    """
    files = ab_clip_files(clips)
    if not files:
        return False
    return any(p.exists() for p in files) if verify else True


def is_pending(row, clips: Optional[List[Dict[str, Any]]] = None,
               pending_ids: Optional[set] = None) -> bool:
    """该原片「还没剪完」—— 精华还没落袋，且没有"确认无产物"的结论。

    这是「待保全」子档的判据，也是唯一一处对它的定义。三种情况：

    1. 从未剪过（`highlight_at IS NULL`）—— 包括硬失败被踢出队列的那些
       （`mark_highlight_error` 只置 error，不动 highlight_at）。
    2. 硬失败（`highlight_error` 非空）—— 在 reset_highlight() 之前它既不
       进 highlight_queue 也不进 rehighlight_queue，但对"能不能删"来说它
       和排队中的行一样是**没保全**的。
    3. 剪过但版本落后，且症状命中重剪队列（粗扫看到过却零产物 / C 级只有静帧）。

    **第 3 条的"版本落后"是刻意的门，不是遗漏**：症状（零产物 / 只有静帧）
    本身不能单独当判据 —— 重剪后仍然零产物的行会被反复选中、队列永不收敛。
    版本号提供了收敛性（剪完版本跟上即退出），所以发现"某版口径曾静默吃掉产物"
    时的正确做法是**修代码 + bump 版本号**，而不是放宽这里的判据。

    ⚠️ 本模块的 HIGHLIGHT_TARGET_VERSION 直接 import 自 db，而剪辑器写库用的是
    `config.highlight.version` —— 两者必须相等（selftest 钉住）。config 更新、
    db 没跟上的话，每一条新剪的行都会被判成"待保全"，回收目录永远不会归零。

    pending_ids：db.pending_highlight_ids() 一次算好的集合。批量路径必须传它
    （否则每条都要跑一次队列 SQL）；move_one 这种单条路径回落到行内字段判断，
    两条分支的等价性由 selftest 的不变量断言钉住。

    ⚠️ 行内判断是**队列谓词在「回收目录」这个域上的等价实现**，不是全局等价 ——
    实测（2026-09-13）：789 条 `human_decision='keep'` 里有 392 条从未剪过，
    行内判断说"待保全"、队列说"不在队里"。两处发散是**刻意不修**的：
    `keep` 是人工已经表过态的行，"还没剪完"对它没有意义；而队列那侧（
    `rehighlight_queue` 不过滤 human_decision）真去重剪一次也只是白跑（
    `_dispose_one` 见到 keep 会返回 not_disposed，不会把它重新移进回收目录）。
    所以：**只在回收目录语境里用这个判据**，别拿它去筛全库。
    """
    def g(key: str, default: Any = None) -> Any:
        try:
            return row[key]
        except (KeyError, IndexError, TypeError):
            return default

    if pending_ids is not None:
        rid = g("id")
        return rid in pending_ids if rid is not None else False

    if g("highlight_error"):
        return True
    if g("highlight_at") is None:
        return True

    if (g("highlight_version") or "hlv1") != HIGHLIGHT_TARGET_VERSION:
        cl = clips_of(row) if clips is None else clips
        dropped = (not cl) and (g("highlight_coarse_hits") or 0) > 0
        thumb_only = bool(cl) and not any(c.get("file") for c in cl)
        if dropped or thumb_only:
            return True
    return False


def recycle_state(row, clips: Optional[List[Dict[str, Any]]] = None,
                  verify: bool = True,
                  pending_ids: Optional[set] = None) -> str:
    """把一行原片判成三态之一（STATE_CLIPPED / STATE_PENDING / STATE_PLAIN）。

    顺序有讲究：**先判有产物**。一条行可以被重剪队列收着（version 落后）却
    已经剪出 A/B mp4 —— 那时它就是"已剪辑"，不该因为"版本旧"被塞进待保全。
    """
    cl = clips_of(row) if clips is None else clips
    if has_ab_clip(cl, verify=verify):
        return STATE_CLIPPED
    if is_pending(row, cl, pending_ids=pending_ids):
        return STATE_PENDING
    return STATE_PLAIN


def subdir_of(row, clips: Optional[List[Dict[str, Any]]] = None,
              verify: bool = True,
              pending_ids: Optional[set] = None) -> str:
    """这一行该落在回收目录顶层的哪个子档（"" = 不建子档）。"""
    return SUBDIR_BY_STATE[recycle_state(row, clips, verify, pending_ids)]


def trash_target_path(trash_root: str, tier: str, row,
                      subdir: str = "") -> Path:
    """构造回收路径，**保留原目录结构**，杜绝重名覆盖。

    subdir 非空时先插一层子档名：
        <回收目录>\\已剪辑\\<档位>\\<年>\\<原目录>\\<文件>
        <回收目录>\\待保全\\<档位>\\<年>\\<原目录>\\<文件>
    空串则不插层，落 <回收目录>\\<档位>\\<年>\\<原目录>\\<文件>。
    """
    year = row["year"] or "_未分类"
    rel_dir = row["rel_dir"] or ""
    base = Path(trash_root)
    if subdir:
        base = base / subdir
    base = base / TIER_NAMES[tier] / str(year) / rel_dir
    return base / (row["filename"] or Path(row["path"]).name)


def current_subdir(trash_path: str, trash_root: str) -> str:
    """从回收路径里反解出它现在待在哪个子档（"" = 直接在档位目录下）。

    只认**紧邻回收目录根**的那一层 —— 原目录树里如果恰好有个同名文件夹，
    不能被误认成子档。
    """
    if not trash_path:
        return ""
    try:
        parts = Path(trash_path).relative_to(Path(trash_root)).parts
    except (ValueError, OSError):
        return ""
    return parts[0] if parts and parts[0] in ALL_SUBDIRS else ""


def move_one(db, row, cfg: Dict, tier: str,
             log: Optional[Callable[[str], None]] = None,
             clips: Optional[List[Dict[str, Any]]] = None,
             state: Optional[str] = None) -> Tuple[bool, str]:
    """移动单个文件到分级回收目录。返回 (成功, 消息)。

    clips：本次**新剪出**的产物清单。库里的 highlight_clips 也是对的了
    （mark_highlight 先执行），但显式传能覆盖"库还没写入"的调用点与测试替身。
    state：覆写保全状态。不传（正常情况）则**回库取新鲜行**再判 —— 传进来的
    row 可能是剪辑前的旧快照，拿它判状态会把"刚剪完、只有 C 级"误判成
    "还没剪过"错放进 `待保全`（真实踩过一次）。
    """
    src = Path(row["path"])
    trash_root = cfg.get("trash_root", "")
    if not trash_root:
        return False, "未配置 trash_root"
    if not src.exists():
        return False, f"源文件不存在：{src}"
    if not same_drive(src, trash_root):
        return False, ("回收目录与源不在同一物理盘，跨盘 move 等于复制 10TB。"
                       "请把 trash_root 设置到源盘上。")

    if state is None:
        # ⚠️ 传进来的 row 可能是**剪辑前的旧快照**（highlighter._dispose_one 手里
        # 那份的 highlight_at 还是 NULL）—— 拿它判状态会把"刚剪完、只有 C 级"
        # 误判成"还没剪过"，错放进 `待保全`。真实踩过一次。
        # 所以这里**回库取新鲜行**，而不是信任调用方手里那份。
        # 剪辑链路里 mark_highlight() 一定先于本函数执行，库里的值就是最新的。
        try:
            fresh = db.get_video(row["id"])
        except Exception:  # noqa: BLE001
            fresh = None
        state = recycle_state(fresh if fresh is not None else row, clips=clips)
    subdir = SUBDIR_BY_STATE[state]
    dst = trash_target_path(trash_root, tier, row, subdir=subdir)
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
        tag = f"［{subdir}］" if subdir else ""
        log(f"已移动 [{tier}]{tag} {src.name}")
    return True, str(dst)


def align_one(db, row, cfg: Dict,
              log: Optional[Callable[[str], None]] = None,
              pending_ids: Optional[set] = None,
              verify: bool = True) -> Tuple[bool, str]:
    """把**已经在**回收目录里的原片挪到它当前保全状态对应的子档（存量对齐）。

    与 move_one 的区别：源路径是 trash_path 而不是源盘的 path，且**不改处置结论**
    —— 只把同盘文件从一个子目录重命名到另一个，档位（trash_tier）原样保留，
    撤销移动仍然照常可用（undo_move 只认 trash_path）。

    双向的：`已剪辑 → 待保全`（产物文件丢了/被清理了）和 `待保全 → 已剪辑`
    （补剪出了 A/B）都能走这里，所以它同时是「补跑保全之后再对齐一次」的入口。
    """
    tp = row["trash_path"]
    if not tp:
        return False, "该行没有回收路径（只是逻辑标记，文件仍在原位）"
    src = Path(tp)
    if not src.exists():
        return False, f"回收路径上的文件已不存在：{src}"

    trash_root = cfg.get("trash_root", "")
    if not trash_root:
        return False, "未配置 trash_root"
    if not same_drive(src, trash_root):
        return False, "回收目录与文件不在同一物理盘，无法就地搬迁"

    state = recycle_state(row, verify=verify, pending_ids=pending_ids)
    want = SUBDIR_BY_STATE[state]
    have = current_subdir(tp, trash_root)
    if have == want:
        return False, (f"已在「{want}」内，无需搬迁" if want else "本就不进子档，无需搬迁")

    tier = row["trash_tier"] or TIER_B
    dst = trash_target_path(trash_root, tier, row, subdir=want)
    if dst.exists():
        stem, suffix = dst.stem, dst.suffix
        dst = dst.with_name(f"{stem}_{int(time.time())}{suffix}")

    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
    except Exception as e:  # noqa: BLE001
        return False, f"搬迁失败：{type(e).__name__}: {e}"

    db.set_human_decision(row["id"], "trash", trash_path=str(dst), trash_tier=tier)
    db.log_action(row["id"], "rehome", str(src), str(dst), row["batch_id"])
    if log:
        frm = f"「{have}」" if have else "档位目录"
        to = f"「{want}」" if want else "档位目录"
        log(f"{frm} → {to}  {src.name}")
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


# ------------------------------------------------------------------- 回收子档
def recycle_stats(db, cfg: Optional[Dict] = None, verify: bool = False,
                  limit: int = 200_000) -> Dict[str, Any]:
    """回收目录的「保全状态」现状 —— UI 显示、对齐前后核对都用它。

    返回：
      in_trash  {状态: 回收目录内（trash_path 非空）该状态的条数}
      to_align  {状态: 其中没待在正确子档的条数}（= 下次对齐要搬的量）
      total     回收目录内的原片总数

    三个数的关系就是"打开回收目录该看到什么"：
        已剪辑  → 验完产物即可删
        待保全  → ★ 别删，精华还没落袋
        无精华产物 → 可直接删

    verify 默认 False（只按库里记录，不碰文件系统）—— UI 每次 rerun 都会调它，
    详见 has_ab_clip 的说明。
    """
    trash_root = (cfg or {}).get("trash_root", "") or ""
    st: Dict[str, Any] = {
        "in_trash": {k: 0 for k in SUBDIR_BY_STATE},
        "to_align": {k: 0 for k in SUBDIR_BY_STATE},
        "total": 0,
    }
    for row in db.recycle_scan_rows(limit=limit):
        state = recycle_state(row, verify=verify)
        st["in_trash"][state] += 1
        st["total"] += 1
        if current_subdir(row["trash_path"] or "", trash_root) != SUBDIR_BY_STATE[state]:
            st["to_align"][state] += 1
    return st


def align_subdirs(db, cfg: Dict, log: Optional[Callable[[str], None]] = None,
                  limit: int = 200_000,
                  verify: bool = True) -> Dict[str, Any]:
    """把所有**已经在回收目录里**的原片对齐到当前保全状态对应的子档。

    只管回收目录内的存量；仍在原位、尚未处置的不用搬 —— 它们下次走 move_one
    时会自己落对地方。幂等：已在正确子档的直接跳过，可反复重跑。

    典型用法（顺序很重要）：
        ① 跑完剪辑 + 重剪（把精华尽量落袋）
        ② 再跑一次本函数 —— 新剪出 A/B 的会从 `待保全` 挪进 `已剪辑`，
           `待保全` 归零就是「可以放心手动删了」的信号
    """
    pending_ids = db.pending_highlight_ids()
    trash_root = cfg.get("trash_root", "")
    ok = skip = fail = 0
    moved_by_state: Dict[str, int] = {k: 0 for k in SUBDIR_BY_STATE}
    errors: List[str] = []
    for row in db.recycle_scan_rows(limit=limit):
        if not (row["human_decision"] == "trash" and row["trash_path"]):
            continue
        # pending_ids 只在**这里**（一次性、离线、有闲工夫）用来核对行内判断；
        # 不一致就说明两套判据漂了，宁可停下也不要照着错的状态搬文件。
        by_queue = recycle_state(row, verify=verify, pending_ids=pending_ids)
        by_row = recycle_state(row, verify=verify)
        if by_queue != by_row:
            errors.append(f"id={row['id']}: 行内判断与队列口径不一致"
                          f"（队列={by_queue} / 行内={by_row}）—— 已跳过")
            fail += 1
            continue
        state = by_row
        if current_subdir(row["trash_path"], trash_root) == SUBDIR_BY_STATE[state]:
            skip += 1
            continue
        success, msg = align_one(db, row, cfg, log, verify=verify)
        if success:
            ok += 1
            moved_by_state[state] += 1
        else:
            fail += 1
            errors.append(f"id={row['id']}: {msg}")
    return {"moved": ok, "skipped": skip, "failed": fail,
            "by_state": moved_by_state, "errors": errors[:20]}
