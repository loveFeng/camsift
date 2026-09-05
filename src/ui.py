"""
Streamlit 审核工作台

五个页签
--------
1. 审核工作台  逐条审核，展示完整判定依据链，支持「误判保留 → 自动归因」
2. 批量浏览    九宫格 + 多选，批量处置
3. 执行进度    批次进度、扫描/分析吞吐、状态统计
4. 误判分析    误判原因分布 + 阈值调整建议（把人工审核变成调参依据）
5. 设置与扫描  路径配置、启动扫描/分析、导出清单

职责边界：本界面**不提供任何删除功能**，只有「移动到分级回收目录」与「撤销」。
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import feedback as fbmod  # noqa: E402
from src import hotkeys  # noqa: E402
from src import jobs as jobsmod  # noqa: E402
from src import mover  # noqa: E402
from src.analyzer import Analyzer  # noqa: E402
from src import config as cfgmod  # noqa: E402
from src.config import THUMB_DIR, load_config, save_config  # noqa: E402
from src.db import Database  # noqa: E402

LEVEL_ICON = {"keep": "🟢", "trash": "⚪", "warn": "🟡", "neutral": "⚫"}
VERDICT_TEXT = {"trash": "建议清理", "review": "需人工确认", "keep": "建议保留"}
PAGE_SIZE = 24  # 批量浏览每页条目数（4 列 × 6 行）

# st.fragment 需要 Streamlit >= 1.37；老版本优雅退化为普通函数（不自动刷新）
try:
    _fragment = st.fragment
except AttributeError:  # pragma: no cover
    def _fragment(**_kw):
        def deco(f):
            return f
        return deco


# --------------------------------------------------------------------------- 基础
@st.cache_resource
def get_db(path: str):
    return Database(path)


def cfg_now():
    return load_config()


def thumb_of(video_id: int) -> Path | None:
    p = THUMB_DIR / f"{video_id}_grid.jpg"
    return p if p.exists() else None


def open_in_player(path: str):
    try:
        if platform.system() == "Windows":
            os.startfile(path)  # noqa: S606
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])
    except Exception as e:  # noqa: BLE001
        st.warning(f"无法调用系统播放器：{e}")


def make_browser_preview(row, cfg: Dict) -> Path | None:
    """把 H.265 转成浏览器可播的 H.264 预览（H.265 在多数浏览器要靠硬件解码）。"""
    from shutil import which
    ffmpeg = cfg.get("runtime", {}).get("ffmpeg_path") or which("ffmpeg") or "ffmpeg"
    out = THUMB_DIR / f"{row['id']}_preview.mp4"
    if out.exists():
        return out
    src = Path(row["path"])
    if not src.exists():
        return None
    cmd = [ffmpeg, "-v", "error", "-i", str(src), "-t", "30",
           "-vf", "scale=960:-2", "-c:v", "libx264", "-preset", "ultrafast",
           "-crf", "30", "-movflags", "+faststart", "-an", "-y", str(out)]
    try:
        subprocess.run(cmd, capture_output=True, timeout=180)
    except Exception:  # noqa: BLE001
        return None
    return out if out.exists() else None


# --------------------------------------------------------------------------- 判定依据
def render_reasons(row):
    """渲染判定依据链 —— 回答"为什么判定要删/要留"。"""
    trace = {}
    try:
        trace = json.loads(row["decision_trace"] or "{}")
    except Exception:  # noqa: BLE001
        pass

    score = trace.get("score", row["retention_score"])
    verdict = trace.get("verdict", row["verdict"])

    c1, c2 = st.columns([1, 3])
    with c1:
        st.metric("保留价值分", f"{score:g}")
        st.markdown(f"**判定：{VERDICT_TEXT.get(verdict, verdict)}**")
    with c2:
        st.info(trace.get("summary", "（无判定摘要）"))

    st.markdown("##### 判定依据逐项拆解")
    reasons = trace.get("reasons", [])
    if not reasons:
        st.caption("该片段尚未生成判定依据")
        return trace

    for r in reasons:
        icon = LEVEL_ICON.get(r.get("level", "neutral"), "⚫")
        contrib = r.get("contribution", 0)
        st.markdown(
            f"{icon} **{r.get('factor','')}**　`{r.get('result','')}`　"
            f"`{contrib:+g} 分`"
        )
        st.caption(r.get("detail", ""))

    # 未达阈值的原始检出 —— 误判的关键线索
    raw = []
    try:
        raw = json.loads(row["raw_detections"] or "[]")
    except Exception:  # noqa: BLE001
        pass
    near = [d for d in raw if d["conf"] < 0.5]
    if near:
        with st.expander(f"⚠️ 存在 {len(near)} 个未达阈值的疑似目标（误判高发线索）", expanded=False):
            st.dataframe(
                [{"类别": d["class"], "置信度": d["conf"], "画面占比": f"{d.get('area_ratio',0)*100:.2f}%"}
                 for d in near[:10]],
                hide_index=True, use_container_width=True,
            )
    return trace


# --------------------------------------------------------------------------- 后台任务面板
@_fragment(run_every="1s")
def job_panel(job, title: str, key: str):
    """后台任务实时面板。

    用 fragment 每秒局部刷新，不阻塞页面、不打断其它页签的操作。
    任务跑在独立线程里，即使关掉浏览器也不会中断（只要 Streamlit 进程还在）。
    """
    st.markdown(f"**{title}**")
    if job.running:
        c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
        c1.caption(f"正在处理：`{job.current or '初始化…'}`")
        c2.metric("已处理", job.processed)
        c3.metric("速度", f"{job.speed:.1f}/秒")
        eta = job.eta_sec
        c4.metric("预计剩余", f"{eta / 60:.1f} 分钟" if eta >= 0 else "—")
        if job.total:
            st.progress(job.progress_ratio,
                        text=f"{job.processed} / {job.total}")
        else:
            st.caption(f"已处理 {job.processed} 个（总数未知，边扫边统计）")
        if st.button("⏹ 停止任务", key=f"stop_{key}", use_container_width=True):
            job.request_stop()
    elif job.status in ("done", "stopped", "error", "idle"):
        if job.status == "idle":
            st.caption("未运行")
            return
        icon = {"done": "✅", "stopped": "⏹", "error": "❌"}.get(job.status, "•")
        r = job.result or {}
        if job.status == "error":
            st.error(f"{icon} 任务异常：{job.error}")
        elif job.kind == "scan":
            st.success(
                f"{icon} 扫描完成：新增 {r.get('added', 0)}，跳过 {r.get('skipped', 0)}，"
                f"失败 {r.get('failed', 0)}，用时 {r.get('seconds', 0):.1f}s"
                f"（批次 #{r.get('batch_id')}）")
        else:
            st.success(
                f"{icon} 分析完成：成功 {r.get('analyzed', 0)}，失败 {r.get('errors', 0)}"
                f"（失败项一律保留），用时 {job.elapsed:.1f}s")

    if job.logs:
        with st.expander("任务日志", expanded=(job.status == "error")):
            st.code("\n".join(job.tail_logs(80)), language="text")


# --------------------------------------------------------------------------- 侧边栏
def render_sidebar(db, cfg):
    with st.sidebar:
        st.header("监控视频筛选")
        s = db.stats()
        st.caption(f"数据库：`{cfg['runtime']['sqlite_path']}`")
        c1, c2 = st.columns(2)
        c1.metric("已扫描", s["meta_ok"])
        c2.metric("已分析", s["analyzed"])
        c1.metric("建议清理", s["verdict_trash"])
        c2.metric("待人工", s["verdict_review"])
        c1.metric("已判保留", s["human_keep"])
        c2.metric("已移回收", s["human_trash"])
        if s["errors"]:
            st.warning(f"{s['errors']} 个片段分析失败（一律保留）")
        if s["feedback"]:
            st.info(f"已记录 {s['feedback']} 条误判反馈")

        st.divider()
        sj, aj = jobsmod.get_scan_job(), jobsmod.get_analyze_job()
        if sj.running:
            st.info(f"🔍 后台扫描中：已处理 {sj.processed} 个")
        if aj.running:
            st.info(f"🤖 后台分析中：已处理 {aj.processed} 个（{aj.speed:.1f}/秒）")
        if not sj.running and sj.status == "error":
            st.error("上次扫描任务异常，见「设置与扫描」日志")
        if not aj.running and aj.status == "error":
            st.error("上次分析任务异常，见「设置与扫描」日志")

        if not cfg.get("source_root"):
            st.error("未配置 source_root，请到「设置与扫描」页填写")
        if not cfg.get("trash_root"):
            st.warning("未配置 trash_root，暂不能执行移动")


# --------------------------------------------------------------------------- 页签 1
QUEUE_PAGE = 100  # 审核队列每页加载条数（避免一次性拉几万条 id）


def _queue_load(db, page: int, size: int) -> List[int]:
    return [r["id"] for r in db.review_queue(limit=size, offset=page * size)]


def _queue_ensure(db, size: int) -> None:
    st.session_state.setdefault("q_page", 0)
    if "queue" not in st.session_state or st.session_state.get("_q_reload"):
        st.session_state.queue = _queue_load(db, st.session_state.q_page, size)
        st.session_state.idx = 0
        st.session_state["_q_reload"] = False


def _queue_step(db, delta: int, size: int) -> None:
    """翻页式移动。走到页首/页尾时自动加载相邻页，而不是一次性拉全量队列。"""
    idx = st.session_state.idx + delta
    q = st.session_state.queue
    if idx < 0:
        if st.session_state.q_page > 0:
            st.session_state.q_page -= 1
            st.session_state.queue = _queue_load(db, st.session_state.q_page, size)
            st.session_state.idx = max(0, len(st.session_state.queue) - 1)
        return
    if idx >= len(q):
        nxt = _queue_load(db, st.session_state.q_page + 1, size)
        if not nxt:
            st.session_state.idx = max(0, len(q) - 1)
            st.toast("已经是最后一条了")
        else:
            st.session_state.q_page += 1
            st.session_state.queue = nxt
            st.session_state.idx = 0
        return
    st.session_state.idx = idx


def _queue_drop_current() -> None:
    """当前条已处置，从队列移除，下一条自动顶上。"""
    q = st.session_state.queue
    i = st.session_state.idx
    if 0 <= i < len(q):
        q.pop(i)
    if st.session_state.idx >= len(q):
        st.session_state.idx = max(0, len(q) - 1)


def _render_last_feedback():
    """渲染上一条的处置结果（处置后会跳到下一条，结果只能放在页面固定位置）。"""
    item = st.session_state.get("last_fb")
    if not item:
        return
    r = item["result"]
    with st.container(border=True):
        st.markdown(f"**上一条的处置结果**　`{item['name']}`")
        if r.get("is_misjudgment"):
            st.error(f"**误判原因：{r.get('label')}**")
            st.write(r.get("detail"))
            st.info(f"**改进建议：** {r.get('suggestion')}")
            with st.expander("查看证据明细"):
                st.json(r.get("evidence", {}))
            st.caption("已记入误判分析，可在「误判分析」页查看整体分布与调参建议")
        else:
            st.success("已标记为保留（与 AI 判定一致，不计入误判）")
        if st.button("知道了，清除提示", key="clear_last_fb"):
            st.session_state["last_fb"] = None
            st.rerun()


def tab_review(db, cfg):
    st.subheader("审核工作台")

    hk_on = st.checkbox("启用键盘快捷键", value=True, key="hk_on")
    hotkeys.install(enabled=hk_on)
    st.caption("快捷键：" + hotkeys.hint_markdown(enabled=hk_on))
    _render_last_feedback()

    _queue_ensure(db, QUEUE_PAGE)
    queue: List[int] = st.session_state.queue
    total_pending = db.count_review_queue()

    if not queue:
        st.success("当前没有待审核的片段。可到「设置与扫描」页启动扫描与分析。")
        if st.button("重新加载队列"):
            st.session_state["_q_reload"] = True
            st.rerun()
        return

    col_nav1, col_nav2, col_nav3 = st.columns([1, 2, 1])
    offset0 = st.session_state.q_page * QUEUE_PAGE
    with col_nav1:
        if st.button("← 上一条", use_container_width=True):
            _queue_step(db, -1, QUEUE_PAGE)
            st.rerun()
    with col_nav2:
        st.markdown(
            f"<div style='text-align:center'>第 "
            f"<b>{offset0 + st.session_state.idx + 1}</b> / {max(total_pending, offset0 + len(queue))} 条"
            f"　<span style='color:#888'>(按保留价值分升序，最可疑的排最前)</span></div>",
            unsafe_allow_html=True)
    with col_nav3:
        if st.button("下一条 →", use_container_width=True):
            _queue_step(db, 1, QUEUE_PAGE)
            st.rerun()

    vid = queue[min(st.session_state.idx, len(queue) - 1)]
    row = db.get_video(vid)
    if not row:
        st.error("记录不存在")
        return

    left, right = st.columns([3, 2])
    with left:
        st.markdown(f"**{row['filename']}**")
        st.caption(f"`{row['path']}`")
        tp = thumb_of(row["id"])
        if tp:
            st.image(str(tp), use_container_width=True)
        else:
            st.warning("尚无缩略图（可能未分析或分析失败）")

        b1, b2 = st.columns(2)
        with b1:
            if st.button("▶ 用系统播放器打开", use_container_width=True):
                if Path(row["path"]).exists():
                    open_in_player(row["path"])
                else:
                    st.error("原文件已不存在（可能已被移动）")
        with b2:
            if st.button("🌐 生成网页可播预览", use_container_width=True):
                with st.spinner("转码中（H.265 → H.264，约需数秒）..."):
                    pv = make_browser_preview(row, cfg)
                if pv:
                    st.session_state[f"pv_{row['id']}"] = str(pv)
                else:
                    st.error("转码失败")
        if f"pv_{row['id']}" in st.session_state:
            st.video(st.session_state[f"pv_{row['id']}"])

        with st.expander("文件元数据", expanded=False):
            st.json({
                "年份": row["year"], "相对目录": row["rel_dir"],
                "时长(秒)": row["duration_sec"], "分辨率": f"{row['width']}x{row['height']}",
                "编码": row["video_codec"], "有音轨": bool(row["has_audio"]),
                "大小(MB)": round((row["size_bytes"] or 0) / 1048576, 2),
                "抽帧数": row["sampled_frames"],
            })

    with right:
        st.markdown("### 为什么这么判定")
        render_reasons(row)

    st.divider()
    st.markdown("### 你的处置")
    note = st.text_input("备注（可选，会记入误判分析）", key=f"note_{row['id']}")

    a1, a2, a3 = st.columns(3)
    with a1:
        if st.button("🗑 确认清理（移入分级回收目录）", type="primary", use_container_width=True):
            if not cfg.get("trash_root"):
                st.error("请先在「设置与扫描」配置 trash_root")
            else:
                tier = mover.decide_tier(row, cfg)
                ok, msg = mover.move_one(db, row, cfg, tier)
                if ok:
                    st.session_state["last_fb"] = None
                    _queue_drop_current()
                    st.toast(f"已移入 {mover.TIER_NAMES[tier]}", icon="🗑")
                    st.rerun()
                else:
                    st.error(msg)
    with a2:
        if st.button("✅ 误判，这段要保留", use_container_width=True):
            analyzer = Analyzer(db, cfg)
            r = fbmod.record_feedback(db, row, "keep", cfg,
                                      yolo_on=analyzer.yolo_on, note=note)
            # 存进 session_state，rerun 后仍能看到「为什么判错了」
            st.session_state["last_fb"] = {
                "vid": row["id"], "name": row["filename"], "result": r}
            _queue_drop_current()
            st.rerun()
    with a3:
        if st.button("↩ 撤销移动", use_container_width=True):
            ok, msg = mover.undo_move(db, row)
            if ok:
                st.session_state["_q_reload"] = True
                st.toast(msg, icon="↩")
                st.rerun()
            else:
                st.error(msg)

    # 移动历史
    with st.expander("该文件的操作历史"):
        with db.conn() as c:
            logs = c.execute("SELECT * FROM action_log WHERE video_id=? ORDER BY id DESC",
                             (row["id"],)).fetchall()
        if logs:
            st.dataframe([{"时间": l["created_at"], "动作": l["action"],
                           "从": l["from_path"], "到": l["to_path"]} for l in logs],
                         hide_index=True, use_container_width=True)
        else:
            st.caption("暂无操作记录")


# --------------------------------------------------------------------------- 页签 2
def _grid_sel_all(page_ids):
    """「全选本页」回调：在脚本体执行前置位本页每个格子的复选框状态，
    rerun 后格子才会真正呈现勾选态（带 key 的组件 value= 只在首次实例化生效）。"""
    for i in page_ids:
        st.session_state[f"g_{i}"] = True


def _grid_sel_clear():
    """「清空选择」回调：复位所有已选格子的复选框状态（含跨页残留），
    并清空选中集合。只清 selected 集合的话，残留的组件状态会让下轮 rerun 重新勾上。"""
    for i in list(st.session_state.get("selected", set())):
        st.session_state[f"g_{i}"] = False
    st.session_state["selected"] = set()


def tab_grid(db, cfg):
    st.subheader("批量浏览")

    st.session_state.setdefault("grid_page", 0)
    st.session_state.setdefault("selected", set())
    total = db.count_review_queue()
    pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)
    st.session_state.grid_page = min(st.session_state.grid_page, pages - 1)

    n1, n2, n3 = st.columns([1, 2, 1])
    with n1:
        if st.button("← 上一页", key="grid_prev", use_container_width=True,
                     disabled=st.session_state.grid_page <= 0):
            st.session_state.grid_page -= 1
            st.rerun()
    with n2:
        st.markdown(
            f"<div style='text-align:center'>第 <b>{st.session_state.grid_page + 1}</b> / {pages} 页"
            f"　<span style='color:#888'>共 {total} 条待审核（每页 {PAGE_SIZE} 条，"
            f"按分数升序）</span></div>",
            unsafe_allow_html=True)
    with n3:
        if st.button("下一页 →", key="grid_next", use_container_width=True,
                     disabled=st.session_state.grid_page >= pages - 1):
            st.session_state.grid_page += 1
            st.rerun()

    rows = db.review_queue(limit=PAGE_SIZE, offset=st.session_state.grid_page * PAGE_SIZE)
    if not rows:
        st.info("没有待审核片段")
        return

    selected = st.session_state.selected
    # 清理已离开本页且未被选中的残留复选框键（移动/保留后视频会退出队列，
    # 若不清除其 g_ 键，撤销回到队列时会呈现"幽灵勾选"）。
    visible_ids = {r["id"] for r in rows}
    for k in [k for k in st.session_state.keys() if k.startswith("g_")]:
        try:
            vid = int(k[2:])
        except ValueError:
            continue
        if vid not in visible_ids and vid not in selected:
            del st.session_state[k]
    cols_per_row = 4
    for i in range(0, len(rows), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, row in enumerate(rows[i:i + cols_per_row]):
            with cols[j]:
                tp = thumb_of(row["id"])
                if tp:
                    st.image(str(tp), use_container_width=True)
                score = row["final_score"] if row["final_score"] is not None else row["retention_score"]
                tier = mover.decide_tier(row, cfg)
                st.caption(f"`{score:g}分` {tier}档　{row['filename'][:24]}")
                if st.checkbox("选择", key=f"g_{row['id']}", value=row["id"] in selected):
                    selected.add(row["id"])
                elif row["id"] in selected:
                    selected.discard(row["id"])
    st.session_state.selected = selected

    st.divider()
    c0, c1, c2, c3 = st.columns([2, 2, 2, 2])
    c0.write(f"已选 **{len(selected)}** 个（跨页累积）")
    page_ids = [r["id"] for r in rows]
    with c1:
        st.button("全选本页", use_container_width=True,
                  on_click=_grid_sel_all, args=(page_ids,))
    with c2:
        st.button("清空选择", use_container_width=True, disabled=not selected,
                  on_click=_grid_sel_clear)
    with c3:
        if st.button("🗑 批量移入回收目录", type="primary", use_container_width=True,
                     disabled=not selected):
            if not cfg.get("trash_root"):
                st.error("请先配置 trash_root")
            else:
                targets = [db.get_video(i) for i in selected]
                targets = [t for t in targets if t]
                with st.spinner(f"正在移动 {len(targets)} 个文件（同盘 move，瞬间完成）..."):
                    res = mover.move_batch(db, targets, cfg)
                st.success(f"已移动 {res['moved']} 个，失败 {res['failed']} 个")
                for e in res["errors"]:
                    st.caption(f"· {e}")
                rep = res.get("reports") or {}
                if rep:
                    st.info(f"已自动生成清单：`{rep.get('csv')}`\n\n"
                            f"空间统计：`{rep.get('txt')}`")
                st.session_state.selected = set()
                st.rerun()

    d1, d2 = st.columns([3, 2])
    with d1:
        if st.button("✅ 批量标记保留", use_container_width=True, disabled=not selected):
            for i in selected:
                r = db.get_video(i)
                if r:
                    db.set_human_decision(i, "keep")
            st.success(f"已标记 {len(selected)} 个为保留")
            st.session_state.selected = set()
            st.rerun()
    with d2:
        st.caption("A 档 = 高置信可批量删；B 档 = 建议抽查后再删")


# --------------------------------------------------------------------------- 页签 3
def _fmt_hms(sec: float) -> str:
    sec = max(0, int(sec))
    h, m, s = sec // 3600, (sec % 3600) // 60, sec % 60
    return f"{h}小时{m:02d}分{s:02d}秒" if h else f"{m}分{s:02d}秒"


def tab_progress(db, cfg):
    st.subheader("执行进度")

    st.markdown("##### 后台任务")
    j1, j2 = st.columns(2)
    with j1:
        job_panel(jobsmod.get_scan_job(), "🔍 扫描元数据", "scan")
    with j2:
        job_panel(jobsmod.get_analyze_job(), "🤖 AI 分析", "analyze")

    st.divider()
    st.markdown("##### 批次进度")
    batches = db.list_batches()
    if not batches:
        st.info("暂无批次，请到「设置与扫描」页启动扫描")
    for b in batches:
        p = db.batch_progress(b["id"])
        with st.container(border=True):
            c1, c2, c3, c4, c5 = st.columns([3, 1, 1, 1, 1])
            c1.markdown(f"**#{b['id']} {b['name']}**　`{b['status']}`")
            c1.caption(b["root_dir"])
            c2.metric("总数", p["total"])
            c3.metric("已扫描", p["scanned"])
            c4.metric("已分析", p["analyzed"])
            c5.metric("待分析", p["pending"])
            if p["total"]:
                st.progress(min(1.0, p["analyzed"] / max(p["total"], 1)),
                            text=f"分析进度 {p['analyzed']}/{p['total']}"
                                 f"（{p['analyzed'] * 100 // max(p['total'], 1)}%）")
            if p["errors"]:
                st.caption(f"⚠️ {p['errors']} 个片段处理失败（一律保留，可在审核台查看原因）")

    st.divider()
    st.markdown("##### 已移入回收目录（待你手动删除）")
    rep = db.space_report(None)
    if not rep["total"]["n"]:
        st.caption("还没有移动任何文件。")
    else:
        data = [{
            "年份": y["year"], "文件数": y["n"],
            "A档(可批量删)": y.get("A", 0), "B档(建议抽查)": y.get("B", 0),
            "容量(GB)": round(y["bytes"] / (1024 ** 3), 2),
            "时长(小时)": round(y["secs"] / 3600, 1),
        } for y in rep["by_year"]]
        st.dataframe(data, hide_index=True, use_container_width=True)
        t = rep["total"]
        st.info(f"合计 **{t['n']}** 个文件　**{t['bytes'] / (1024 ** 3):.2f} GB**　"
                f"**{t['secs'] / 3600:.1f} 小时**　"
                f"（A 档 {t.get('A', 0)} 个 / B 档 {t.get('B', 0)} 个）")
        st.caption("注意：文件只是被移动，磁盘空间尚未释放 —— 需要你手动删除后才会释放。")

        if st.button("📄 重新生成清单与空间统计"):
            r = mover.write_reports(db, cfg, None)
            st.success(f"已生成：`{r['csv']}`　`{r['txt']}`")

    st.divider()
    with st.expander("↩ 按批次撤销移动（把文件移回原位）"):
        moved_batches = []
        for b in batches:
            n = len(db.list_moved(b["id"], limit=10000))
            if n:
                moved_batches.append((b["id"], b["name"], n))
        if not moved_batches:
            st.caption("没有可撤销的移动记录")
        else:
            opt = {f"#{i} {name}（{n} 个）": i for i, name, n in moved_batches}
            sel = st.selectbox("选择批次", list(opt.keys()))
            confirm = st.checkbox("我确认要把这些文件移回原位", key="undo_confirm")
            if st.button("执行撤销", disabled=not confirm, type="primary"):
                rows = db.list_moved(opt[sel])
                with st.spinner(f"正在撤销 {len(rows)} 个文件..."):
                    res = mover.undo_batch(db, rows)
                st.success(f"已撤销 {res['undone']} 个，失败 {res['failed']} 个")
                for e in res["errors"][:10]:
                    st.caption(f"· {e}")

    st.divider()
    st.markdown("##### 全局统计")
    st.dataframe([db.stats()], hide_index=True, use_container_width=True)


# --------------------------------------------------------------------------- 页签 4
def tab_feedback(db, cfg):
    st.subheader("误判分析")
    s = fbmod.summarize_feedback(db)
    if not s["total"]:
        st.info("还没有误判记录。在「审核工作台」中把 AI 判定要清理、但你认为该保留的片段"
                "标记为「误判，这段要保留」，系统会自动归因并在这里汇总。")
        return

    c1, c2 = st.columns([1, 2])
    c1.metric("累计误判", s["total"])
    c2.markdown("#### 原因分布")
    st.dataframe(s["distribution"], hide_index=True, use_container_width=True)

    st.markdown("#### 优化建议")
    for sug in s["suggestions"]:
        st.warning(sug)

    st.divider()
    st.markdown("#### 误判明细")
    rows = db.list_feedback(limit=200)
    data = []
    for r in rows:
        v = db.get_video(r["video_id"])
        data.append({
            "时间": r["created_at"],
            "文件": (v["filename"] if v else f"#{r['video_id']}"),
            "AI判定": r["ai_verdict"],
            "人工判定": r["human_verdict"],
            "原因": fbmod.REASON_LABELS.get(r["reason_category"], r["reason_category"]),
            "说明": (r["reason_detail"] or "")[:80],
        })
    st.dataframe(data, hide_index=True, use_container_width=True)

    sel = st.selectbox("查看某条误判的完整证据",
                       options=[r["id"] for r in rows],
                       format_func=lambda i: f"#{i}")
    if sel:
        r = next((x for x in rows if x["id"] == sel), None)
        if r:
            st.markdown(f"**原因：** {fbmod.REASON_LABELS.get(r['reason_category'], r['reason_category'])}")
            st.write(r["reason_detail"])
            st.info(f"**建议：** {r['suggestion']}")
            st.json(json.loads(r["evidence"] or "{}"))


# --------------------------------------------------------------------------- 页签 5
def tab_settings(db, cfg):
    st.subheader("设置与扫描")

    # 网盘同步目录会把正在写入的 SQLite 覆盖回旧版本 —— 几天的分析结果可能凭空消失。
    # 这里主动检测并给出可操作的解决办法。
    risk = cfgmod.detect_sync_risk(cfg.get("work_dir") or ROOT)
    if risk:
        with st.container(border=True):
            st.error(
                f"**⚠ 工作目录位于网盘同步目录内：{risk}**\n\n"
                f"数据库 `videos.db` 会被同步软件来回覆盖，长任务跑出来的分析结果"
                f"可能整批丢失（本次 `config.json` 变 0 字节就是这个原因）。\n\n"
                f"**建议**：在 Synology Drive 的同步任务里排除本项目目录，"
                f"或把「本地工作目录」改到非同步位置（例如 "
                f"`E:\\video-search-data`），再点保存。")
            st.caption("修改工作目录后，旧数据不会自动迁移；如需保留请在保存后手动复制 data 目录。")

    st.markdown("##### 路径配置")
    c1, c2 = st.columns(2)
    src = c1.text_input("视频源根目录（移动硬盘，只读）", cfg.get("source_root", ""))
    trash = c2.text_input("分级回收目录（必须与源同盘）", cfg.get("trash_root", ""))
    work = c1.text_input("本地工作目录（数据库/缩略图，放 SSD）", cfg.get("work_dir", ""))
    model = c2.text_input("YOLO ONNX 模型路径", cfg["analysis"].get("yolo_model_path", ""))

    c3, c4 = st.columns(2)
    thr = c3.slider("目标检测判定阈值", 0.10, 0.90,
                    float(cfg["analysis"].get("yolo_conf_threshold", 0.5)), 0.05)
    nfr = c4.slider("每个片段抽帧数", 3, 16, int(cfg["analysis"].get("sample_frames", 6)), 1)
    yolo_en = c3.checkbox("启用 YOLO 目标检测（无模型时自动降级）",
                          bool(cfg["analysis"].get("yolo_enabled", True)))

    if st.button("💾 保存配置", type="primary"):
        cfg["source_root"] = src
        cfg["trash_root"] = trash
        cfg["work_dir"] = work
        cfg["analysis"]["yolo_model_path"] = model
        cfg["analysis"]["yolo_conf_threshold"] = thr
        cfg["analysis"]["sample_frames"] = nfr
        cfg["analysis"]["yolo_enabled"] = yolo_en
        save_config(cfg)
        st.success(f"已保存到 config.json（修改阈值后需重新分析才生效）")

    st.divider()
    st.markdown("##### 执行（后台运行，不阻塞页面）")
    st.caption("任务跑在独立线程里：启动后可以随意切换页签、继续审核，进度在下方实时显示。"
               "修改阈值后请先点「保存配置」再启动，否则用的还是旧配置。")
    limit = st.number_input("本次处理上限（建议先用小值试点）", 1, 1000000, 200, 100)
    with_audio = st.checkbox(
        "扫描时分析音频能量（明显变慢，仅在摄像头确实录音时才有意义）",
        value=False, key="scan_audio")

    sj, aj = jobsmod.get_scan_job(), jobsmod.get_analyze_job()
    e1, e2 = st.columns(2)
    with e1:
        if st.button("🔍 启动扫描（后台）", use_container_width=True, disabled=sj.running):
            if not src or not Path(src).exists():
                st.error("请先填写有效的源目录")
            else:
                ok, msg = jobsmod.start_scan(db, cfg, src, int(limit), with_audio)
                st.success(msg) if ok else st.warning(msg)
    with e2:
        if st.button("🤖 启动 AI 分析（后台）", use_container_width=True, disabled=aj.running):
            ok, msg = jobsmod.start_analyze(db, cfg, int(limit))
            if ok:
                model = Path(cfg["analysis"].get("yolo_model_path", ""))
                if not model.exists():
                    st.warning("已启动，但未找到 YOLO 模型，将退化为纯动静检测。"
                               "可运行 scripts/download_model.py 下载。")
                else:
                    st.success(msg)
            else:
                st.warning(msg)

    st.divider()
    p1, p2 = st.columns(2)
    with p1:
        job_panel(sj, "🔍 扫描元数据", "scan")
    with p2:
        job_panel(aj, "🤖 AI 分析", "analyze")

    st.divider()
    st.markdown("##### 导出清单")
    st.caption("清单与空间统计会写到回收目录下的 `_reports/`，你手动删除时顺手就能看到。")
    if st.button("📄 生成清理清单 + 空间统计"):
        try:
            r = mover.write_reports(db, cfg, None)
            t = r["report"]["total"]
            if not t["n"]:
                st.warning("还没有任何文件被移入回收目录，清单为空。")
            else:
                st.success(f"清单：`{r['csv']}`")
                st.success(f"统计：`{r['txt']}`")
                st.info(f"合计 {t['n']} 个文件，{t['bytes'] / (1024 ** 3):.2f} GB，"
                        f"{t['secs'] / 3600:.1f} 小时"
                        f"（A 档 {t.get('A', 0)} 个 / B 档 {t.get('B', 0)} 个）")
        except Exception as e:  # noqa: BLE001
            st.error(f"生成失败：{type(e).__name__}: {e}")

    st.divider()
    st.caption("提示：本程序只做分析、打分与分级移动，**不包含任何删除功能**。"
               "移动后的文件位于回收目录，可随时撤销；最终删除请你手动完成。")


# --------------------------------------------------------------------------- main
def main():
    st.set_page_config(page_title="监控视频 AI 筛选工作台",
                       page_icon="🎬", layout="wide")
    cfg = cfg_now()

    # 配置损坏后由 config.load_config 自动恢复过，这里把情况告诉用户，
    # 否则用户只会看到"配置莫名其妙变了"而不知道发生过什么。
    notice = getattr(cfgmod, "LAST_NOTICE", "")
    if notice:
        st.warning(f"**配置文件异常已自动处理**：{notice}")

    db = get_db(cfg["runtime"]["sqlite_path"])
    render_sidebar(db, cfg)

    t1, t2, t3, t4, t5 = st.tabs(
        ["审核工作台", "批量浏览", "执行进度", "误判分析", "设置与扫描"])
    with t1:
        tab_review(db, cfg)
    with t2:
        tab_grid(db, cfg)
    with t3:
        tab_progress(db, cfg)
    with t4:
        tab_feedback(db, cfg)
    with t5:
        tab_settings(db, cfg)


if __name__ == "__main__":
    main()
