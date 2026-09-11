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

import html as htmllib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import feedback as fbmod  # noqa: E402
from src import hotkeys  # noqa: E402
from src import jobs as jobsmod  # noqa: E402
from src import mover  # noqa: E402
from src.analyzer import Analyzer  # noqa: E402
from src import config as cfgmod  # noqa: E402
from src.config import ROOT, THUMB_DIR, load_config, save_config  # noqa: E402
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


def reveal_in_explorer(path: str):
    """在文件管理器中定位该文件（Windows 下选中文件本身，便于周边查看）。"""
    p = Path(path)
    try:
        if platform.system() == "Windows":
            subprocess.Popen(["explorer", "/select,", str(p.resolve())])
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", "-R", str(p)])
        else:
            subprocess.Popen(["xdg-open", str(p.parent)])
    except Exception as e:  # noqa: BLE001
        st.warning(f"无法打开文件管理器：{e}")


def current_video_path(row) -> Path | None:
    """该视频当前实际存在的路径：未移动用源路径，已移入回收目录则用回收路径。

    移动后源路径自然失效，这里逐个 exists() 探测，避免"文件明明在回收目录
    却报已不存在"。都找不到才返回 None（真被手动删除了）。
    """
    keys = row.keys() if hasattr(row, "keys") else ()
    for key in ("path", "trash_path"):
        if key not in keys:
            continue
        v = row[key]
        if v and Path(v).exists():
            return Path(v)
    return None


def make_browser_preview(row, cfg: Dict) -> Path | None:
    """把 H.265 转成浏览器可播的 H.264 预览（H.265 在多数浏览器要靠硬件解码）。"""
    ffmpeg = cfgmod.bin_path(cfg, "ffmpeg_path", "ffmpeg") or "ffmpeg"
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
def render_reasons(row, cfg: Dict | None = None):
    """渲染判定依据链 —— 回答"为什么判定要删/要留"。"""
    trace = {}
    try:
        trace = json.loads(row["decision_trace"] or "{}")
    except Exception:  # noqa: BLE001
        pass

    score = trace.get("score", row["retention_score"])
    verdict = trace.get("verdict", row["verdict"])

    if score is None and verdict is None:
        # 重新分析退回后，缓存队列里的旧 id 会取到已清空结果的记录
        st.warning("该片段的分析结果已被重置（可能刚执行了「重新分析」），"
                   "等待 AI 分析跑完后会在此显示新的判定依据。")
        return trace

    c1, c2 = st.columns([1, 3])
    with c1:
        st.metric("保留价值分", f"{score:g}" if score is not None else "—")
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
    # 阈值取当前配置：调过阈值后"未达阈值"的口径必须跟着变，写死 0.5 会漏报/多报
    conf_thr = float(((cfg or {}).get("analysis") or {}).get("yolo_conf_threshold", 0.5))
    near = [d for d in raw if d["conf"] < conf_thr]
    if near:
        with st.expander(f"⚠️ 存在 {len(near)} 个未达阈值的疑似目标（误判高发线索）", expanded=False):
            st.dataframe(
                [{"类别": d["class"], "置信度": d["conf"], "画面占比": f"{d.get('area_ratio',0)*100:.2f}%"}
                 for d in near[:10]],
                hide_index=True, use_container_width=True,
            )
    return trace


# --------------------------------------------------------------------------- 音频复核卡
def _audio_wav_path(cfg: Dict, video_id: int) -> Path | None:
    wav = Path(cfg.get("work_dir") or "") / "audio" / f"{video_id}.wav"
    return wav if wav.exists() else None


def render_audio_card(row, cfg: Dict):
    """审核工作台右侧的音频复核状态卡。

    回答用户的第二问：这段判删视频的声音里到底有没有内容、说了什么。
    LLM 结论仅是参考，只展示不替代人工判断。
    """
    st.markdown("### 🔊 音频复核")
    if not row["has_audio"]:
        st.caption("该片段**无音轨** —— 没有声音内容可复核，画面判定即为最终依据。")
        return
    ac = row["audio_checked"]
    if ac is None:
        if cfg.get("audio", {}).get("enabled"):
            st.warning("该片段**有音轨但尚未音频复核**。若画外有人声（电话/对讲/警报…），"
                       "这段可能有用。到「设置与扫描」启动音频复核后，"
                       "这里会展示：是否有人声 / 转写全文 / LLM 结论。")
        else:
            st.caption("该片段有音轨。可在「设置与扫描」启用音频复核后批量检查"
                       "「声音里有没有说话内容」，作为清理前的二次确认。")
        return
    if ac == -1:
        st.caption(f"已跳过复核：{row['audio_error'] or '无音轨'}")
        return

    sp = bool(row["speech_present"])
    ratio = float(row["speech_ratio"] or 0.0)
    if sp:
        st.warning(f"**检出人声**（语音占比约 {ratio * 100:.0f}%）"
                   " —— 可能有画外事件，请看下方转写后再决定清理与否")
    else:
        st.success(f"未检出有效人声（语音占比约 {ratio * 100:.0f}%）")

    txt = (row["transcript_text"] or "").strip()
    if txt:
        with st.expander(f"📝 转写全文（{len(txt)} 字，供你确认内容）",
                         expanded=bool(sp)):
            st.write(txt)
    else:
        st.caption("转写为空 —— 无有效说话内容")

    if row["llm_useful"] is not None:
        label = {1: "🟢 可能有用", 0: "⚪ 基本无用", 2: "🟡 不确定"}.get(
            row["llm_useful"], "•")
        reason = row["llm_reason"] or ""
        st.caption(f"LLM 参考：**{label}**　{reason}")
        if sp and row["llm_useful"] == 1:
            st.info("LLM 认为这段声音**可能有用** —— 建议保留或再人工核对一遍转写。")

    wav = _audio_wav_path(cfg, row["id"])
    if wav:
        with st.expander("🔊 播放抽出的音频（复核中间产物）"):
            st.audio(str(wav))
    else:
        st.caption("（抽出的音频已被清理或尚不存在）")


# --------------------------------------------------------------------------- 后台任务面板
def fmt_eta(sec: float) -> str:
    """秒 → 完整时长，不省略任何一级：2小时34分 / 26小时05分 / 1天3小时05分 / 45秒。

    之前用自适应缩写（如 2.6 小时）且 metric 列窄会被截断，用户看不出确切时间。
    """
    if sec is None or sec < 0:
        return "—"
    s = int(round(sec))
    d, s = divmod(s, 86400)
    h, s = divmod(s, 3600)
    m, sec_ = divmod(s, 60)
    if d:
        return f"{d}天{h}小时{m:02d}分"
    if h:
        return f"{h}小时{m:02d}分"
    if m:
        return f"{m}分{sec_:02d}秒"
    return f"{sec_}秒"


@_fragment(run_every="1s")
def job_panel(job, title: str, key: str):
    """后台任务实时面板。

    用 fragment 每秒局部刷新，不阻塞页面、不打断其它页签的操作。
    任务跑在独立线程里，即使关掉浏览器也不会中断（只要 Streamlit 进程还在）。
    """
    st.markdown(f"**{title}**")
    if job.running:
        if job.stage_count:
            # 一键执行：进度条说的是"当前这个阶段"，所以阶段序号必须显式写出来，
            # 否则用户看到 3/6 的进度条会误以为整条链才跑了 3/6。
            st.caption(f"阶段 **{job.stage_index}/{job.stage_count}**："
                       f"{job.stage_name or '准备中…'}")
        c1, c2, c3, c4 = st.columns([2.6, 1, 1.1, 1.7])
        c1.caption(f"正在处理：`{job.current or '初始化…'}`")
        c2.metric("已处理", job.processed)
        spd = job.speed
        # 音频复核速度常 <0.1 段/秒，按"分"显示才看得清（0.0/秒 等于没显示）
        c3.metric("速度", f"{spd:.1f}/秒" if spd >= 0.1 else f"{spd * 60:.1f} 分/分")
        eta_txt = fmt_eta(job.eta_sec)
        c4.metric("预计剩余", eta_txt)
        if job.total:
            # 剩余时间完整写进进度条文本 —— metric 列过窄时可能被截断，此处保底全量显示
            st.progress(job.progress_ratio,
                        text=f"{job.processed} / {job.total} · 预计剩余 {eta_txt}")
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
        elif job.kind == "pipeline":
            st.success(f"{icon} 一键执行结束，用时 {fmt_eta(job.elapsed)}"
                       f"（阶段进度与结果见下）")
        elif job.kind == "scan":
            st.success(
                f"{icon} 扫描完成：新增 {r.get('added', 0)}，跳过 {r.get('skipped', 0)}，"
                f"失败 {r.get('failed', 0)}，用时 {fmt_eta(r.get('seconds', 0))}"
                f"（批次 #{r.get('batch_id')}）")
        elif job.kind == "audio":
            st.success(
                f"{icon} 音频复核完成：{r.get('checked', 0)} 段（检出人声 "
                f"{r.get('speech', 0)} 段），失败 {r.get('errors', 0)} 段，"
                f"用时 {fmt_eta(job.elapsed)}")
        elif job.kind == "highlight":
            g = r.get("grades", {})
            d = r.get("disposed", {}) or {}
            tail = []
            if d.get("keep_full"):
                tail.append(f"自动保留 {d['keep_full']}")
            if d.get("undo_keep"):
                tail.append(f"捞回保留 {d['undo_keep']}")
            if d.get("moved"):
                tail.append(f"移入回收 {d['moved']}")
            if d.get("not_disposed"):
                tail.append(f"留待人工 {d['not_disposed']}")
            disp_txt = "　｜　处置：" + " / ".join(tail) if tail else ""
            st.success(
                f"{icon} 精华剪辑完成：处理 {r.get('done', 0) + r.get('empty', 0)} 段 —— "
                f"切出视频 {r.get('clips', 0)} 段（A {g.get('A', 0)} / B {g.get('B', 0)}），"
                f"疑似静帧 {g.get('C', 0)} 张，无精华 {r.get('empty', 0)} 段，"
                f"失败 {r.get('failed', 0)} 段，用时 {fmt_eta(job.elapsed)}"
                f"{disp_txt}")
        else:
            st.success(
                f"{icon} 分析完成：成功 {r.get('analyzed', 0)}，失败 {r.get('errors', 0)}"
                f"（失败项一律保留），用时 {fmt_eta(job.elapsed)}")

    # 一键执行：逐阶段列结果（停在哪一步、为什么，都写在面板上）
    if job.kind == "pipeline" and not job.running and job.status != "idle":
        r = job.result or {}
        stages = r.get("stages") or []
        for i, s in enumerate(stages, 1):
            secs = f"（{s.get('secs')}s）" if s.get("secs") else ""
            if s.get("status") == "error":
                st.error(f"❌ {i}. **{s['name']}**{secs} — {s['text']}")
            else:
                st.markdown(f"✅ {i}. **{s['name']}**{secs} — {s['text']}")
        done_keys = {s.get("key") for s in stages}
        skipped = [n for k, n in jobsmod.PIPELINE_STAGES if k not in done_keys]
        if r.get("aborted"):
            st.warning(
                f"已在阶段「{r.get('stopped_at')}」停下，后续阶段没有跑"
                f"（{'、'.join(skipped)}）。修好原因后**再点一次即可** —— "
                "各阶段都有幂等标记，已完成的会跳过，不会重复劳动。")
        elif job.status == "stopped":
            st.info("已手动停止 —— 再点一次从断点续跑（已完成的阶段自动跳过）。")
        elif stages and not skipped:
            st.success("全链路跑完 —— **回收目录已就绪**。接下来：翻「精华复核」页看 A/B 产物，"
                       "确认无误后手动清空回收目录（在此之前原片都还在，可撤销、可精扫捞回）。")

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


def _queue_filters() -> Dict[str, Any]:
    """当前审核队列的筛选条件。

    键名与 db.review_queue / db.count_review_queue 的形参一致，可直接 `**filt`
    展开 —— 队列、计数、跳转定位三处必须打同一组参数，否则会出现
    「页面说 300 条、翻到第 2 页却是空的」这类口径漂移。
    """
    f: Dict[str, Any] = {}
    if st.session_state.get("rv_speech_only"):
        f["speech_only"] = True
    if st.session_state.get("rv_person_only"):
        f["person_only"] = True
    ms = float(st.session_state.get("rv_min_score") or 0)
    if ms > 0:
        f["min_score"] = ms
    # Track C · C3：产物**只有 C 级**的视频是「档案」不是「待办」——
    # 剪辑已留底（Track C 生效后 C 级也出可播放 mp4），再逐条过一遍等于把
    # "多留"变成"多看"。默认滤掉，勾「含仅 C 级档案」可放回来。
    # 只影响列表展示，不参与放行/审核任何判定。
    if not st.session_state.get("rv_show_c_archive", False):
        f["hide_c_archive"] = True
    return f


def _queue_sig(filt: Dict[str, Any] | None = None) -> str:
    """筛选条件的指纹。变了就说明用户换了筛子，队列得从头重载。"""
    return json.dumps(filt if filt is not None else _queue_filters(), sort_keys=True)


def _queue_load(db, page: int, size: int, filt: Dict[str, Any]) -> List[int]:
    return [r["id"] for r in db.review_queue(limit=size, offset=page * size, **filt)]


def _queue_ensure(db, size: int, filt: Dict[str, Any]) -> None:
    st.session_state.setdefault("q_page", 0)
    sig = _queue_sig(filt)
    filt_changed = st.session_state.get("_q_sig") not in (None, sig)
    if ("queue" not in st.session_state or st.session_state.get("_q_reload")
            or filt_changed):
        if filt_changed:
            st.session_state.q_page = 0      # 换了筛子，从第一条看起
        st.session_state.queue = _queue_load(db, st.session_state.q_page, size, filt)
        st.session_state.idx = 0
        st.session_state["_q_reload"] = False
        st.session_state["_q_sig"] = sig


def _queue_step(db, delta: int, size: int, filt: Dict[str, Any]) -> None:
    """翻页式移动。走到页首/页尾时自动加载相邻页，而不是一次性拉全量队列。"""
    idx = st.session_state.idx + delta
    q = st.session_state.queue
    if idx < 0:
        if st.session_state.q_page > 0:
            st.session_state.q_page -= 1
            st.session_state.queue = _queue_load(db, st.session_state.q_page, size, filt)
            st.session_state.idx = max(0, len(st.session_state.queue) - 1)
        else:
            st.toast("已经是第一条了")
        return
    if idx >= len(q):
        nxt = _queue_load(db, st.session_state.q_page + 1, size, filt)
        if not nxt:
            st.session_state.idx = max(0, len(q) - 1)
            st.toast("已经是最后一条了")
        else:
            st.session_state.q_page += 1
            st.session_state.queue = nxt
            st.session_state.idx = 0
        return
    st.session_state.idx = idx


def _queue_jump(db, target: int, size: int, filt: Dict[str, Any]) -> None:
    """跳到队列第 target 条（0-based）。按需只加载它所在的那一页。

    队列是「分数升序」的活视图，跳过去看到的就是那一刻库里的真实顺序；
    这里不做位置补偿 —— 处置/放行会让条目离开队列，位置本就会漂。
    """
    total = db.count_review_queue(**filt)
    if total <= 0:
        return
    target = max(0, min(int(target), total - 1))
    page, idx = divmod(target, size)
    if page != st.session_state.get("q_page") or not st.session_state.get("queue"):
        st.session_state.q_page = page
        st.session_state.queue = _queue_load(db, page, size, filt)
        st.session_state["_q_sig"] = _queue_sig(filt)
    st.session_state.idx = idx
    st.session_state["_q_reload"] = False


def _queue_clear_filters() -> None:
    """清空筛选（回调里改 widget 状态 —— 渲染后再改 session_state 会抛异常）。"""
    for k in ("rv_person_only", "rv_speech_only"):
        st.session_state[k] = False
    st.session_state["rv_show_c_archive"] = False   # 回到默认视图（C 级档案不占队列）
    st.session_state["rv_min_score"] = 0.0


def _render_queue_nav(db, filt: Dict[str, Any], total: int, pos: int) -> None:
    """审核台的「定位 / 筛选」条（ADR 0002 的配套改动）。

    队列降到几百条后，问题从"条目太多"变成"在这几百条里怎么定位" ——
    跳转与筛选共用同一组谓词，跳过去的分数段和页面上显示的总数才对得上。
    """
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns([1.6, 0.7, 1.6, 0.7])
        with c1:
            st.number_input(f"跳到第几条（共 {total} 条）", min_value=1, step=1,
                            value=min(max(pos, 1), max(total, 1)),
                            key="rv_jump_idx",
                            help="队列按分数升序，条号即当前位置")
        with c2:
            st.write("")   # 与左侧输入框的 label 对齐
            if st.button("跳转", key="rv_jump_idx_go", use_container_width=True,
                         disabled=total <= 0):
                _queue_jump(db, int(st.session_state.rv_jump_idx) - 1, QUEUE_PAGE, filt)
                st.rerun()
        with c3:
            st.number_input("跳到分数 ≥", min_value=0.0, step=10.0, value=0.0,
                            key="rv_jump_score",
                            help="跳到第一条「分数不低于该值」的片段；0 = 不用")
        with c4:
            st.write("")
            if st.button("跳转", key="rv_jump_score_go", use_container_width=True,
                         disabled=total <= 0):
                sc = float(st.session_state.rv_jump_score or 0)
                # 分数升序 ⇒ 低于 sc 的条数就是第一条 ≥sc 的位置；别在 UI 里自己算，
                # 否则一加筛选条件跳转位置就与队列错位。
                tgt = db.count_review_below(sc, **filt)
                if tgt >= total:
                    # 没有"分数够高"的片段：说清楚，别默默把人丢到末尾
                    st.warning(f"没有分数 ≥ {sc:g} 的片段（队列里最高分还不到）")
                else:
                    _queue_jump(db, tgt, QUEUE_PAGE, filt)
                    st.rerun()

        f1, f2, f3, f4, f5 = st.columns([1.2, 1.2, 1.2, 1.5, 0.9])
        with f1:
            st.checkbox("仅看检出人物", key="rv_person_only",
                        help="只看 L1 抽帧里检出人物的片段（无人片段交给批量放行 + 画面精华兜底）")
        with f2:
            st.checkbox("仅看检出人声", key="rv_speech_only",
                        help="只看音频复核确认有说话内容的片段")
        with f3:
            # 不传 value=（默认取 min_value）——「清除筛选」回调要写这个 key，
            # 同时传 value 会让 Streamlit 打 "default value vs Session State" 警告
            st.number_input("最低分 ≥", min_value=0.0, step=10.0,
                            key="rv_min_score", help="0 = 不限；与上面的跳转共用同一口径")
        with f4:
            st.checkbox("含仅 C 级档案", key="rv_show_c_archive",
                        help="C 级产物是**档案**不是**待办**：剪辑已留底（含可播放 mp4），"
                             "默认不进这条队列。想连它们一起过一遍时勾上。")
        with f5:
            st.write("")
            if st.button("清除筛选", key="rv_filter_clear",
                         use_container_width=True,
                         disabled=not (filt.get("speech_only") or filt.get("person_only")
                                       or filt.get("min_score")),
                         on_click=_queue_clear_filters):
                pass


def _queue_drop_current() -> None:
    """当前条已处置，从队列移除，下一条自动顶上。"""
    q = st.session_state.queue
    i = st.session_state.idx
    if 0 <= i < len(q):
        q.pop(i)
    if st.session_state.idx >= len(q):
        st.session_state.idx = max(0, len(q) - 1)


def _render_release_candidates(db, cfg):
    """审核减负 · 无人候选一键放行（ADR 0003）。

    放行候选 = 判删候选里「机器没看到任何人物信号」的片段 —— 未检出人物，
    且 decision_trace 里也没有"疑似人物未达阈值"。人眼盯 L1 那 16 张抽帧
    不会比机器看到更多，逐条审它们不产生任何信息。
    一键移入回收目录，之后由画面精华剪辑（8 倍采样密度 + 更低检出阈值）兜底重扫。
    """
    thr, need = mover.release_params(cfg)
    n_rel = db.count_release(threshold=thr, require_no_person=need)
    n_pending = db.count_review_queue()
    if n_rel == 0:
        return
    # 提示只在仍有大批候选时出现；数量接近 0 时自动消失，不占版面
    pct = n_rel * 100 // max(1, n_pending) if n_pending else 100
    what = ("这些片段机器**没有检出任何人物信号**——没检出人，"
            "连「疑似未达阈值」都没有。" if need
            else "这些片段分数不高于放行门槛。")
    with st.container(border=True):
        c1, c2 = st.columns([3, 2])
        with c1:
            st.markdown(
                f"**⚡ 审核减负：{n_rel} 个「无人」片段可一键放行**"
                f"（占当前待审核的 {pct}%）\n\n"
                + what
                + "人眼盯的缩略图正是机器抽的那 16 帧，逐条看不会比机器看到更多 —— "
                  "属于无效劳动。放行后由「画面精华剪辑」用 **8 倍采样密度** 兜底重扫，"
                  "万一有内容仍会剪出来。")
            st.caption("检出的猫狗不拦（动物权重只有人物的三分之一）；"
                       "「疑似人物」是唯一会拦住放行的信号。")
        with c2:
            trash_root = (cfg.get("trash_root") or "").strip()
            if st.button(f"🗑 一键放行 {n_rel} 个无人片段", type="primary",
                         use_container_width=True,
                         disabled=not trash_root or jobsmod.any_running()):
                rows = db.release_queue(limit=100000, threshold=thr,
                                        require_no_person=need)
                if not rows:
                    st.rerun()
                with st.spinner(f"正在移入分级回收目录（{len(rows)} 个，同盘 move 瞬间完成）..."):
                    res = mover.move_all(db, rows, cfg)
                if res["moved"]:
                    st.toast(f"已放行 {res['moved']} 个，失败 {res['failed']} 个", icon="🗑")
                else:
                    st.warning(f"放行失败 {res['failed']} 个，详见报告")
                for e in res["errors"]:
                    st.caption(f"· {e}")
                rep = res.get("reports") or {}
                if rep:
                    st.info(f"移动清单：`{rep.get('csv')}`\n\n空间统计：`{rep.get('txt')}`")
                st.session_state["_q_reload"] = True   # 放行后待审队列已变化
                st.rerun()
            if not trash_root:
                st.warning("未配置 trash_root，无法放行")
            st.caption("程序只移动不删除；放行后请勿手动清空回收目录，"
                       "先跑完「画面精华剪辑」留底。")


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

    # —— 审核减负 · 无人候选一键放行（ADR 0003：没看到人物信号的不值得人看）——
    _render_release_candidates(db, cfg)

    hk_on = st.checkbox("启用键盘快捷键", value=True, key="hk_on")
    hotkeys.install(enabled=hk_on)
    st.caption("快捷键：" + hotkeys.hint_markdown(enabled=hk_on))
    _render_last_feedback()

    fil = _queue_filters()
    _queue_ensure(db, QUEUE_PAGE, fil)
    queue: List[int] = st.session_state.queue
    total_pending = db.count_review_queue(**fil)
    offset0 = st.session_state.q_page * QUEUE_PAGE
    cur_pos = offset0 + st.session_state.idx + 1

    # 定位/筛选条先于空态渲染 —— 否则筛选条件把队列清空后，用户没有入口改回来
    _render_queue_nav(db, fil, total_pending, cur_pos)

    if not queue:
        if fil:
            st.warning("当前筛选条件下没有待审核片段 —— 放宽筛选，或点上面的「清除筛选」"
                       "看全部待审核。")
        else:
            st.success("当前没有待审核的片段。可到「设置与扫描」页启动扫描与分析。")
        if st.button("重新加载队列"):
            st.session_state["_q_reload"] = True
            st.rerun()
        return

    col_nav1, col_nav2, col_nav3 = st.columns([1, 2, 1])
    with col_nav1:
        if st.button("← 上一条", use_container_width=True):
            _queue_step(db, -1, QUEUE_PAGE, fil)
            st.rerun()
    with col_nav2:
        st.markdown(
            f"<div style='text-align:center'>第 "
            f"<b>{cur_pos}</b> / {max(total_pending, offset0 + len(queue))} 条"
            f"　<span style='color:#888'>(按保留价值分升序，最可疑的排最前"
            f"{'，已筛选' if fil else ''})</span></div>",
            unsafe_allow_html=True)
    with col_nav3:
        if st.button("下一条 →", use_container_width=True):
            _queue_step(db, 1, QUEUE_PAGE, fil)
            st.rerun()

    vid = queue[min(st.session_state.idx, len(queue) - 1)]
    row = db.get_video(vid)
    if not row:
        st.error("记录不存在")
        return
    if row["verdict"] not in ("trash", "review"):
        # 队列 id 缓存已过期：该条被重新分析退回 / 已被处置，不再属于审核范围。
        # 重新加载当前页 —— _queue_load 按 review_queue 谓词查库，新页不会再失效，
        # 因此这里 rerun 一次即可收敛，不会循环。
        st.session_state["_q_reload"] = True
        st.toast("队列已刷新（有片段被重新分析或已处置）", icon="🔄")
        st.rerun()

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
                vp = current_video_path(row)
                if vp:
                    open_in_player(str(vp))
                else:
                    st.error("原文件已不存在（源文件与回收目录都找不到）")
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
        render_reasons(row, cfg)
        render_audio_card(row, cfg)

    st.divider()
    st.markdown("### 你的处置")

    # —— 音频复核关口：有音轨且未复核的判删候选，要求人工明示后才允许一键清理 ——
    audio_gate = (bool(row["has_audio"]) and row["audio_checked"] is None
                  and cfg.get("audio", {}).get("enabled"))
    confirm_skip_audio = True
    if audio_gate:
        st.warning("⚠️ 此片段**有音轨但尚未音频复核**。若画外有人声，它可能是你要留的"
                   "事件记录。建议先启动音频复核查看转写，再决定。")
        confirm_skip_audio = st.checkbox(
            "我已人工看过原始视频，确认不需要音频复核，允许直接清理",
            key=f"askip_{row['id']}")
    elif row["audio_checked"] == 1 and row["speech_present"] and row["llm_useful"] == 1:
        st.info("该片段音频复核**检出人声**且 LLM 认为可能有用 —— 若你确认这段没有保留"
                "价值，仍可清理；此提示仅为避免误删。")

    note = st.text_input("备注（可选，会记入误判分析）", key=f"note_{row['id']}")

    a1, a2, a3 = st.columns(3)
    with a1:
        if st.button("🗑 确认清理（移入分级回收目录）", type="primary",
                     use_container_width=True, disabled=not confirm_skip_audio):
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
# 批量浏览格子里的音频摘要样式。九宫格每格很窄，转写必须限高截断，
# 全文挂 title 由浏览器 hover 弹出（比塞进格子可读得多）。
_GRID_AUDIO_CSS = """
<style>
.g-aud{font-size:.76rem;line-height:1.35;margin:2px 0 5px;word-break:break-all;
  display:-webkit-box;-webkit-line-clamp:3;-webkit-box-orient:vertical;overflow:hidden}
.g-aud-sp{color:#ff9f1c;font-weight:600}
.g-aud-none{color:#8892a0}
.g-aud-wait{color:#8892a0;font-style:italic}
</style>
"""

# LLM 有用性在格子里的紧凑标记（与审核台三态一致）
_LLM_TAG = {1: "🟢有用", 0: "⚪无用", 2: "🟡不定"}


def _grid_audio_html(row) -> str:
    """批量浏览格子下方的音频信息行：有人声则高亮「占比 + 转写」，否则说明状态。

    返回 HTML 片段；无音轨时返回空串（整行不渲染，避免给静音片段加噪音）。
    """
    if not row["has_audio"]:
        return ""
    ac = row["audio_checked"]
    if ac is None or ac == 0:
        return "<div class='g-aud g-aud-wait'>🎧 未音频复核</div>"
    if ac == -1:
        err = htmllib.escape(str(row["audio_error"] or "跳过"))
        return f"<div class='g-aud g-aud-wait' title='{err}'>🎧 跳过复核</div>"

    sp = bool(row["speech_present"])
    ratio = float(row["speech_ratio"] or 0.0)
    txt = (row["transcript_text"] or "").strip()
    if not sp and not txt:
        return f"<div class='g-aud g-aud-none'>🔇 无人声（占比 {ratio * 100:.0f}%）</div>"

    tag = _LLM_TAG.get(row["llm_useful"], "")
    head = f"🔊 {ratio * 100:.0f}%" if sp else "🔊 转写"
    if tag:
        head += f" {tag}"
    shown = htmllib.escape(txt) if txt else "<i>（转写为空）</i>"
    full = htmllib.escape(txt or "转写为空")
    if len(full) > 400:
        full = full[:400] + "…"
    cls = "g-aud-sp" if sp else "g-aud-none"
    return f"<div class='g-aud {cls}' title='{full}'>{head}　{shown}</div>"


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


def _grid_filters() -> Dict[str, Any]:
    """批量浏览的筛选条件 —— 与审核台同一套 db 谓词，两页看到的条目口径一致。"""
    f: Dict[str, Any] = {}
    if st.session_state.get("grid_speech_only"):
        f["speech_only"] = True
    if st.session_state.get("grid_person_only"):
        f["person_only"] = True
    ms = float(st.session_state.get("grid_min_score") or 0)
    if ms > 0:
        f["min_score"] = ms
    return f


def tab_grid(db, cfg):
    st.subheader("批量浏览")

    st.session_state.setdefault("grid_page", 0)
    st.session_state.setdefault("selected", set())
    st.session_state.setdefault("grid_show_audio", True)
    st.session_state.setdefault("grid_speech_only", False)
    st.session_state.setdefault("grid_person_only", False)

    def _to_first_page():
        st.session_state.grid_page = 0

    f1, f2, f3, f4, f5 = st.columns([1.1, 1.1, 1.1, 1, 2.2])
    with f1:
        st.checkbox("🎧 显示音频转写", key="grid_show_audio")
    with f2:
        st.checkbox("仅看检出人声", key="grid_speech_only", on_change=_to_first_page,
                    help="只列出音频复核确认有说话内容的片段 —— 画外可能有人声事件，"
                         "清理前值得单独过一遍")
    with f3:
        st.checkbox("仅看检出人物", key="grid_person_only", on_change=_to_first_page,
                    help="只看 L1 抽帧里检出人物的片段；无人片段已由批量放行接管")
    with f4:
        st.number_input("最低分 ≥", min_value=0.0, step=10.0, value=0.0,
                        key="grid_min_score", on_change=_to_first_page,
                        help="0 = 不限")
    with f5:
        n_sp = db.count_review_queue(speech_only=True)
        st.caption(f"待审核中 **{n_sp}** 段检出人声（画外可能有事件，"
                   "清理前建议先看转写）")

    fil = _grid_filters()
    total = db.count_review_queue(**fil)
    pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)
    st.session_state.grid_page = min(st.session_state.grid_page, pages - 1)

    n1, n2, n3, n4, n5 = st.columns([0.8, 1, 2, 1, 0.8])
    with n1:
        if st.button("⏮ 首页", key="grid_first", use_container_width=True,
                     disabled=st.session_state.grid_page <= 0):
            st.session_state.grid_page = 0
            st.rerun()
    with n2:
        if st.button("← 上一页", key="grid_prev", use_container_width=True,
                     disabled=st.session_state.grid_page <= 0):
            st.session_state.grid_page -= 1
            st.rerun()
    with n3:
        st.markdown(
            f"<div style='text-align:center'>第 <b>{st.session_state.grid_page + 1}</b> / {pages} 页"
            f"　<span style='color:#888'>共 {total} 条待审核（每页 {PAGE_SIZE} 条，"
            f"按分数升序{'，已筛选' if fil else ''}）</span></div>",
            unsafe_allow_html=True)
    with n4:
        if st.button("下一页 →", key="grid_next", use_container_width=True,
                     disabled=st.session_state.grid_page >= pages - 1):
            st.session_state.grid_page += 1
            st.rerun()
    with n5:
        if st.button("末页 ⏭", key="grid_last", use_container_width=True,
                     disabled=st.session_state.grid_page >= pages - 1):
            st.session_state.grid_page = pages - 1
            st.rerun()

    # 页码跳转：翻页按钮在几百页面前等于没有导航，"直接去第 N 页"才是主通道
    g1, g2, g3 = st.columns([1.6, 0.7, 3.7])
    with g1:
        st.number_input(f"跳到第几页（共 {pages} 页）", min_value=1, step=1,
                        value=st.session_state.grid_page + 1, key="grid_jump_page",
                        help="输入页码后点「跳转」，越界会自动收到最后一页")
    with g2:
        st.write("")
        if st.button("跳转", key="grid_jump_go", use_container_width=True):
            st.session_state.grid_page = int(st.session_state.grid_jump_page) - 1
            st.rerun()
    with g3:
        st.write("")
        st.caption("队列是按分数升序排的：分数低的在前 —— 越靠前越可能是删掉的，"
                   "越靠后越值得看一眼画面。")

    rows = db.review_queue(limit=PAGE_SIZE,
                           offset=st.session_state.grid_page * PAGE_SIZE, **fil)
    if not rows:
        st.info("当前筛选条件下没有待审核片段" if fil else "没有待审核片段")
        return

    st.markdown(_GRID_AUDIO_CSS, unsafe_allow_html=True)

    selected = st.session_state.selected
    # 清理已离开本页且未被选中的残留复选框键（移动/保留后视频会退出队列，
    # 若不清除其 g_ 键，撤销回到队列时会呈现"幽灵勾选"）。p_/f_ 是播放/定位
    # 按钮键，一并回收避免随翻页无限累积。
    visible_ids = {r["id"] for r in rows}
    for k in [k for k in st.session_state.keys()
              if k.startswith(("g_", "p_", "f_"))]:
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
                st.caption(f"`{score:g}分` {tier}档　{row['filename'][:24]}"
                           if score is not None else
                           f"`—分` {tier}档　{row['filename'][:24]}")
                if st.session_state.grid_show_audio:
                    ah = _grid_audio_html(row)
                    if ah:
                        st.markdown(ah, unsafe_allow_html=True)

                # 静态缩略图不足以判断画面内容 —— 直接调系统播放器 / 资源管理器
                pb, fb, cb = st.columns([1, 1, 1.3])
                with pb:
                    if st.button("▶ 播放", key=f"p_{row['id']}",
                                 help="用系统默认播放器打开原视频",
                                 use_container_width=True):
                        vp = current_video_path(row)
                        if vp:
                            open_in_player(str(vp))
                            st.toast(f"已调用播放器：{vp.name}", icon="▶")
                        else:
                            st.error("原文件已不存在（可能已被手动删除）")
                with fb:
                    if st.button("📂 定位", key=f"f_{row['id']}",
                                 help="在文件管理器中定位该文件",
                                 use_container_width=True):
                        vp = current_video_path(row)
                        if vp:
                            reveal_in_explorer(str(vp))
                        else:
                            st.error("原文件已不存在（可能已被手动删除）")
                with cb:
                    if st.checkbox("选择", key=f"g_{row['id']}",
                                   value=row["id"] in selected):
                        selected.add(row["id"])
                    elif row["id"] in selected:
                        selected.discard(row["id"])
    st.session_state.selected = selected

    st.divider()

    # —— 批量移入关口：所选里混有"有音轨且未复核"的判删候选时要求明示 ——
    b_gate_ok = True
    if selected and cfg.get("audio", {}).get("enabled"):
        sel_rows = [db.get_video(i) for i in selected]
        gate_rows = [r for r in sel_rows
                     if r and r["has_audio"] and r["audio_checked"] is None]
        if gate_rows:
            st.warning(f"所选 **{len(selected)}** 个中有 **{len(gate_rows)} 个有音轨但尚未"
                       "音频复核** —— 画面无人不代表没有内容，画外可能有人声事件。"
                       "建议先跑音频复核（转写文本会在审核台展示），再回来批量移入。")
            b_gate_ok = st.checkbox("我已人工确认这批无需音频复核，允许直接批量移入",
                                    key="b_gate_ok")

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
                     disabled=(not selected or not b_gate_ok)):
            if not cfg.get("trash_root"):
                st.error("请先配置 trash_root")
            else:
                targets = [db.get_video(i) for i in selected]
                targets = [t for t in targets if t]
                with st.spinner(f"正在移动 {len(targets)} 个文件（同盘 move，瞬间完成）..."):
                    res = mover.move_rows(db, targets, cfg)
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
    j1, j2, j3 = st.columns(3)
    with j1:
        job_panel(jobsmod.get_scan_job(), "🔍 扫描元数据", "scan")
    with j2:
        job_panel(jobsmod.get_analyze_job(), "🤖 AI 分析", "analyze")
    with j3:
        job_panel(jobsmod.get_audio_job(), "🔊 音频复核", "audio")
    _aj = jobsmod.get_archive_job()
    if _aj.status != "idle":      # 只在归档跑过/在跑时才占版面
        job_panel(_aj, "📦 语音档案归档", "archive")
    _hj = jobsmod.get_highlight_job()
    if _hj.status != "idle":
        job_panel(_hj, "🎞 画面精华剪辑", "highlight_progress")

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


# --------------------------------------------------------------------------- 页签 4.5
GRADE_BADGE = {"A": "🅰 A 级 · 确认", "B": "🅱 B 级 · 短暂", "C": "❓ C 级 · 疑似静帧"}
HIGHLIGHT_PAGE = 16  # 每页 4×4


def tab_highlight(db, cfg):
    """精华复核：浏览已切出的画面精华，按 A/B/C 分级。

    全自动模式下没有"勾选"环节 —— 机器切完分级存好，想看哪级看哪级。
    定位是**原片被删之前的最后一道保全**（ADR 0006）：
    A/B 是"确认有内容"，C 是"看到过但不敢确认"——C 现在也切可播放 mp4，
    但它**是档案不是待办**，默认不占用你的注意力。
    """
    st.subheader("精华复核")
    st.caption(
        "删除前的画面留底。**A 级**确认有人且 ≥8 秒或多人同框，**B 级**确认有人但较短，"
        "**C 级**是粗扫看到过、精扫没敢确认的疑似段 —— 它也切了可播放的视频（+3 张静帧），"
        "作为**档案**存在：默认不用看，但原片删了以后它还找得回来。")

    rows = db.list_highlight_rows()
    clips = []
    n_empty = n_err = 0
    for r in rows:
        if r["highlight_error"]:
            n_err += 1
            continue
        try:
            row_clips = json.loads(r["highlight_clips"] or "[]")
        except (TypeError, ValueError):
            row_clips = []
        if not row_clips:
            n_empty += 1
        for c in row_clips:
            c = dict(c, video_id=r["id"], filename=r["filename"], year=r["year"])
            clips.append(c)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("已处理", len(rows))
    c2.metric("A 级", sum(1 for c in clips if c["grade"] == "A"))
    c3.metric("B 级", sum(1 for c in clips if c["grade"] == "B"))
    c4.metric("C 级档案", sum(1 for c in clips if c["grade"] == "C"))
    c5.metric("无精华", n_empty)

    # ---- Track 0 尺子：「无精华」拆三义 ----
    # 没有这块，"无精华 3780" 就永远是个糊涂数 —— 分不清"真复核过没有"、
    # "压根没跑检测"、还是"看到过却被丢弃了"。第三个数就是 Track C 的成效指标。
    nb = db.no_clip_breakdown()
    if nb["total"]:
        with st.expander(f"「无精华」{nb['total']} 条拆开看（Track 0 尺子）"):
            st.markdown(
                f"| 分义 | 条数 | 含义 |\n|---|---|---|\n"
                f"| 已检测·确实没有 | {nb['detected_empty']} | 粗扫跑过检测，一帧命中都没有 |\n"
                f"| 预筛跳过 | {nb['skipped']} | 帧差预筛判定整片画面没变，YOLO 一帧没跑"
                f"（已接受的口径） |\n"
                f"| **看到过但被否决** | **{nb['dropped']}** | 粗扫有命中，最终却零产物"
                f" —— Track C 要救的就是这批 |\n")
            if nb["unknown"]:
                st.caption(f"另有 {nb['unknown']} 条是回填前剪的，没有送检帧数记录。")
            st.caption("第三个数随着 Track C 重剪会趋近于 0 —— 它就是「删前保全」的成效指标。")

    job = jobsmod.get_highlight_job()
    if job.running:
        _hj = job
        job_panel(_hj, "🎞 画面精华剪辑", "highlight_page")

    if not clips:
        if n_err:
            st.warning(f"{n_err} 段处理失败（源文件缺失/损坏），已跳过不重试；"
                       "可在「设置与扫描」里重置后重跑。")
        st.info("还没有精华产物。到「设置与扫描」页的「画面精华剪辑」启动任务。")
        return

    st.divider()
    view = st.radio("看哪一档", ["A/B（需要看）", "C 档案（可以不看）", "全部"],
                    horizontal=True, key="hl_grade")
    if view.startswith("A/B"):
        shown = [c for c in clips if c["grade"] in ("A", "B")]
    elif view.startswith("C"):
        shown = [c for c in clips if c["grade"] == "C"]
    else:
        shown = list(clips)
    shown.sort(key=lambda c: (-(c.get("score") or 0), c.get("shot_at") or ""))
    st.caption(f"共 {len(shown)} 条，按价值排序")

    page = st.number_input("页码", 1, max(1, (len(shown) - 1) // HIGHLIGHT_PAGE + 1),
                           1, key="hl_page")
    chunk = shown[(page - 1) * HIGHLIGHT_PAGE: page * HIGHLIGHT_PAGE]
    for row_start in range(0, len(chunk), 4):
        cols = st.columns(4)
        for col, c in zip(cols, chunk[row_start: row_start + 4]):
            with col:
                thumbs = [t for t in (c.get("thumbs") or ([c.get("thumb")] if c.get("thumb") else []))
                          if t and Path(t).exists()]
                if thumbs:
                    col.image(thumbs[0], use_container_width=True)
                    if len(thumbs) > 1:      # C 级多张：一张证明不了这 30 秒里发生过什么
                        tc = col.columns(min(3, len(thumbs) - 1))
                        for tcol, t2 in zip(tc, thumbs[1:4]):
                            tcol.image(t2, use_container_width=True)
                else:
                    col.caption("（缩略图缺失）")
                shot = (c.get("shot_at") or "")[:19].replace("T", " ")
                col.markdown(
                    f"**{GRADE_BADGE.get(c['grade'], c['grade'])}**  \n"
                    f"`{shot}`  \n"
                    f"{c.get('dur', 0):.0f} 秒 · 置信 {c.get('conf', 0):.2f}"
                    + (f" · {c.get('n_person', 0)} 人" if c.get("n_person") else ""))
                tgt = c.get("file") or c.get("thumb")
                b1, b2 = col.columns(2)
                if tgt and Path(tgt).exists():
                    if b1.button("▶ 播放", key=f"hl_play_{c['video_id']}_{c['index']}",
                                 use_container_width=True):
                        open_in_player(tgt)
                    if b2.button("📂 定位", key=f"hl_loc_{c['video_id']}_{c['index']}",
                                 use_container_width=True):
                        reveal_in_explorer(tgt)
                else:
                    col.caption("产物文件已不存在")


# --------------------------------------------------------------------------- 页签 5
def _apply_pipeline_cfg(cfg, boxes: Dict) -> None:
    """把「一键执行」区的控件值写回 cfg（不落盘 —— 落盘统一走 _save_all_settings）。"""
    cfg["source_root"] = boxes["src"]
    cfg["trash_root"] = boxes["trash"]
    cfg["work_dir"] = boxes["work"]
    cfg["analysis"]["yolo_model_path"] = boxes["model"]
    cfg["analysis"]["yolo_conf_threshold"] = boxes["thr"]
    cfg["analysis"]["sample_frames"] = boxes["nfr"]
    cfg["analysis"]["yolo_enabled"] = boxes["yolo_en"]
    cfg.setdefault("archive", {})["dir"] = boxes["arch"]
    cfg.setdefault("highlight", {})["dir"] = boxes["hldir"]
    p = cfg.setdefault("pipeline", {})
    p["stages"] = dict(boxes["stages"])
    p["scan_with_audio"] = bool(boxes["with_audio"])
    cfg["runtime"]["default_batch_limit"] = int(boxes["limit"])


def _save_all_settings(cfg) -> None:
    """落盘全部设置，顺带把「高级 / 分步」里那几个控件的当前值也带上。

    音频/LLM 控件虽然画在下面的高级段，但它们同样决定一键执行的第 3 阶段
    （音频复核用什么引擎、什么设备、什么范围）。这些控件都带显式 key，
    值一定在 session_state 里 —— 直接取最新值，不必要求用户"先去点保存音频配置"。
    """
    ss = st.session_state
    ac = cfg.setdefault("audio", {})
    for k, dst in (("audio_en", "enabled"), ("audio_engine", "engine"),
                   ("audio_device", "device"), ("audio_mdir", "model_dir"),
                   ("audio_scope", "review_scope")):
        if k in ss:
            ac[dst] = ss[k]
    if "audio_sample" in ss:
        ac["sample_seconds"] = int(ss["audio_sample"])
    llm = cfg.setdefault("llm", {})
    for k, dst in (("llm_en", "enabled"), ("llm_model", "model"), ("llm_ep", "endpoint")):
        if k in ss:
            llm[dst] = ss[k]
    save_config(cfg)


def _render_pipeline(db, cfg) -> int:
    """⚡ 一键执行区（ADR 0004）—— 设置在上、按钮在下，点一次跑完整条链。

    返回「本次导入上限」，供下面的高级段复用（避免两处各画一个同名控件）。
    """
    st.markdown("##### ⚡ 一键执行")
    st.caption(
        "把 **扫描 → AI 分析 → 音频复核 → 语音归档 → 批量放行 → 画面精华剪辑"
        "（含自动处置）** 串成一条链，参数在这里一次配好、点一下跑完，"
        "终点是「回收目录就绪」，剩下的只有手动清空回收目录。\n\n"
        "顺序由依赖决定：分析先于音频复核/剪辑（它们的队列只认已判定的片段）、"
        "音频复核先于语音归档（归档吃的是复核产物）、**放行必须排在剪辑之前**"
        "（剪辑的自动处置会搬走未处置候选，反序放行就成了空操作）。")

    p = cfg.setdefault("pipeline", {})
    stages = p.setdefault("stages", {})

    st.markdown("**① 路径**")
    k1, k2 = st.columns(2)
    src = k1.text_input("视频源根目录（移动硬盘，只读）", cfg.get("source_root", ""),
                        key="pipe_src")
    trash = k2.text_input("分级回收目录（必须与源同盘）", cfg.get("trash_root", ""),
                          key="pipe_trash")
    work = k1.text_input("本地工作目录（数据库/缩略图，放 SSD）", cfg.get("work_dir", ""),
                         key="pipe_work")
    model = k2.text_input("YOLO ONNX 模型路径", cfg["analysis"].get("yolo_model_path", ""),
                          key="pipe_model")
    k3, k4 = st.columns(2)
    arch = k3.text_input("语音档案目录", str(cfg.get("archive", {}).get("dir", "")),
                         key="pipe_archive",
                         help="红线：绝不能设在源盘/回收目录所在的盘 —— 那里会被手动清空")
    hldir = k4.text_input("画面精华目录", str(cfg.get("highlight", {}).get("dir", "")),
                          key="pipe_highlight",
                          help="红线同上：精华是原片删除前的最后一道画面保险")

    st.markdown("**② 分析参数**")
    k5, k6, k7 = st.columns(3)
    thr = k5.slider("目标检测判定阈值", 0.10, 0.90,
                    float(cfg["analysis"].get("yolo_conf_threshold", 0.5)), 0.05,
                    key="pipe_conf")
    nfr = k6.slider("每个片段抽帧数", 3, 16,
                    int(cfg["analysis"].get("sample_frames", 6)), 1, key="pipe_frames")
    yolo_en = k7.checkbox("启用 YOLO 目标检测（无模型时自动降级）",
                          bool(cfg["analysis"].get("yolo_enabled", True)), key="pipe_yolo")

    st.markdown("**③ 本次要跑什么**")
    k8, k9 = st.columns([1, 1.2])
    limit = int(k8.number_input(
        "本次导入上限（只限制扫描；下游一律跑全库队列）", 1, 1000000,
        int(cfg["runtime"].get("default_batch_limit", 200)), 100, key="pipe_limit"))
    with_audio = k9.checkbox(
        "扫描时分析音频能量（明显变慢，仅在摄像头确实录音时才有意义）",
        value=bool(p.get("scan_with_audio", False)), key="pipe_scan_audio")

    st.caption("阶段开关（默认全开；去掉勾选就跳过那一步，之后的阶段照跑）：")
    on: Dict[str, bool] = {}
    for row in (jobsmod.PIPELINE_STAGES[0:3], jobsmod.PIPELINE_STAGES[3:6]):
        cols = st.columns(3)
        for c, (key, name) in zip(cols, row):
            on[key] = c.checkbox(name, value=bool(stages.get(key, True)),
                                 key=f"pipe_st_{key}")

    # 待办概览：这些数字就是"点一下会发生什么"
    _m = st.columns(6)
    _m[0].metric("待分析", _count_analyze_todo(db))
    _m[1].metric("待音频复核", db.count_audio_pending())
    _m[2].metric("待归档", db.count_archive_pending())
    _m[3].metric("待放行", db.count_release(*mover.release_params(cfg)))
    _m[4].metric("待剪辑", db.count_highlight_pending())
    _m[5].metric("待审核", db.count_review_queue())
    st.caption("「待分析」= 已导入但没跑过判定的；「待放行」= 判删候选里没有任何人物信号的"
               "（动物/车辆不拦）。放行不清空剪辑队列 —— 它们仍会被重扫兜底。")

    boxes = {"src": src, "trash": trash, "work": work, "model": model, "arch": arch,
             "hldir": hldir, "thr": thr, "nfr": nfr, "yolo_en": yolo_en,
             "limit": limit, "with_audio": with_audio, "stages": on}
    # 即改即用（只改内存里的 cfg，不写盘）—— 下面的高级段与各项前置检查读的就是这份，
    # 免得"填了路径但没点保存导致自检仍报缺失"。落盘只在两个按钮里做，
    # 否则每次控件交互都会 fsync 一次 config.json。
    _apply_pipeline_cfg(cfg, boxes)

    pj = jobsmod.get_pipeline_job()
    b1, b2 = st.columns([1, 2.4])
    with b1:
        if st.button("💾 保存设置", use_container_width=True, key="pipe_save"):
            _save_all_settings(cfg)
            st.success("已保存到 config.json（改了检测阈值需重新分析才生效）")
    with b2:
        if st.button("⚡ 一键执行（跑完到「回收目录就绪」）", type="primary",
                     use_container_width=True, key="pipe_run",
                     disabled=jobsmod.any_running()):
            _save_all_settings(cfg)          # 先落盘：保证重启/分步执行看到的是同一份配置
            opts = {"stages": on, "scan_limit": int(limit),
                    "scan_with_audio": bool(with_audio),
                    "stage_limit": int(p.get("stage_limit") or 100000)}
            ok, msg = jobsmod.start_pipeline(db, cfg, opts)
            if ok:
                st.success("已启动：可以关掉浏览器，进度在下方实时显示")
                st.rerun()
            else:
                st.error(msg)
    job_panel(pj, "⚡ 一键执行", "pipeline")
    return int(limit)


def _count_analyze_todo(db) -> int:
    return db.count_pending_analysis()


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

    limit = _render_pipeline(db, cfg)

    st.divider()
    with st.expander("🔧 高级 / 分步执行 —— 单独跑某一步、调参、重新分析", expanded=False):
        _render_step_by_step(db, cfg, limit)


def _render_step_by_step(db, cfg, limit: int):
    st.markdown("##### 执行（后台运行，不阻塞页面）")
    st.caption("任务跑在独立线程里：启动后可以随意切换页签、继续审核，进度在下方实时显示。"
               "修改阈值后请先点上面的「💾 保存设置」再启动，否则用的还是旧配置。"
               "**填好上面的设置、点了「一键执行」的，这里通常不用动。**")

    _scan_src = (cfg.get("source_root") or "").strip()
    _scan_wav = bool(cfg.get("pipeline", {}).get("scan_with_audio", False))
    busy = jobsmod.any_running()
    sj, aj = jobsmod.get_scan_job(), jobsmod.get_analyze_job()
    e1, e2 = st.columns(2)
    with e1:
        if st.button("🔍 启动扫描（后台）", use_container_width=True, disabled=busy):
            if not _scan_src or not Path(_scan_src).exists():
                st.error("请先在上面的「一键执行」区填写有效的源目录")
            else:
                cfg["runtime"]["default_batch_limit"] = int(limit)   # 记住本次上限，重启不清空
                save_config(cfg)
                ok, msg = jobsmod.start_scan(db, cfg, _scan_src, int(limit), _scan_wav)
                st.success(msg) if ok else st.warning(msg)
    with e2:
        if st.button("🤖 启动 AI 分析（后台）", use_container_width=True, disabled=busy):
            cfg["runtime"]["default_batch_limit"] = int(limit)       # 记住本次上限
            save_config(cfg)
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

    # ------------------------------------------------------------ 重新分析
    st.divider()
    st.markdown("##### 🔄 改了判定参数？重新分析")
    st.caption(
        "分析队列只处理**未分析过**的记录，所以改完阈值直接点「启动 AI 分析」"
        "不会有任何反应 —— 需要先把已分析的记录**退回待分析**状态。\n\n"
        "退回只清除机器分析结果（分数/判定/依据），**不动视频文件、不删记录**，"
        "人工处置决定（保留/移入回收）也默认保留。耗时与首次分析相当。")

    n_und = db.count_resettable("undecided")
    n_err = db.count_resettable("failed")
    n_all = db.count_resettable("all")
    scope = st.radio(
        "退回范围",
        ["undecided", "failed", "all"],
        format_func=lambda s: {
            "undecided": f"仅未人工处置的（{n_und} 条）—— 推荐，不动你的决定",
            "failed": f"仅分析失败的（{n_err} 条）—— 修好环境后重试",
            "all": f"全部（{n_all} 条，含已人工处置的）—— 会覆盖人工结论",
        }[s],
        key="reset_scope", horizontal=False)

    n_sel = {"undecided": n_und, "failed": n_err, "all": n_all}[scope]
    danger = (scope == "all")
    go = True
    if danger:
        go = st.checkbox(
            f"我明白：已人工处置的 {n_all - n_und} 条会被机器重新判定，"
            "原先标「保留」的可能被改判为「建议清理」（文件位置不变）",
            key="reset_all_ok")
    r1, r2 = st.columns([1.4, 1.6])
    with r1:
        if st.button("↩ 退回待分析", use_container_width=True,
                     disabled=(n_sel == 0 or not go or busy)):
            n = db.reset_analysis(scope)
            st.success(f"已把 **{n}** 条退回待分析状态。"
                       "现在点上面的「🤖 启动 AI 分析」即可用新参数重跑。")
            st.rerun()
    with r2:
        if st.button("🔄 退回并立即重跑分析", type="primary", use_container_width=True,
                     disabled=(n_sel == 0 or not go or busy)):
            n = db.reset_analysis(scope)
            cfg["runtime"]["default_batch_limit"] = int(limit)
            save_config(cfg)
            ok, msg = jobsmod.start_analyze(db, cfg, int(limit))
            st.success(f"已退回 **{n}** 条并启动分析（用新参数）。") if ok \
                else st.warning(msg)
            st.rerun()
    if n_sel == 0:
        st.caption("该范围内没有可退回的记录。")
    if aj.running:
        st.caption("分析任务正在运行，结束后才能退回。")

    st.divider()
    st.markdown("##### 🔊 音频复核（L2.5 有声事件捞回）")
    st.caption(
        "画面判定「无目标」的删除候选，可能画面无人但**画外有人声**（电话/对讲/询价/警报…）。"
        "音频复核 = 抽音轨 → 人声检测 → 转写文本 → LLM 判断是否有用，把有内容的片段捞回来给你看。\n"
        "复核结果是**参考**，最终仍由你在审核台决定；程序依然只区分与移动，不删除。")
    ac = cfg["audio"]
    a1, a2 = st.columns(2)
    a_en = a1.checkbox("启用音频复核", bool(ac.get("enabled")), key="audio_en")
    a_engine = a2.selectbox(
        "ASR 引擎", ["funasr", "faster_whisper"],
        index=0 if ac.get("engine", "funasr") == "funasr" else 1, key="audio_engine")
    a3, a4 = st.columns(2)
    a_sample = a3.number_input("抽音轨秒数（0 = 全量）", 0, 3600,
                               int(ac.get("sample_seconds", 0)), key="audio_sample")
    a_mdir = a4.text_input("模型目录", str(ac.get("model_dir", "")), key="audio_mdir")
    a_dev = st.selectbox(
        "推理设备",
        ["auto", "cuda", "cpu"],
        index=["auto", "cuda", "cpu"].index(str(ac.get("device", "auto"))),
        key="audio_device",
        help="auto=有 GPU 就用 GPU。若 GPU 被 DVR 转码等任务占满、或驱动异常导致复核"
             "卡住（一段几分钟不动），改成 cpu 可稳定跑完（慢 3-5 倍但不会饿死）。")
    scope_map = {"all": "待清理 + 需人工确认（全部）", "trash": "仅待清理（判定清理）",
                 "review": "仅需人工确认（边界片段）"}
    cur_scope = str(ac.get("review_scope", "all"))
    a_scope = st.radio(
        "复核范围（两类都经 音频复核 + LLM 文字判断）",
        list(scope_map.keys()),
        index=["all", "trash", "review"].index(cur_scope) if cur_scope in scope_map else 0,
        format_func=lambda k: scope_map[k],
        key="audio_scope",
        horizontal=True,
        help="待清理=分数最低、最可疑的片段；需人工确认=分数居中的边界片段，"
             "画面无人但最需要靠声音帮你判断留不留。默认全部，两类都会复核。\n"
             "若你正打算人工审核「需人工确认」的片段，可先选「仅需人工确认」"
             "单独把这一档跑完，让人工审核时每段都有转写文本可看。")
    st.markdown("**LLM 有用性判断**（需本机 ollama 已安装并拉取模型，如 qwen3:4b）")
    l1, l2, l3 = st.columns(3)
    l_en = l1.checkbox("启用 LLM 判断", bool(cfg["llm"].get("enabled")), key="llm_en")
    l_model = l2.text_input("模型", str(cfg["llm"].get("model", "qwen3:4b")), key="llm_model")
    l_ep = l3.text_input("端点", str(cfg["llm"].get("endpoint", "http://127.0.0.1:11434")),
                         key="llm_ep")

    def _apply_audio_cfg() -> None:
        """把控件当前值写回 cfg（供保存/自检/启动共用）。"""
        ac["enabled"] = a_en
        ac["engine"] = a_engine
        ac["device"] = a_dev
        ac["sample_seconds"] = int(a_sample)
        ac["model_dir"] = a_mdir
        ac["review_scope"] = str(a_scope)
        cfg["llm"]["enabled"] = l_en
        cfg["llm"]["model"] = l_model
        cfg["llm"]["endpoint"] = l_ep

    acb1, acb2, acb3 = st.columns(3)
    with acb1:
        if st.button("💾 保存音频复核配置", use_container_width=True, key="save_audio"):
            _apply_audio_cfg()
            save_config(cfg)
            st.success("已保存音频复核配置")
    with acb2:
        if st.button("🏥 引擎自检", use_container_width=True, key="check_audio"):
            _apply_audio_cfg()
            from src.audioreview import AudioReviewer
            problems = AudioReviewer(cfg).check_ready()
            if problems:
                st.warning("待处理：\n" + "\n".join(f"· {p}" for p in problems))
            else:
                st.success("引擎就绪：ASR 依赖与模型在位 ✓（LLM 判断按开关是否启用）")
    with acb3:
        if st.button("▶ 启动音频复核（后台）", use_container_width=True, key="start_audio",
                     disabled=busy):
            _apply_audio_cfg()
            cfg["runtime"]["default_batch_limit"] = int(limit)   # 记住本次上限
            save_config(cfg)
            from src.audioreview import AudioReviewer
            problems = AudioReviewer(cfg).check_ready()
            if problems:
                st.warning("引擎未就绪，未启动：\n" + "\n".join(f"· {p}" for p in problems))
            else:
                ok, msg = jobsmod.start_audio_review(db, cfg, int(limit), scope=str(a_scope))
                st.success(msg) if ok else st.warning(msg)

    # 待复核计数按范围拆开显示（两类一目了然）
    n_trash = db.count_audio_pending(scope="trash")
    n_review = db.count_audio_pending(scope="review")
    scope_txt = scope_map.get(str(a_scope), scope_map["all"])
    st.caption(
        f"当前待复核：**待清理 {n_trash}** 段 + **需人工确认 {n_review}** 段"
        f"（均有音轨、未复核、未处置）。当前范围：**{scope_txt}** —— 启动后自动跳过"
        "已复核/无音轨片段，可随时停止，进度写入数据库不丢。")

    st.divider()
    # —— L2.7 语音档案：删除视频前把"有效语音"的音频 + 转写永久收藏 ——
    st.markdown("##### 📦 语音档案")
    st.caption(
        "视频画面没有保存价值，但里面的人声以后可能有用（ASR 训练 / 资料检索）。"
        "按一次下面的按钮，把所有**待处置候选（已移入回收目录 + 判定待清理/需人工确认）"
        "中确认为有效语音**的音频复制成 `wav + 同名转写 txt`，并追加到 `manifest.jsonl`。"
        "数据源是复核缓存，**不依赖视频本体** —— 未移入回收目录的 review 桶也能归档"
        "（画外音正是它们最常藏内容的地方）。**只复制、不移动不删除**，可反复按（已归档的自动跳过）。")

    from src import archiver as armod          # noqa: E402 局部导入，避免拖慢首屏
    st.caption(f"归档目录：`{cfg.get('archive', {}).get('dir', '')}`"
               "（在顶部「一键执行 → ① 路径」里改）")

    n_pending = db.count_archive_pending()
    n_done = db.count_archived()
    st.caption(f"待归档：**{n_pending}** 段（处置候选 + 有效语音）"
               f"　已归档：**{n_done}** 段")

    for _p in armod.check_ready(cfg):          # 硬错误：目录非法，禁止执行
        st.error(f"⛔ {_p}")
    for _w in armod.check_warnings(cfg):       # 软提示：同盘风险，不阻止
        st.warning(f"💡 {_w}")
    if st.button("📦 立即归档有效语音", use_container_width=True,
                 key="start_archive",
                 disabled=busy or n_pending == 0 or bool(armod.check_ready(cfg))):
        cfgmod.save_config(cfg)
        ok, msg = jobsmod.start_archive(db, cfg, limit=5000)
        st.success(msg) if ok else st.warning(msg)
        st.rerun()

    st.divider()
    # —— L2.8 画面精华：删除前的画面留底（两级检测 + 三级产出，全自动） ——
    st.markdown("##### 🎞 画面精华剪辑")
    st.caption(
        "音频归档管声音，这里管画面：对**待处置候选**（已移入回收目录 + 判定待清理/"
        "需人工确认的 review 桶）做两级检测（3fps@640 粗扫全片 + 4fps@1280 精扫候选窗），"
        "把画面里还有点东西的几秒自动剪出来。**A/B 级切视频含原音，C 级疑似只留静帧**。\n\n"
        "**审核减负（先剪后删）**：剪完自动处置未处置候选 —— 精华占比 >60% 的自动"
        "**整片保留**（切片丢上下文，不如留原片）；其余（含确认无精华）自动**移入回收目录**，"
        "你只到「精华复核」页翻 A/B 级产物，不用逐条审原片。\n"
        "**顺序提醒**：先跑完音频复核 + 语音归档再启动本任务 —— 有音轨但还没复核过"
        "画外音的片段不会被自动移入（避免声音没留底就进回收）。"
        "纯 CPU 约每小时 120-180 段，1372+ 段预计一晚上。")

    from src import highlighter as hlmod      # noqa: E402 局部导入，避免拖慢首屏
    st.caption(f"精华目录：`{cfg.get('highlight', {}).get('dir', '')}`"
               "（在顶部「一键执行 → ① 路径」里改）")

    _hl_problems = hlmod.check_ready(cfg)
    for _p in _hl_problems:                    # 硬错误：缺 ffmpeg/模型、目录非法
        st.error(f"⛔ {_p}")
    for _w in hlmod.check_warnings(cfg):       # 软提示：同盘风险，不阻止
        st.warning(f"💡 {_w}")
    _n_hl_pending = db.count_highlight_pending()
    _hl_stats = db.highlight_stats()
    st.caption(f"待剪辑：**{_n_hl_pending}** 段（含 review 桶）"
               f"　已出结果：**{_hl_stats['done']}** 段"
               f"（切出 {_hl_stats['clips']} 段 / {_hl_stats['secs'] / 60:.0f} 分钟）"
               f"　失败：**{_hl_stats['failed']}** 段")
    if st.button("🎞 启动画面精华剪辑（后台）", use_container_width=True,
                 key="start_highlight",
                 disabled=busy or _n_hl_pending == 0 or bool(_hl_problems)):
        cfgmod.save_config(cfg)
        ok, msg = jobsmod.start_highlight(db, cfg, limit=5000)
        st.success(msg) if ok else st.warning(msg)
        st.rerun()

    _n_hl_err = _hl_stats["failed"]
    if _n_hl_err:
        if st.button(f"↩ 重置 {_n_hl_err} 条失败记录（重跑时再试）",
                     key="reset_highlight_failed"):
            n = db.reset_highlight("failed")
            st.success(f"已重置 {n} 条失败记录。")
            st.rerun()

    # —— Track C · C5 重剪：新口径对**已剪过**的产物回溯生效 ——
    _n_re_first = jobsmod.count_rehighlight_pending(db, cfg, only_dropped=True)
    _n_re_all = jobsmod.count_rehighlight_pending(db, cfg, only_dropped=False)
    if _n_re_all:
        with st.container(border=True):
            st.markdown("##### ♻️ 按新口径重剪（删前保全）")
            st.caption(
                "这些视频是用**旧口径**剪的：粗扫看到过目标、却被精扫「否决」后丢弃，"
                "结果零产物；或者 C 级只留了一张静帧。重剪一次就会补出**可播放的视频**。\n\n"
                f"第一批（收益最集中）**{_n_re_first}** 条 → 约 "
                f"{_n_re_first * 6.1 / 3600:.1f} 小时；"
                f"全量 **{_n_re_all}** 条 → 约 {_n_re_all * 6.1 / 3600:.1f} 小时"
                "（按实测均值 6.1s/条估）。**幂等**：跑过的不会重来，随时可停可续。")
            _only_first = st.checkbox(
                "只重剪第一批（零产物 + 只有静帧的）", value=True,
                key="re_hl_only_first",
                help="取消勾选则全量重剪到新口径，产物会统一规范，但耗时更长")
            if st.button(f"♻️ 启动重剪（{_n_re_first if _only_first else _n_re_all} 条，后台）",
                         use_container_width=True, key="start_rehighlight",
                         disabled=busy or bool(_hl_problems)):
                cfgmod.save_config(cfg)
                ok, msg = jobsmod.start_rehighlight(db, cfg, limit=100000,
                                                     only_dropped=_only_first)
                st.success(msg) if ok else st.warning(msg)
                st.rerun()

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

    t1, t2, t3, t4, t5, t6 = st.tabs(
        ["审核工作台", "批量浏览", "精华复核", "执行进度", "误判分析", "设置与扫描"])
    with t1:
        tab_review(db, cfg)
    with t2:
        tab_grid(db, cfg)
    with t3:
        tab_highlight(db, cfg)
    with t4:
        tab_progress(db, cfg)
    with t5:
        tab_feedback(db, cfg)
    with t6:
        tab_settings(db, cfg)


if __name__ == "__main__":
    main()
