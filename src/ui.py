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
            # 否则用户看到 3/7 的进度条会误以为整条链才跑了 3/7。
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
        elif job.kind == "rehighlight":
            # 重剪的核心指标是「救回」：零产物→有产物（rescued）/ 只有静帧→有视频（upgraded）
            st.success(
                f"{icon} 精华重剪完成：处理 {r.get('done', 0) + r.get('empty', 0)} 段 —— "
                f"救回 {r.get('rescued', 0)} 段（零产物→有产物），"
                f"升级 {r.get('upgraded', 0)} 段（只有静帧→有视频），"
                f"切出视频 {r.get('clips', 0)} 段，无精华 {r.get('empty', 0)} 段，"
                f"失败 {r.get('failed', 0)} 段，用时 {fmt_eta(job.elapsed)}")
            if r.get("failed"):
                with st.expander("失败明细"):
                    for e in (r.get("errors") or [])[:20]:
                        st.caption(f"· {e}")
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


# --------------------------------------------------------------------------- 音频摘要行（可复用）
# 原挂在「批量浏览」九宫格上，那页已随全量审核导航删除（ADR 0007）。
# 但这段本身是**可复用资产**：格子/列表很窄时转写必须限高截断、全文挂 title
# 由浏览器 hover 弹出，比塞进正文可读得多。抽查页与将来的列表视图还会用它。
_AUDIO_LINE_CSS = """
<style>
.g-aud{font-size:.76rem;line-height:1.35;margin:2px 0 5px;word-break:break-all;
  display:-webkit-box;-webkit-line-clamp:3;-webkit-box-orient:vertical;overflow:hidden}
.g-aud-sp{color:#ff9f1c;font-weight:600}
.g-aud-none{color:#8892a0}
.g-aud-wait{color:#8892a0;font-style:italic}
</style>
"""
# 兼容旧名（测试与历史引用）
_GRID_AUDIO_CSS = _AUDIO_LINE_CSS

# LLM 有用性在三态卡片里的紧凑标记（与审核台一致）
_LLM_TAG = {1: "🟢有用", 0: "⚪无用", 2: "🟡不定"}


def _grid_audio_html(row) -> str:
    """音频信息行：有人声则高亮「占比 + 转写」，否则说明状态。

    四种状态：无音轨（返回空串，整行不渲染）/ 未复核 / 跳过 / 已复核。
    无音轨返回空串是为了不给静音片段加噪音。
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


def _spot_step(key: str, delta: int) -> None:
    """抽查批次内的前后翻页。夹在 [0, 批大小-1]。"""
    cur = int(st.session_state.get(f"{key}_i", 0))
    n = len(st.session_state.get(f"{key}_batch", []) or [1])
    st.session_state[f"{key}_i"] = max(0, min(cur + delta, n - 1))


def _spot_reroll(key_prefix: str) -> None:
    """「换一批」：种子 +1。

    必须走 `on_click` 回调而不是在按钮分支里 `st.session_state[...] = ...` ——
    Streamlit **禁止在 widget 实例化之后改它自己的 session_state 键**
    （`StreamlitWidgetAlreadyInstantiatedError`）。回调在 widget 构造前注册，
    触发时先更新值再 rerun，是唯一合法的做法。
    """
    st.session_state[f"{key_prefix}_seed"] = \
        int(st.session_state.get(f"{key_prefix}_seed", 1)) + 1
    st.session_state[f"{key_prefix}_i"] = 0


def _spot_sample_bar(db, key_prefix: str, pool: Dict[str, int],
                     total: int, extra: str = "") -> int:
    """抽查工具条：抽多少条 + 随机种子 + 换一批。返回种子。

    **「换一批」= seed + 1 而不是随机数** —— 这样"换一批"是可回退的
    （把种子手填回上一个数就回到上一批），而真随机做不到这点。
    抽查的核心用途是"多人对齐口径"或"复查某条"，两样都要求可复现。
    """
    c1, c2, c3, c4 = st.columns([1, 1, 1.3, 1.4])
    with c1:
        n = st.number_input("抽多少条", min_value=1, max_value=200,
                            value=10, step=1, key=f"{key_prefix}_n",
                            disabled=total == 0)
    with c2:
        seed = st.number_input("随机种子", min_value=0, max_value=99999,
                               value=1, step=1, key=f"{key_prefix}_seed",
                               help="同一个种子 + 同一份库 = 同一批样本。"
                                    "想复查某条就把种子填回原来的值。")
    with c3:
        st.metric("抽样池", total)
    with c4:
        st.markdown(" ")
        st.button("🎲 换一批", key=f"{key_prefix}_reroll",
                   use_container_width=True, disabled=total == 0,
                   on_click=_spot_reroll, args=(key_prefix,))
    if total == 0:
        st.warning("抽样池是空的 —— 没有符合条件的行可抽。")
    if extra:
        st.caption(extra)
    return int(seed)


def _spot_batch_state(key: str, seed: int) -> List[Any]:
    """取当前批样本。**必须落 session_state** ——
    否则每次 rerun 都重抽，点「下一条」时样本会漂移（抽到的和看到的不是一条）。"""
    return st.session_state.get(f"{key}_batch", [])


def tab_review(db, cfg):
    st.subheader("抽查")
    st.caption(
        "程序已经把该做的做完了：扫描 → 分析 → 音频复核 → 语音档案 → 放行 → 精华剪辑 → "
        "分堆，**全自动**（一键执行跑完整条链）。这一页只剩一件事：**随机抽几条，"
        "看看机器判得准不准**。\n\n"
        "**删除始终由你手动做** —— 程序只创建和移动，永不删除。")

    entry = st.radio("抽查什么", ["🎯 L1 判定结果", "✂️ 已剪的 A/B 精华"],
                     horizontal=True, key="spot_entry")
    st.divider()
    if entry.startswith("🎯"):
        _spot_l1(db, cfg)
    else:
        _spot_clips(db, cfg)


def _spot_l1(db, cfg) -> None:
    """抽查入口 1：L1 判定结果。跨 verdict 分层抽样。

    配额 trash:review:keep = 4:3:3 —— trash 加倍是因为那些**人已经被机器处置**
    了，误判代价最大；keep 是 L1 漏洞最大的区（旧公式下 conf 0.4~0.6 全部
    误判 keep），也必须抽到。
    """
    pool = db.count_sample_pool()
    st.markdown("### 🎯 L1 判定结果")
    quota = {"trash": 4, "review": 3, "keep": 3}
    st.caption("按判定结果分层抽样："
               + "、".join(f"{k} {v}" for k, v in quota.items())
               + "（trash 加倍 —— 那些条已经被机器处置了，误判代价最大）。"
               f"　当前池：keep {pool['keep']} / review {pool['review']} / "
               f"trash {pool['trash']}。")
    seed = _spot_sample_bar(db, "sp_l1", pool, pool["total"],
                            extra="只抽 `quality_v2` 口径判出的行 —— "
                                  "旧公式的样本混进来会把「公式不准」和"
                                  "「样本不是这个公式判的」混为一谈。")
    if pool["total"] == 0:
        return

    # 分层抽样：各层用 seed 派生子种子，保证"同 seed 必得同一批"
    import random as _rnd
    picked: List[int] = []
    want = int(st.session_state.get("sp_l1_n", 10))
    with db.conn() as c:
        for i, (v, q) in enumerate(quota.items()):
            k = max(1, round(want * q / sum(quota.values())))
            rows = db.sample_ids(seed=seed + i * 1000, n=k, l1_only=True)
            picked += [r["id"] for r in rows if r["verdict"] == v]
        picked = _rnd.Random(seed).sample(picked, min(len(picked), want)) \
            if len(picked) > want else picked

    st.session_state["sp_l1_batch"] = picked
    if not picked:
        st.info("这一批没抽到行（池太小或种子恰好落空），点「换一批」试试。")
        return
    _spot_pager(db, cfg, picked, "sp_l1")


def _spot_clips(db, cfg) -> None:
    """抽查入口 2：已剪出的 A/B 级精华。按**片段**抽，不是按视频抽。"""
    pool = db.count_sample_pool_ab()
    st.markdown("### ✂️ 已剪的 A/B 精华")
    seed = _spot_sample_bar(db, "sp_ab", pool, pool["total"],
                            extra="按**片段**抽而不是按视频抽 —— 一条视频可能剪出 5 段，"
                                  "按视频抽会让「剪得最多的那条」占掉大半样本，"
                                  "而那正是机器最自信的部分。")
    if pool["total"] == 0:
        return
    picked = db.sample_ab_clips(seed=seed, n=int(st.session_state.get("sp_ab_n", 10)))
    st.session_state["sp_ab_batch"] = picked
    if not picked:
        st.info("这一批没抽到片段，点「换一批」试试。")
        return

    batch = picked
    st.markdown(f"本批 **{len(batch)}** 段：")
    st.dataframe(
        [{"#": i + 1, "视频": b["video_id"], "级别": b.get("grade"),
          "时长(秒)": b.get("dur"), "置信度": b.get("conf"),
          "拍到的人": b.get("n_person"),
          "文件": (b.get("file") or "")[-40:]}
         for i, b in enumerate(batch)],
        hide_index=True, use_container_width=True)
    st.divider()
    for i, b in enumerate(batch, 1):
        _spot_clip_one(db, cfg, b, i, len(batch))


def _spot_clip_one(db, cfg, clip, idx: int, total: int) -> None:
    """一段精华的详情 + 改级别。"""
    f = clip.get("file")
    head = f"**{idx}/{total}** · {clip.get('grade')} 级 · {clip.get('dur')} 秒"
    with st.expander(head, expanded=(idx == 1)):
        left, right = st.columns([3, 2])
        with left:
            if f and Path(f).exists():
                st.video(f)
            elif f:
                st.warning(f"产物文件已不在：{f}")
            else:
                st.info("这一段没有 mp4（只有静帧）")
            if f and Path(f).exists():
                b1, b2 = st.columns(2)
                with b1:
                    if st.button("▶ 播放", key=f"sp_play_{clip['video_id']}_{clip['index']}",
                                 use_container_width=True):
                        open_in_player(f)
                with b2:
                    if st.button("📂 定位", key=f"sp_loc_{clip['video_id']}_{clip['index']}",
                                 use_container_width=True):
                        reveal_in_explorer(f)
        with right:
            st.json({
                "视频": clip.get("video_id"), "级别": clip.get("grade"),
                "起止(秒)": [clip.get("start"), clip.get("end")],
                "置信度": clip.get("conf"), "拍到的人": clip.get("n_person"),
                "分数": clip.get("score"), "产出口径": clip.get("ver"),
            })
            g1, g2 = st.columns(2)
            with g1:
                if st.button("⬆️ 判高了 → 降为 B", key=f"sp_db_{clip['video_id']}_{clip['index']}",
                             use_container_width=True,
                             disabled=clip.get("grade") == "B"):
                    _spot_regrade(db, cfg, clip, "B")
            with g2:
                if st.button("⬇️ 判低了 → 升为 A", key=f"sp_ua_{clip['video_id']}_{clip['index']}",
                             use_container_width=True,
                             disabled=clip.get("grade") == "A"):
                    _spot_regrade(db, cfg, clip, "A")
            st.caption("改级别只改**这一段**。若降级后该原片已无 A/B 产物，"
                       "它会从回收目录的「已剪辑」挪回顶层档位 —— "
                       "因为「已剪辑」的判据是这个原片有 A/B 级 mp4 在。")


def _spot_regrade(db, cfg, clip, grade: str) -> None:
    """改一段精华的级别，并按需重排子档。

    ⚠️ clips 变了 ⇒ `mover.recycle_state()` 的判据变了 ⇒ 原片该待在哪个子档
    也变了。db.set_clip_grade 已经同步了 highlight_best_grade / ab_clips_count
    （不这么做，"已剪辑"标记会失真 ⇒ 机主以为可删、实际 A 级 mp4 还在）。
    这里还要再调一次 align_one 把文件挪到正确的子档。
    """
    vid, idx = clip["video_id"], clip["index"]
    if not db.set_clip_grade(vid, idx, grade):
        st.error("改级别失败（片段索引可能已失效）")
        st.rerun()
    row = db.get_video(vid)
    if row and row["trash_path"]:
        try:
            pending = db.pending_highlight_ids()
        except Exception:  # noqa: BLE001 —— 抽样的辅助路径，失败不该炸页面
            pending = None
        ok, msg = mover.align_one(db, row, cfg, pending_ids=pending)
        st.toast(msg if ok else f"级别已改，但子档未重排：{msg}",
                 icon="✅" if ok else "⚠️")
    st.rerun()


def _spot_pager(db, cfg, batch: List[int], key: str) -> None:
    """在抽中的样本里翻页看（复用展示组件，不复用审核台的排序/跳转）。"""
    i = int(st.session_state.get(f"{key}_i", 0))
    i = max(0, min(i, len(batch) - 1))
    st.session_state[f"{key}_i"] = i
    c1, c2, c3 = st.columns([1, 2, 1])
    with c1:
        st.button("← 上一条", key=f"{key}_prev", use_container_width=True,
                  disabled=i == 0, on_click=_spot_step, args=(key, -1))
    with c2:
        st.markdown(f"<div style='text-align:center'>本批第 <b>{i+1}</b> / "
                    f"{len(batch)} 条</div>", unsafe_allow_html=True)
    with c3:
        st.button("下一条 →", key=f"{key}_next", use_container_width=True,
                  disabled=i >= len(batch) - 1,
                  on_click=_spot_step, args=(key, 1))

    row = db.get_video(batch[i])
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
            if st.button("▶ 用系统播放器打开", key=f"{key}_play_{row['id']}",
                         use_container_width=True):
                vp = current_video_path(row)
                if vp:
                    open_in_player(str(vp))
                else:
                    st.error("原文件已不存在")
        with b2:
            if st.button("🌐 生成网页可播预览", key=f"{key}_pv_{row['id']}",
                         use_container_width=True):
                with st.spinner("转码中（H.265 → H.264，约需数秒）..."):
                    pv = make_browser_preview(row, cfg)
                if pv:
                    st.session_state[f"pv_{row['id']}"] = str(pv)
                else:
                    st.error("转码失败")
        if f"pv_{row['id']}" in st.session_state:
            st.video(st.session_state[f"pv_{row['id']}"])
    with right:
        st.markdown("### 为什么这么判定")
        render_reasons(row, cfg)
        # 音频摘要行（可复用资产，原挂九宫格；抽查时"画外说了什么"同样要紧）
        st.markdown(_AUDIO_LINE_CSS, unsafe_allow_html=True)
        st.markdown(_grid_audio_html(row) or "", unsafe_allow_html=True)
        render_audio_card(row, cfg)

    st.divider()
    st.markdown("### 抽查处置")
    _spot_actions(db, cfg, row, key)


def _spot_actions(db, cfg, row, key: str) -> None:
    """抽查的四个动作：判对了 / 该留 / 该删 / 记为误判。

    与旧审核台的三个按钮相比，**核心区别是"该留"能掉头**：
    旧审核台的"误判保留"只写 keep 不搬文件，于是原片留在回收目录里 ——
    机主后来清空回收目录时它一起没了。抽查看到"这个不该删"时，正确的反应
    是**立刻把它搬回源目录**，因为那时它还在。

    ⚠️ meta_ok=0 的行**不能**开「该删」：解码失败行（私有编码）的 verdict='keep'
    是"解不开就绝不判删"的技术兜底，而 `_PENDING_OR_TRASHED` 的 A 分支
    （human_decision='trash' AND trash_path IS NOT NULL）**没有 meta_ok=1** ——
    不设 disabled 就能把"出于安全保留"的行移进回收目录。
    """
    from src import feedback as fbmod

    vid = row["id"]
    is_meta_ok = bool(row["meta_ok"])
    has_trash = bool(row["trash_path"])
    decided = row["human_decision"]

    st.caption(f"当前机器处置："
               + ("未处置" if decided is None else
                  ("已移入回收目录" if decided == "trash" else "判为保留"))
               + (f"　·　本行由**{'程序' if row['decided_by'] == 'auto' else ('人' if row['decided_by'] == 'human' else '历史数据')}**"
                  "决定" if "decided_by" in row.keys() else ""))

    # 音频未复核 → 提示但**不拦**。
    #
    # 为什么不再拦（旧审核台会 `disabled=not confirm_skip_audio`）：
    # 音频复核队列早已放宽为「处置候选」—— 移入回收目录后仍能补复核
    # （audioreview.review_one 有 trash_path 回落），补完照常落成语音档案。
    # 所以"半拦半不拦"（程序路径不拦、UI 路径拦）反而让一键执行的终点不可预期，
    # `_dispose_one` 早已因此撤掉了同一道守卫。这里保持同口径：
    # **给提示（画外有人声就丢了这���信息值得知道），不阻断**。
    # 唯一的硬期限是"手动清空回收目录"之前必须把复核+归档跑完。
    if bool(row["has_audio"]) and row["audio_checked"] is None \
            and cfg.get("audio", {}).get("enabled"):
        st.warning("⚠️ 此片段**有音轨但尚未音频复核**。若画外有人声，它可能是你要留的"
                   "事件记录。移入回收目录后仍可补复核（只要还没手动清空回收目录）。")
    elif row["audio_checked"] == 1 and row["speech_present"] and row["llm_useful"] == 1:
        st.info("该片段音频复核**检出人声**且 LLM 认为可能有用 —— 若你确认这段没有保留"
                "价值，仍可移走；此提示仅为避免误删。")

    a1, a2, a3, a4 = st.columns(4)
    with a1:
        if st.button("✅ 判对了", key=f"{key}_ok_{vid}", use_container_width=True):
            db.log_action(vid, "spotcheck_ok", "", "", row["batch_id"])
            st.toast("已记：判对了", icon="✅")
            st.rerun()
    with a2:
        if st.button("⬆️ 判错了，该留", key=f"{key}_keep_{vid}",
                     type="primary", use_container_width=True):
            if has_trash:
                ok, msg = mover.undo_move(db, row)
                if ok:
                    st.toast(f"已搬回源目录：{msg}", icon="↩")
                else:
                    # 回收路径上的文件已经没了（被手动删过？）—— 只改判定，
                    # 但必须显式说清楚，否则行会带着失效的 trash_path，
                    # 空间统计会错报"这个文件在回收目录里"
                    db.set_human_decision(vid, "keep", decided_by="human")
                    st.warning(f"搬回失败（{msg}）。已改为「判为保留」，"
                               f"但原文件已不在回收目录 —— 请你确认它还在别处。")
            else:
                db.set_human_decision(vid, "keep", decided_by="human")
                st.toast("已记：判为保留", icon="↩")
            st.rerun()
    with a3:
        if st.button("⬇️ 判错了，该删", key=f"{key}_trash_{vid}",
                     use_container_width=True,
                     disabled=not is_meta_ok or decided == "trash"):
            if not cfg.get("trash_root"):
                st.error("请先在「设置与扫描」配置 trash_root")
            elif not has_trash:
                tier = mover.decide_tier(row, cfg)
                ok, msg = mover.move_one(db, row, cfg, tier, by="human")
                st.toast(msg if ok else msg, icon="🗑" if ok else "⚠️")
                st.rerun()
            else:
                st.info("已在回收目录里了，无需再移。")
        if not is_meta_ok:
            st.caption("⚠️ 该片段分析失败（多半是私有编码解不开）。"
                       "按安全兜底它永远判「保留」，不允许从这里移走。")
    with a4:
        if st.button("🏷️ 记为误判（留证据）", key=f"{key}_fb_{vid}",
                     use_container_width=True):
            analyzer = Analyzer(db, cfg)
            r = fbmod.record_feedback(db, row, "keep", cfg,
                                      yolo_on=analyzer.yolo_on,
                                      note="抽查发现")
            st.session_state["last_fb"] = {
                "vid": vid, "name": row["filename"], "result": r}
            st.toast("已记入误判分析页（不改变处置结论）", icon="🏷️")

    with st.expander("该文件的操作历史"):
        with db.conn() as c:
            logs = c.execute(
                "SELECT * FROM action_log WHERE video_id=? ORDER BY id DESC",
                (vid,)).fetchall()
        if logs:
            st.dataframe([{"时间": l["created_at"], "动作": l["action"],
                           "从": l["from_path"], "到": l["to_path"]} for l in logs],
                         hide_index=True, use_container_width=True)
        else:
            st.caption("（无记录）")



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
    _rj = jobsmod.get_rehighlight_job()
    if _rj.status != "idle":
        job_panel(_rj, "♻️ 画面精华重剪", "rehighlight_progress")

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
    _rj2 = jobsmod.get_rehighlight_job()
    if _rj2.status != "idle":
        job_panel(_rj2, "♻️ 画面精华重剪", "rehighlight_page")

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
        "（含自动处置）→ 回收目录三堆对齐** 串成一条链，参数在这里一次配好、"
        "点一下跑完，终点是「回收目录就绪」，剩下的只有手动清空回收目录。\n\n"
        "顺序由依赖决定：分析先于音频复核/剪辑（它们的队列只认已判定的片段）、"
        "音频复核先于语音归档（归档吃的是复核产物）、**放行必须排在剪辑之前**"
        "（剪辑的自动处置会搬走未处置候选，反序放行就成了空操作）、"
        "**三堆对齐必须排在最后**（它按保全状态归档，早于剪辑跑等于白搬一次）。")

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
    for row in (jobsmod.PIPELINE_STAGES[0:4], jobsmod.PIPELINE_STAGES[4:8]):
        cols = st.columns(4)
        for c, (key, name) in zip(cols, row):
            on[key] = c.checkbox(name, value=bool(stages.get(key, True)),
                                 key=f"pipe_st_{key}")

    # 待办概览：这些数字就是"点一下会发生什么"。
    # 「待审核」用 hide_c_archive=True（审核台的默认视图：滤掉「仅 C 级档案」）
    # —— 之前这里传的是空参，与审核台实际显示的队列规模天然不等，
    # 是 ADR 0005「三处逐字一致」被破掉的最后一个实例。
    _m = st.columns(7)
    _m[0].metric("待分析", _count_analyze_todo(db))
    _m[1].metric("待音频复核", db.count_audio_pending())
    _m[2].metric("待归档", db.count_archive_pending())
    _m[3].metric("待放行", db.count_release(*mover.release_params(cfg)))
    _m[4].metric("待剪辑", db.count_highlight_pending())
    _m[5].metric("待审核", db.count_review_queue(hide_c_archive=True))
    _m[6].metric("待归位", sum(mover.recycle_stats(db, cfg)["to_align"].values()))
    st.caption("「待分析」= 已导入但没跑过判定的；「待放行」= 判删候选里没有任何人物信号的"
               "（动物/车辆不拦）。放行不清空剪辑队列 —— 它们仍会被重扫兜底。"
               "「待归位」= 回收目录里没待在正确子档的条数（对齐阶段会搬它们）。")

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


def _render_scoring_params(cfg) -> None:
    """判定参数：口径切换（legacy / quality_v2）+ 三档阈值 + 抽帧数。

    为什么这一段要暴露在 UI 上而不是只改 config.json：口径切换与阈值是**一对**，
    换公式不换阈值会让 trash 几乎不可达（v2 是 0-100 质量分，而 trash_threshold=0
    是旧 125 分制的口径）。让人在同一个页面里同时看到两者 + 一条校验提示，
    比让人自己去猜"这两个要一起改"可靠。

    ⚠️ 改完**不会**自动重跑 —— 下面的「退回待分析」是独立一步。
    这里只在参数自洽时保存；不自洽（quality_v2 + trash_threshold<=0）时给警告
    但仍允许保存（机主可能就是要先看效果）。
    """
    sc = cfg.setdefault("scoring", {})
    vd = cfg.setdefault("verdict", {})
    an = cfg.setdefault("analysis", {})

    c1, c2 = st.columns([1, 1])
    with c1:
        mode = st.selectbox(
            "打分口径", ["quality_v2", "legacy"],
            index=0 if str(sc.get("mode", "legacy")) == "quality_v2" else 1,
            key="sc_mode",
            help="quality_v2 = 多帧一致性 + 质量扣分（推荐）。\n"
                 "legacy = 旧公式（单帧最高 conf），保留用于对比与回滚。\n"
                 "切换后**必须**重跑分析才生效。")
        st.caption("**quality_v2 修掉的核心问题**：旧公式下 conf ≥ 0.6 一律判保留，"
                   "而检测阈值是 0.4 —— 0.4~0.6 之间的真人全部被误判为「有价值」，"
                   "永不进任何队列，精华剪辑的重扫也就无从触发。")
    with c2:
        n_frames = st.number_input(
            "每片段抽帧数", min_value=3, max_value=60,
            value=int(an.get("sample_frames", 6) or 6), step=1,
            key="sc_frames",
            help="quality_v2 的多帧一致性靠这个数：抽得越少，"
                 "一致性判据越不可靠。**低于 12 基本失效**，建议 15。")

    t1, t2, t3 = st.columns(3)
    with t1:
        trash_thr = st.number_input(
            "建议清理 ≤", min_value=0.0, max_value=200.0,
            value=float(vd.get("trash_threshold", 0) or 0), step=1.0,
            key="sc_trash", help="分数低于此值 ⇒ 建议清理。")
    with t2:
        review_thr = st.number_input(
            "需人工确认 <", min_value=0.0, max_value=300.0,
            value=float(vd.get("review_threshold", 30) or 30), step=1.0,
            key="sc_review",
            help="分数低于此值但高于「建议清理」⇒ 进队列。\n"
                 "⚠️ 旧公式是 125 分制（满分约 125），quality_v2 是 0-100 质量分，"
                 "**两套口径的阈值不能混用**。")
    with t3:
        rel_thr = st.number_input(
            "批量放行 ≤", min_value=0.0, max_value=200.0,
            value=float(vd.get("release_threshold", 30) or 30), step=1.0,
            key="sc_release",
            help="「机器没看到任何人」且分数不高于此值 ⇒ 免审核移入回收目录。"
                 "quality_v2 的起点值是 20。")

    changed = (str(sc.get("mode", "legacy")) != mode
               or float(vd.get("trash_threshold", 0) or 0) != float(trash_thr)
               or float(vd.get("review_threshold", 30) or 30) != float(review_thr)
               or float(vd.get("release_threshold", 30) or 30) != float(rel_thr)
               or int(an.get("sample_frames", 6) or 6) != int(n_frames))

    # 口径与阈值的自洽检查（config._warn_sample_frames 里有同款逻辑，
    # 这里是 UI 侧的即时反馈，让人不保存就看见问题）
    if mode == "quality_v2":
        if float(trash_thr) <= 0:
            st.warning("⚠️ quality_v2 是 **0-100 质量分**，而「建议清理 ≤」是 0 —— "
                       "这会让「建议清理」几乎不可达（只有恰好 0 分才算）。"
                       "起点值建议 **20**。")
        if int(n_frames) < 12:
            st.warning(f"⚠️ 只抽 {int(n_frames)} 帧时，多帧一致性判据基本失效"
                       f"（分数照算，但不可信）。建议提到 **12** 以上。")

    if st.button("💾 保存判定参数", disabled=not changed, use_container_width=True):
        sc["mode"] = mode
        vd["trash_threshold"] = float(trash_thr)
        vd["review_threshold"] = float(review_thr)
        vd["release_threshold"] = float(rel_thr)
        an["sample_frames"] = int(n_frames)
        save_config(cfg)
        st.success("已保存。**已分析的记录不会自动改变** —— "
                   "要生效请在下一段点「退回待分析」再重跑。")
        st.rerun()


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

    # ------------------------------------------------------------ 判定参数
    st.divider()
    st.markdown("##### 🎚 判定参数（改完记得「退回待分析」再重跑）")
    _render_scoring_params(cfg)

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
        "把画面里还有点东西的几秒自动剪出来。**A/B/C 三级都切可播放视频**"
        "（含原音）—— C 级是**档案**不是待办，所以「多留」不会变成「多看」。\n\n"
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
    # 磁盘空间：硬闸门 —— 不足时不阻止启动按钮（由 run_batch 抛错），但先说清楚，
    # 否则用户只会看到"启动失败"却不知道要先清空间。
    _hl_no_room = hlmod.disk_block_reason(cfg)
    if _hl_no_room:
        st.error(f"⛔ {_hl_no_room}")
    _n_hl_pending = db.count_highlight_pending()
    _hl_stats = db.highlight_stats()
    st.caption(f"待剪辑：**{_n_hl_pending}** 段（含 review 桶）"
               f"　已出结果：**{_hl_stats['done']}** 段"
               f"（切出 {_hl_stats['clips']} 段 / {_hl_stats['secs'] / 60:.0f} 分钟）"
               f"　失败：**{_hl_stats['failed']}** 段")
    _hl_free = hlmod.free_gib(hlmod.highlight_dir_of(cfg))
    if _hl_free is not None:
        st.caption(f"精华目录剩余空间：**{_hl_free:.1f} GiB**"
                   f"（安全阈值 {hlmod.MIN_FREE_GIB:.0f} GiB）")
    if st.button("🎞 启动画面精华剪辑（后台）", use_container_width=True,
                 key="start_highlight",
                 disabled=busy or _n_hl_pending == 0 or bool(_hl_problems)):
        cfgmod.save_config(cfg)
        ok, msg = jobsmod.start_highlight(db, cfg, limit=5000)
        st.success(msg) if ok else st.warning(msg)
        st.rerun()

    _n_hl_err = _hl_stats["failed"]
    if _n_hl_err:
        # 失败行分两种，处理方式完全不同，不能只给一个"清 error"的按钮：
        #  ① 零产物（highlight_count=0）—— 多半是磁盘满/中断这类**半成品**，
        #     它的 highlight_at 与 highlight_version 已经写进去了，两条队列
        #     都会把它挡在外面，光清 error 无效 ⇒ 必须用 failed_retry 整组退回。
        #  ② 已有产物 —— 那不是"没剪完"，重跑只会浪费机时，让它在已剪辑里躺着。
        #
        # ⚠️ ①的按钮**不在这里** —— 它挪到了下面「🗂 回收目录三堆」卡片里。
        # 理由：它的目的是让回收目录的「待保全」归零（那些原片删不得），属
        # 回收目录的补救动作；挂在剪辑段里，机主从"为什么待保全不清零"查起
        # 要翻过整段剪辑参数才能找到，方向是反的。
        _n_retry = db.count_failed_retryable()
        if _n_retry:
            st.warning(
                f"⚠️ {_n_retry} 条**零产物**的失败记录（多半是磁盘满或任务中断）"
                "被挡在两条队列之外，光清 error 无效。"
                f"**退回入口在下方「🗂 回收目录三堆」卡片里**"
                f"（那里同时管着「待保全」这一堆 —— 退回来重剪就是为了让它归零）。")
        if _n_hl_err > _n_retry:
            with st.expander(f"其余 {_n_hl_err - _n_retry} 条失败记录（已有产物，非零产物）"):
                st.caption(
                    "这些行已经剪出过片段，**不占待重跑队列**（`highlight_count>0`）。"
                    "若确实要连已有产物一起重来，用下面的「全部重来」—— 它会让全部已剪行"
                    "重新排队，代价较大。")
                if st.button("↩ 全部重来（所有已剪行重新排队）",
                             key="reset_highlight_all"):
                    n = db.reset_highlight("all")
                    st.success(f"已重置 {n} 条。下次剪辑会全量重跑。")
                    st.rerun()

    # —— Track C · C5 重剪：只针对「零产物 / 只剩静帧」这两类**症状** ——
    # 不做「全量重剪」是**取舍**，不是"重剪没用"：全量把版本落后但产物正常的行
    # 也收进来，量级约是症状行的 5 倍（数小时 vs 数十小时），而收益集中在症状行。
    # ⚠️ 这里早先写过"同素材同参数重剪必然得到同样结果，白烧机时"——**已被实测
    #    推翻**（见 docs/自动剪辑方案说明.md §5.3/§5.4）：产物为空是当时那版聚合
    #    逻辑吃掉的，不是"没东西"。别再拿那条理由把全量入口焊死。
    _n_re = jobsmod.count_rehighlight_pending(db, cfg, only_dropped=True)
    _n_re_all = jobsmod.count_rehighlight_pending(db, cfg, only_dropped=False)
    if _n_re:
        with st.container(border=True):
            st.markdown("##### ♻️ 重新审核「零产物 / 只有静帧」的视频")
            st.caption(
                "这两类是用**旧口径**剪过的，重剪一次就能补出**可播放的视频**：\n\n"
                "- **零产物** —— 粗扫明明看到过目标，却被精扫「否决」后丢弃，"
                "删除后一个字都不留（旧口径的漏检高发区）；\n"
                "- **C 级只有一张静帧** —— 当时没切视频，光靠一张图没法回放。\n\n"
                f"共 **{_n_re}** 条 → 约 {_n_re * 6.1 / 3600:.1f} 小时"
                "（按实测均值 6.1s/条估）。**幂等**：跑过的不会重来，随时可停可续。")
            if st.button(f"♻️ 启动重剪（{_n_re} 条，后台）",
                         use_container_width=True, key="start_rehighlight",
                         disabled=busy or bool(_hl_problems)):
                cfgmod.save_config(cfg)
                ok, msg = jobsmod.start_rehighlight(db, cfg, limit=100000,
                                                     only_dropped=True)
                st.success(msg) if ok else st.warning(msg)
                st.rerun()
            # 重剪在后台线程里跑，本页只在 rerun 时刷一次快照 —— 实时面板在「执行进度」页
            _rj = jobsmod.get_rehighlight_job()
            if _rj.running:
                st.info(f"♻️ 重剪进行中：**{_rj.processed} / {_rj.total}**"
                        f"（{_rj.progress_ratio * 100:.0f}%）· 正在处理 "
                        f"`{_rj.current or '…'}` · 预计剩余 {fmt_eta(_rj.eta_sec)}"
                        "　→ **实时进度见「执行进度」页**")
    elif _n_re_all:
        st.caption(f"另有 **{_n_re_all}** 条版本落后、**产物正常**的行，暂不重剪 —— "
                   "它们是取舍掉的那一批（收益集中在上面那批症状行），"
                   "不是「重剪也白剪」。要全量走 `jobs.start_rehighlight(only_dropped=False)`。")

    # —— 回收目录三堆：手动删除时唯一要看的判据是「这一条能不能删」——
    # 为什么不按"已剪辑 / 未剪辑"两堆分：need-del 里真正的风险只有一种 ——
    # 精华还没落袋、却被删了。而"没剪出 A/B"的那一大堆里有六成其实可以闭眼删，
    # 把危险的那两千多条淹在里面，等于没有保护。所以按保全状态分三堆。
    _rec = mover.recycle_stats(db, cfg)
    _n_clip = _rec["in_trash"][mover.STATE_CLIPPED]
    _n_hold = _rec["in_trash"][mover.STATE_PENDING]
    _n_plain = _rec["in_trash"][mover.STATE_PLAIN]
    _n_fix = sum(_rec["to_align"].values())
    _dir_clip = Path(cfg.get("trash_root") or "") / mover.CLIPPED_SUBDIR
    _dir_hold = Path(cfg.get("trash_root") or "") / mover.PENDING_SUBDIR
    if _rec["total"]:
        with st.container(border=True):
            st.markdown("##### 🗂 回收目录三堆（按「能不能删」分）")
            st.caption(
                f"回收目录里共 **{_rec['total']}** 条原片，按保全状态落成三堆。"
                "打开回收目录第一层就是这三堆，不用逐条回溯「这个我验过没有」。\n\n"
                f"| 子档 | 条数 | 怎么处置 |\n|---|---:|---|\n"
                f"| `{mover.CLIPPED_SUBDIR}` | {_n_clip} | 精华已落袋，**验完产物即可删** |\n"
                f"| `{mover.PENDING_SUBDIR}` | {_n_hold} | ⚠️ **精华还没落袋，一个都别删** |\n"
                f"| （留在档位目录） | {_n_plain} | 机器剪过、无 A/B 产物，**可直接删** |\n\n"
                "**看产物**：产物名以原片名开头（`<原片名>_E01_A_时刻_秒数.mp4`），"
                "复制原片名到精华目录里搜就能配上。")
            _pr = _rec.get("pending_reasons") or {}
            _r_never = _pr.get(mover.PEND_NEVER, 0)
            _r_ver = _pr.get(mover.PEND_VERSION, 0)
            _r_err = _pr.get(mover.PEND_ERROR, 0)
            if _n_hold:
                # 三种成因的下一步完全不同，不能再写一句"先跑剪辑 + 重剪"的
                # 万能话 —— 对磁盘满那批（N 条 error）那是**空动作**，它们两条
                # 队列都进不去，开跑也剪不到。
                _fix = []
                if _r_never:
                    _fix.append(
                        f"**{_r_never} 条从没剪过**（在「画面精华剪辑」队列里排队）"
                        + ("—— 磁盘空间够就会自动开跑" if not _hl_no_room
                           else f" —— ⚠️ 但磁盘闸门正在拦着（{_hl_no_room}），"
                                "先腾出空间再回来启动"))
                if _r_ver:
                    _fix.append(
                        f"**{_r_ver} 条版本落后**（旧口径留下的漏检）"
                        "→ 跑「按新口径重剪」")
                if _r_err:
                    _fix.append(
                        f"**{_r_err} 条剪到一半炸了**（多半是磁盘满）"
                        "→ 点下面「↩ 退回 N 条零产物失败记录」再重跑；"
                        "**光刷新页面没用**，它们的时间戳已写入，两条队列都不再收它们")
                st.warning(
                    f"⚠️ `{mover.PENDING_SUBDIR}` 里还有 **{_n_hold}** 条没剪完 —— "
                    "它们的精华此刻还在原片里，一个都别删。\n\n- "
                    + "\n- ".join(_fix)
                    + "\n\n处理完再点「📦 对齐子档」，这一堆会自然清空。"
                    "在它归零之前，不要清空回收目录。")
            if _n_fix:
                st.info(
                    f"📦 另有 **{_n_fix}** 条**已经处理完了、只是还待在错误的子档里**"
                    "（精华已落袋 / 已确认无产物）。这些不用再剪，"
                    "点「📦 对齐子档」搬到位即可。"
                    + ("（其中待保全为 0 —— 说明这一堆是纯归位问题。）"
                       if _n_hold == 0 else ""))
            _c1, _c2, _c3 = st.columns(3)
            if _c1.button(f"📦 对齐子档（{_n_fix} 条待归位）",
                          use_container_width=True, key="align_subdirs",
                          disabled=busy or _n_fix == 0,
                          help="把每条原片挪到它当前保全状态对应的子档；同盘改名，不复制数据"):
                with st.spinner("正在同盘搬迁（只改目录，不复制数据）…"):
                    _ar = mover.align_subdirs(db, cfg)
                if _ar["failed"]:
                    st.warning(f"搬入 {_ar['moved']} 条，失败 {_ar['failed']} 条："
                               + "；".join(_ar["errors"][:3]))
                else:
                    _bs = _ar["by_state"]
                    st.success(
                        f"已归位 {_ar['moved']} 条（跳过 {_ar['skipped']} 条）"
                        f" —— {mover.CLIPPED_SUBDIR} {_bs[mover.STATE_CLIPPED]} / "
                        f"{mover.PENDING_SUBDIR} {_bs[mover.STATE_PENDING]} / "
                        f"无产物 {_bs[mover.STATE_PLAIN]}")
                st.rerun()
            for _col, _tag, _d in ((_c2, mover.CLIPPED_SUBDIR, _dir_clip),
                                   (_c3, mover.PENDING_SUBDIR, _dir_hold)):
                if _col.button(f"📂 打开「{_tag}」", use_container_width=True,
                               key=f"open_subdir_{_tag}"):
                    if _d.exists():
                        reveal_in_explorer(str(_d))
                    else:
                        st.warning(f"目录还不存在：{_d}（先点左边的「对齐子档」）")

            # 「零产物失败记录退回」放在这儿而不是剪辑段：它的**唯一目的**是让
            # 这些原片重新变成「可剪」从而让待保全那一堆归零 —— 也就是说它是
            # 回收目录的补救动作，不是剪辑参数的一部分。原先挂在剪辑段里，
            # 找它的人从"待保全为什么不清零"查起，翻到剪辑段才知道要往回收
            # 目录这一张卡片走，反了。
            _n_retry = db.count_failed_retryable()
            if _n_retry:
                st.markdown("---")
                st.error(
                    f"⛔ **{_n_retry} 条零产物失败记录卡在队列外**（多半是磁盘满或任务中断）。"
                    "它们的时间戳与版本都已写入，两条队列都不会再收走 ⇒ "
                    "**不退回就永远剪不到，`待保全` 也永远归不了零**。"
                    + (f"其中 {min(_r_err, _n_retry)} 条还在回收目录里，就是上面那个"
                       f" `{mover.PENDING_SUBDIR}`。" if _r_err else ""))
                if st.button(f"↩ 退回 {_n_retry} 条零产物失败记录（可重剪）",
                             key="reset_highlight_failed_retry"):
                    n = db.reset_highlight("failed_retry")
                    st.success(f"已退回 {n} 条，现在可以「启动画面精华剪辑」重跑。")
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

    # 自检钩子（scripts/selftest.py 与 smoke_ui.py 用）：设了环境变量
    # VIDEO_SEARCH_DB_OVERRIDE 就把库路径指到临时库。
    #
    # 为什么需要：AppTest 跑的是整个 main()，而 main() 自己 load_config() +
    # get_db()，没法从 session_state 注入 —— 结果自检会去连**项目真实库**。
    # 真实库不存在（新机首次装 / 库被清 / CI）时整个自检在第一处就崩，
    # 后面上百项根本没跑。2026-10-04 真实库丢失时就是这样炸的。
    # 这个钩子让自检只依赖代码，不依赖任何既有数据。
    override = os.environ.get("VIDEO_SEARCH_DB_OVERRIDE")
    if override:
        cfg["runtime"]["sqlite_path"] = override

    # 配置损坏后由 config.load_config 自动恢复过，这里把情况告诉用户，
    # 否则用户只会看到"配置莫名其妙变了"而不知道发生过什么。
    notice = getattr(cfgmod, "LAST_NOTICE", "")
    if notice:
        st.warning(f"**配置文件异常已自动处理**：{notice}")

    db = get_db(cfg["runtime"]["sqlite_path"])
    render_sidebar(db, cfg)

    # 5 个页签（2026-10-04 从 6 个减来）：审核工作台与批量浏览已重做为
    # **抽查**（tab_review），旧「批量浏览」整页删除 ——
    # `st.tabs` 每次 rerun 会执行**所有** tab 的代码，那一页一次渲染
    # 24 缩略图 + 72 个按钮，删它是性能收益不是审美选择（见 ADR 0007）。
    t1, t2, t3, t4, t5 = st.tabs(
        ["抽查", "执行进度", "精华复核", "误判与改判", "设置与扫描"])
    with t1:
        tab_review(db, cfg)
    with t2:
        tab_progress(db, cfg)
    with t3:
        tab_highlight(db, cfg)
    with t4:
        tab_feedback(db, cfg)
    with t5:
        tab_settings(db, cfg)


if __name__ == "__main__":
    main()
