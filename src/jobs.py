"""
后台任务管理 —— 让扫描/分析脱离 Streamlit 主线程

为什么需要它
------------
Streamlit 每次交互都会重跑整个脚本，如果在按钮回调里同步跑分析，
页面会卡死几十分钟，中途刷新还会把任务一起弄丢。

这里的做法：
  * 用普通线程跑任务，任务状态存在**进程级**单例里（不是 session_state）
    —— 这样刷新页面、切换页签都不会丢任务，进度照常显示。
  * 任务只碰数据库，**绝不碰 st.***（Streamlit 对象跨线程不安全）。
  * 支持 stop_event：置位后在当前文件处理完立即收尾，已完成的写入不丢。

任务状态：idle → running → done / error / stopped
"""
from __future__ import annotations

import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional

_SCAN = "scan"
_ANALYZE = "analyze"
_AUDIO = "audio"
_ARCHIVE = "archive"          # L2.7 语音档案归档（纯本地复制，分钟级）
_HIGHLIGHT = "highlight"      # L2.8 画面精华剪辑（抽帧+检测+重编码，最慢的一个）
_REHIGHLIGHT = "rehighlight"  # Track C · C5 按新口径重剪（删前保全）
_PIPELINE = "pipeline"        # ADR 0004 一键执行：把上面五步 + 批量放行串成一条链

from .archiver import check_ready as check_archive_ready

KIND_NAMES = {_SCAN: "扫描元数据", _ANALYZE: "AI 分析", _AUDIO: "音频复核",
              _ARCHIVE: "语音档案归档", _HIGHLIGHT: "画面精华剪辑",
              _REHIGHLIGHT: "画面精华重剪", _PIPELINE: "一键执行"}

# 一键执行的阶段与**固定顺序**（ADR 0004）。顺序由依赖决定，不能随意调换：
#   分析先于音频复核/剪辑 —— 两者的队列都只认已判定的 trash/review
#   音频复核先于语音归档 —— 归档的数据源是复核产出的转写与缓存 wav
#   放行排在剪辑之前     —— 剪辑的自动处置会搬走未处置候选，而放行队列只认未处置，
#                          排在后面就是空操作
PIPELINE_STAGES = (
    ("scan", "扫描元数据"),
    ("analyze", "AI 分析"),
    ("audio", "音频复核"),
    ("archive", "语音档案归档"),
    ("release", "批量放行"),
    ("highlight", "画面精华剪辑"),
)


class PipelineError(RuntimeError):
    """阶段级失败 —— 立即停整条链，不跑下游（ADR 0004）。"""


class Job:
    def __init__(self, kind: str):
        self.kind = kind
        self.status: str = "idle"          # idle | running | done | error | stopped
        self.processed: int = 0
        self.total: int = 0
        self.current: str = ""
        self.result: Dict[str, Any] = {}
        self.error: str = ""
        self.logs: Deque[str] = deque(maxlen=300)
        self.started_at: float = 0.0
        self.ended_at: float = 0.0
        self.batch_id: Optional[int] = None
        # 一键执行专用：当前处在第几个阶段（面板显示"阶段 3/6 · 音频复核"）。
        # 其它任务恒为 0/0，面板照旧只显示进度条。
        self.stage_index: int = 0
        self.stage_count: int = 0
        self.stage_name: str = ""
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------ 状态
    @property
    def running(self) -> bool:
        return self.status == "running" and self._thread is not None and self._thread.is_alive()

    @property
    def elapsed(self) -> float:
        if not self.started_at:
            return 0.0
        return (self.ended_at or time.time()) - self.started_at

    @property
    def speed(self) -> float:
        """个/秒。"""
        e = self.elapsed
        return self.processed / e if e > 0.05 else 0.0

    @property
    def eta_sec(self) -> float:
        """预计剩余秒数；无法估算时返回 -1。"""
        if not self.running or self.total <= 0 or self.processed <= 0:
            return -1.0
        s = self.speed
        if s <= 0:
            return -1.0
        return max(0.0, (self.total - self.processed) / s)

    @property
    def progress_ratio(self) -> float:
        if self.total <= 0:
            return 0.0
        return min(1.0, max(0.0, self.processed / self.total))

    def request_stop(self) -> None:
        self._stop.set()
        self.log("收到停止请求，正在当前文件结束后收尾...")

    def stop_requested(self) -> bool:
        return self._stop.is_set()

    # ------------------------------------------------------------------ 日志
    def log(self, msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        self.logs.append(f"[{ts}] {msg}")

    def tail_logs(self, n: int = 40) -> list[str]:
        return list(self.logs)[-n:]


class JobManager:
    """进程级单例 —— Streamlit 多会话共享同一份任务状态。"""

    def __init__(self):
        self._lock = threading.Lock()
        self._jobs: Dict[str, Job] = {}

    def get(self, kind: str) -> Job:
        with self._lock:
            if kind not in self._jobs:
                self._jobs[kind] = Job(kind)
            return self._jobs[kind]

    def start(self, kind: str, target: Callable[[Job], None]) -> tuple[bool, str]:
        job = self.get(kind)
        with self._lock:
            if job.running:
                return False, f"{KIND_NAMES.get(kind, kind)} 正在运行，请等待完成或先停止"
            job.__init__(kind)  # 重置状态，保留对象引用
        job.status = "running"
        job.started_at = time.time()
        job.log(f"启动 {KIND_NAMES.get(kind, kind)}")

        def runner():
            # 任务期间禁止系统休眠 —— 长任务（扫描/分析/剪辑动辄几十分钟）
            # 最怕跑一半机器睡了。ES_CONTINUOUS 挂在**本线程**上，
            # 下面的 finally 会清掉；线程结束系统也会自动回收。
            if _keep_awake(True):
                job.log("已申请「禁止系统休眠」，任务结束后自动释放")
            else:
                job.log("提示：未能申请禁止休眠（非 Windows 或调用被拦截），"
                        "请留意别让电脑睡过去")
            try:
                target(job)
                job.status = "stopped" if job.stop_requested() else "done"
            except Exception as e:  # noqa: BLE001
                job.status = "error"
                job.error = f"{type(e).__name__}: {e}"
                job.log(f"任务异常：{job.error}")
            finally:
                _keep_awake(False)          # 释放禁止休眠 + 干脆结束本线程的电源请求
                job.ended_at = time.time()
                job.current = ""
                job.log(f"任务结束（{job.status}），用时 {job.elapsed:.1f}s")

        t = threading.Thread(target=runner, name=f"job-{kind}", daemon=True)
        job._thread = t
        t.start()
        return True, "已启动"

    def stop(self, kind: str) -> None:
        self.get(kind).request_stop()


MANAGER = JobManager()


# ---------------------------------------------------------------------------- 任务体
def _keep_awake(enable: bool, note: str = "") -> bool:
    """任务运行期间申请「禁止系统休眠」；任务结束即释放。

    为什么用 SetThreadExecutionState 而不是 powercfg 改系统方案：
      * 它只在**当前任务运行期间**生效，任务一停（含异常退出）自动失效，
        不会把用户的电源计划永久改掉，也不需要在别处恢复；
      * 按线程记录 —— 在**工作线程自己**里申请、自己在 finally 里释放，最干净。
        （本函数由 runner 线程调用，不是主线程。）

    返回值仅供参考；失败（例如被策略拦截）不抛异常，只让任务照常跑。
    """
    try:
        import ctypes

        ES_CONTINUOUS = 0x80000000       # 状态持续有效，直到下次显式变更
        ES_SYSTEM_REQUIRED = 0x00000001  # 阻止系统自动睡眠/休眠
        # 注意：刻意**不加** ES_DISPLAY_REQUIRED —— 屏幕该黑就黑，省电，
        # 只是系统别睡。任务在跑就不需要人盯着屏幕。
        flags = ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if enable else 0x00000000)
        ctypes.windll.kernel32.SetThreadExecutionState(flags)
        return True
    except Exception:  # noqa: BLE001 — 非 Windows 或调用失败都只是"没防成"，不该炸任务
        return False


def _hook(job: Job):
    """把模块的 progress(done, total, name) 回调接到 Job 上。"""
    def on_progress(done: int, total: int, name: str) -> None:
        job.processed = done
        if total:
            job.total = max(job.total, total)
        job.current = name
    return on_progress


def _run_scan(job: Job, db, cfg: Dict, root: str, limit: Optional[int],
              with_audio: bool) -> None:
    from .scanner import Scanner
    sc = Scanner(db, cfg)
    job.total = limit or 0
    res = sc.scan(
        root, limit=limit, with_audio_energy=with_audio,
        progress=_hook(job), log=job.log, stop_event=job._stop,
    )
    job.batch_id = res.get("batch_id")
    job.result = res
    job.processed = res.get("added", 0)


def _run_analyze(job: Job, db, cfg: Dict, limit: Optional[int]) -> None:
    from .analyzer import Analyzer
    an = Analyzer(db, cfg)
    if not an.yolo_on:
        job.log("注意：未启用 YOLO（缺少模型文件），判定仅依赖画面动静")
    job.total = max(limit or 0, _count_pending(db))
    res = an.run(
        limit=limit, progress=_hook(job), log=job.log, stop_event=job._stop,
    )
    job.result = res
    job.processed = res.get("analyzed", 0) + res.get("errors", 0)


def _run_audio(job: Job, db, cfg: Dict, limit: Optional[int],
               batch_id: Optional[int], scope: str) -> None:
    """音频复核：对待清理/需人工确认候选（有音轨、未复核）做 VAD+ASR+LLM。

    scope: all = 两类都复核 / trash = 仅待清理 / review = 仅需人工确认
    """
    from .audioreview import AudioReviewer, AudioEngineError
    ar = AudioReviewer(cfg)
    problems = ar.check_ready()
    if problems:
        raise AudioEngineError("引擎未就绪：" + "；".join(problems))
    pending = db.count_audio_pending(batch_id, scope=scope)
    if limit:
        pending = min(pending, limit)
    job.total = pending
    res = ar.run_batch(
        db, limit=limit, progress=_hook(job), log=job.log,
        stop_event=job._stop, batch_id=batch_id, scope=scope,
    )
    job.result = res
    job.processed = res.get("checked", 0) + res.get("errors", 0)


def _run_archive(job: Job, db, cfg: Dict, limit: int = 5000) -> None:
    """语音档案归档：把已入回收目录的有效语音复制成 wav+txt 并追加 manifest。"""
    from .archiver import archive_pending, ArchiveError
    problems = check_archive_ready(cfg)
    if problems:
        raise ArchiveError("归档未就绪：" + "；".join(problems))
    pending = db.count_archive_pending()
    job.total = min(pending, limit)
    res = archive_pending(db, cfg, limit=limit, progress=_hook(job),
                          log=job.log, stop_event=job._stop)
    job.result = res
    job.processed = res.get("done", 0) + res.get("skipped", 0) + res.get("failed", 0)


def _run_highlight(job: Job, db, cfg: Dict, limit: int = 5000,
                   batch_id: Optional[int] = None) -> None:
    """画面精华剪辑：逐个视频抽帧 → 检测 → 切出有价值的几秒。

    这是整条流水线里最慢的一步（每段要抽近百帧做推理 + 重编码切段），
    所以必须放后台：跑起来之后该审核审核，随时可停，进度写库不丢。
    """
    from .highlighter import HighlightCutter, HighlightError
    cut = HighlightCutter(cfg)
    problems = check_highlight_ready(cfg)
    if problems:
        raise HighlightError("剪辑未就绪：" + "；".join(problems))
    pending = db.count_highlight_pending()
    job.total = min(pending, limit)
    res = cut.run_batch(
        db, limit=limit, progress=_hook(job), log=job.log,
        stop_event=job._stop, batch_id=batch_id,
    )
    job.result = res
    job.processed = res.get("done", 0) + res.get("empty", 0) + res.get("failed", 0)


def check_highlight_ready(cfg: Dict) -> list:
    """延迟导入用：避免 jobs 模块启动时就把 cv2 拖进来。"""
    from .highlighter import check_ready
    return check_ready(cfg)


def _run_rehighlight(job: Job, db, cfg: Dict, limit: int = 5000,
                     only_dropped: bool = True) -> None:
    """重剪：把已剪过、口径落后的视频按新口径重跑一遍（Track C · C5）。

    幂等靠 `highlight_version` —— 剪完版本号就跟上了，再跑队列是空的。
    中断安全：顺序跑、单条落库，停了下次从剩下的人开始。
    """
    from .highlighter import HighlightCutter, HighlightError
    problems = check_highlight_ready(cfg)
    if problems:
        raise HighlightError("剪辑未就绪：" + "；".join(problems))
    cut = HighlightCutter(cfg)
    rowst = db.rehighlight_queue(limit=limit, target_version=cut.hl_version,
                                 only_dropped=only_dropped)
    job.total = len(rowst)
    if not job.total:
        job.result = {"total": 0, "rescued": 0, "upgraded": 0, "rehighlight": True,
                      "note": "没有口径落后的产物，无需重剪"}
        return
    res = cut.run_batch(db, limit=limit, progress=_hook(job), log=job.log,
                        stop_event=job._stop, rehighlight=True,
                        only_dropped=only_dropped)
    job.result = res
    job.processed = res.get("done", 0) + res.get("empty", 0) + res.get("failed", 0)


def count_rehighlight_pending(db, cfg: Dict, only_dropped: bool = True) -> int:
    """口径落后的待重剪条数（UI 上按钮的显示依据）。"""
    from .highlighter import HL_VERSION
    tv = str((cfg.get("highlight") or {}).get("version") or HL_VERSION)
    return db.count_rehighlight_pending(target_version=tv, only_dropped=only_dropped)


def _count_pending(db) -> int:
    return db.count_pending_analysis()


# ---------------------------------------------------------------------- 一键执行
def _run_release(job: Job, db, cfg: Dict, limit: int) -> str:
    """批量放行：把「无人」候选整体移入回收目录（ADR 0003 的语义条件）。"""
    from .mover import move_all, release_params
    thr, need = release_params(cfg)
    trash_root = (cfg.get("trash_root") or "").strip()
    if not trash_root:
        raise PipelineError("未配置 trash_root，无法放行")

    rows = db.release_queue(limit=limit, threshold=thr, require_no_person=need)
    job.total = len(rows)
    if not rows:
        return "没有可放行的「无人」候选"
    job.current = f"放行 {len(rows)} 个「无人」候选"
    res = move_all(db, rows, cfg, log=job.log)
    job.processed = res["moved"] + res["failed"]
    if res["failed"] and not res["moved"]:
        raise PipelineError(f"放行全部失败：{res['errors'][:1]}")
    tail = f"（失败 {res['failed']}）" if res["failed"] else ""
    return f"放行 {res['moved']} 个{tail}"


def _run_stage(job: Job, db, cfg: Dict, key: str, opts: Dict) -> str:
    """执行单个阶段，返回一行人类可读的结果摘要。

    阶段体直接复用分步执行的 _run_* —— 口径完全一致，避免"一键跑的"和
    "手点按钮跑的"出现两套行为。stage_limit 只在真正会推进队列的阶段生效。
    """
    limit = int(opts.get("stage_limit") or 100000)
    if key == "scan":
        root = (cfg.get("source_root") or "").strip()
        if not root:
            raise PipelineError("未配置视频源目录 source_root")
        _run_scan(job, db, cfg, root, opts.get("scan_limit"),
                  bool(opts.get("scan_with_audio")))
        r = job.result
        return (f"新增 {r.get('added', 0)}，跳过 {r.get('skipped', 0)}，"
                f"失败 {r.get('failed', 0)}（批次 #{r.get('batch_id')}）")

    if key == "analyze":
        # 全库队列（ADR 0004）：限额只管"本次导入多少新料"，分析要连带清掉历史积压
        _run_analyze(job, db, cfg, None)
        r = job.result
        return f"成功 {r.get('analyzed', 0)}，失败 {r.get('errors', 0)}"

    if key == "audio":
        scope = str((cfg.get("audio", {}) or {}).get("review_scope") or "all")
        _run_audio(job, db, cfg, limit, None, scope)
        r = job.result
        if r.get("error"):            # enabled=False 时 run_batch 只返回 error 不抛异常
            raise PipelineError(str(r["error"]))
        return (f"复核 {r.get('checked', 0)} 段（检出人声 {r.get('speech', 0)}），"
                f"失败 {r.get('errors', 0)}")

    if key == "archive":
        _run_archive(job, db, cfg, limit=limit)
        r = job.result
        return (f"归档 {r.get('done', 0)} 段，跳过 {r.get('skipped', 0)}，"
                f"失败 {r.get('failed', 0)}")

    if key == "release":
        return _run_release(job, db, cfg, limit)

    if key == "highlight":
        _run_highlight(job, db, cfg, limit=limit)
        r = job.result
        d = r.get("disposed", {}) or {}
        kept = d.get("keep_full", 0) + d.get("undo_keep", 0)
        return (f"处理 {r.get('done', 0) + r.get('empty', 0)} 段，切出 {r.get('clips', 0)} 段，"
                f"失败 {r.get('failed', 0)}；处置：保留 {kept} / 移入回收 {d.get('moved', 0)} / "
                f"留待人工 {d.get('not_disposed', 0)}")

    raise PipelineError(f"未知阶段：{key}")


def _run_pipeline(job: Job, db, cfg: Dict, opts: Dict) -> None:
    """一键执行：按依赖顺序串行跑完各阶段（ADR 0004）。

    失败语义：单条失败（坏文件/解码失败）由各阶段自己记录后继续；**阶段级失败**
    （引擎未就绪/目录非法/抛异常）立即停下，不跑下游 —— 上游失败会让下游空跑或
    半成品，面板却显示"做完了"，比停下来更危险。

    停止 = 当前文件处理完收尾，不进入下一阶段；再点一次从断点续跑（各阶段的
    队列都有幂等标记：status / audio_checked / archived_at / highlight_at）。
    """
    stages = [(k, n) for k, n in PIPELINE_STAGES
              if (opts.get("stages") or {}).get(k, True)]
    job.stage_count = len(stages)
    job.stage_index = 0
    summary: List[Dict[str, Any]] = []

    for i, (key, name) in enumerate(stages, 1):
        if job.stop_requested():
            job.log("收到停止请求，不再进入下一阶段")
            break
        job.stage_index = i
        job.stage_name = name
        job.processed = 0
        job.total = 0
        job.current = ""
        job.log(f"——— 阶段 {i}/{len(stages)}：{name} ———")
        t0 = time.time()
        try:
            text = _run_stage(job, db, cfg, key, opts)
        except Exception as e:  # noqa: BLE001 阶段级失败 → 停整条链
            secs = round(time.time() - t0, 1)
            job.log(f"⛔ 阶段「{name}」失败：{type(e).__name__}: {e}（用时 {secs}s）")
            summary.append({"key": key, "name": name, "status": "error",
                            "text": f"{type(e).__name__}: {e}", "secs": secs})
            job.result = {"stages": summary, "stopped_at": name, "aborted": True}
            raise PipelineError(f"阶段「{name}」失败，已停后续阶段：{e}") from e
        secs = round(time.time() - t0, 1)
        summary.append({"key": key, "name": name, "status": "done",
                        "text": text, "secs": secs})
        job.log(f"✅ 「{name}」完成：{text}（{secs}s）")

    job.stage_name = ""
    job.current = ""
    job.result = {"stages": summary, "aborted": False,
                  "stopped_at": "" if not job.stop_requested() else "手动停止"}


def pipeline_precheck(cfg: Dict, opts: Dict) -> List[str]:
    """一键执行的**一次性**前置检查（ADR 0004）：缺一个就拒绝启动并列出缺什么。

    只查"会让某个已勾选阶段整体失败"的硬条件（目录非法 / 缺 ffmpeg / 缺模型 /
    ASR 未就绪）。单条视频解码失败之类不算 —— 那是单条失败，由阶段记录后继续。
    """
    problems: List[str] = []
    st = opts.get("stages") or {}

    if not (cfg.get("work_dir") or "").strip():
        problems.append("未配置本地工作目录 work_dir（数据库/缩略图要放 SSD）")

    if st.get("scan", True):
        src = (cfg.get("source_root") or "").strip()
        if not src:
            problems.append("未配置视频源目录 source_root —— 扫描阶段需要它")
        elif not Path(src).exists():
            problems.append(f"视频源目录不存在：{src}")

    if st.get("release", True) or st.get("highlight", True):
        if not (cfg.get("trash_root") or "").strip():
            problems.append("未配置分级回收目录 trash_root —— 放行/剪辑自动处置需要它")

    if st.get("audio", True):
        if not (cfg.get("audio", {}) or {}).get("enabled"):
            problems.append("音频复核未启用（config.audio.enabled=false）"
                            "—— 不跑这一步请把它的勾去掉")
        else:
            from .audioreview import AudioReviewer
            problems += [f"音频复核未就绪：{p}" for p in AudioReviewer(cfg).check_ready()]

    if st.get("archive", True):
        problems += [f"语音归档未就绪：{p}" for p in check_archive_ready(cfg)]

    if st.get("highlight", True):
        problems += [f"画面剪辑未就绪：{p}" for p in check_highlight_ready(cfg)]

    return problems


def start_pipeline(db, cfg, opts: Dict) -> tuple[bool, str]:
    """启动一键执行。前置检查不过或有任务在跑时拒绝启动。"""
    problems = pipeline_precheck(cfg, opts)
    if problems:
        return False, "前置检查未通过，已拒绝启动：\n" + "\n".join(f"· {p}" for p in problems)
    if any_running():
        return False, "已有后台任务在运行 —— 同时跑会互相抢队列，请等它结束或先停止"
    return MANAGER.start(_PIPELINE, lambda j: _run_pipeline(j, db, cfg, opts))


def get_pipeline_job() -> Job:
    return MANAGER.get(_PIPELINE)


def start_scan(db, cfg, root: str, limit: Optional[int] = None,
               with_audio: bool = False) -> tuple[bool, str]:
    return MANAGER.start(_SCAN, lambda j: _run_scan(j, db, cfg, root, limit, with_audio))


def start_analyze(db, cfg, limit: Optional[int] = None) -> tuple[bool, str]:
    return MANAGER.start(_ANALYZE, lambda j: _run_analyze(j, db, cfg, limit))


def start_audio_review(db, cfg, limit: Optional[int] = None,
                       batch_id: Optional[int] = None,
                       scope: str = "all") -> tuple[bool, str]:
    """启动音频复核后台任务。scope: all=待清理+需人工确认 / trash / review。

    引擎/依赖未就绪时返回错误信息。
    """
    return MANAGER.start(_AUDIO,
                         lambda j: _run_audio(j, db, cfg, limit, batch_id, scope))


def start_archive(db, cfg, limit: int = 5000) -> tuple[bool, str]:
    """启动语音档案归档后台任务。目录不合法时返回错误信息。"""
    return MANAGER.start(_ARCHIVE, lambda j: _run_archive(j, db, cfg, limit))


def get_scan_job() -> Job:
    return MANAGER.get(_SCAN)


def get_analyze_job() -> Job:
    return MANAGER.get(_ANALYZE)


def get_audio_job() -> Job:
    return MANAGER.get(_AUDIO)


def start_highlight(db, cfg, limit: int = 5000,
                    batch_id: Optional[int] = None) -> tuple[bool, str]:
    """启动画面精华剪辑后台任务。目录不合法时返回错误信息。"""
    return MANAGER.start(_HIGHLIGHT,
                         lambda j: _run_highlight(j, db, cfg, limit, batch_id))


def get_archive_job() -> Job:
    return MANAGER.get(_ARCHIVE)


def get_highlight_job() -> Job:
    return MANAGER.get(_HIGHLIGHT)


def start_rehighlight(db, cfg, limit: int = 5000,
                      only_dropped: bool = True) -> tuple[bool, str]:
    """启动画面精华重剪后台任务（Track C · C5）。"""
    return MANAGER.start(_REHIGHLIGHT,
                         lambda j: _run_rehighlight(j, db, cfg, limit, only_dropped))


def get_rehighlight_job() -> Job:
    return MANAGER.get(_REHIGHLIGHT)


def any_running() -> bool:
    return (get_scan_job().running or get_analyze_job().running
            or get_audio_job().running or get_archive_job().running
            or get_highlight_job().running or get_rehighlight_job().running
            or get_pipeline_job().running)
