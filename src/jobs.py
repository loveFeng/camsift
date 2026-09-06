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
from typing import Any, Callable, Deque, Dict, Optional

_SCAN = "scan"
_ANALYZE = "analyze"
_AUDIO = "audio"

KIND_NAMES = {_SCAN: "扫描元数据", _ANALYZE: "AI 分析", _AUDIO: "音频复核"}


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
            try:
                target(job)
                job.status = "stopped" if job.stop_requested() else "done"
            except Exception as e:  # noqa: BLE001
                job.status = "error"
                job.error = f"{type(e).__name__}: {e}"
                job.log(f"任务异常：{job.error}")
            finally:
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


def _count_pending(db) -> int:
    with db.conn() as c:
        r = c.execute("SELECT COUNT(*) FROM videos WHERE status IN ('metadata','pending') "
                      "AND meta_ok=1").fetchone()
        return r[0] if r else 0


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


def get_scan_job() -> Job:
    return MANAGER.get(_SCAN)


def get_analyze_job() -> Job:
    return MANAGER.get(_ANALYZE)


def get_audio_job() -> Job:
    return MANAGER.get(_AUDIO)


def any_running() -> bool:
    return (get_scan_job().running or get_analyze_job().running
            or get_audio_job().running)
