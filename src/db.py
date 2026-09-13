"""
数据库层 —— SQLite

设计要点
--------
1. **保存未过滤的原始检出**（videos.raw_detections）
   这是「误判归因」能成立的前提。如果只存 has_person=0，事后就永远无法知道
   "模型其实看到了人、只是置信度 0.38 没过阈值"。误判分析全靠这份数据。

2. **判定解释链**（videos.decision_trace）
   记录打分的每一项贡献，让 UI 能回答"为什么判定要删/要留"。

3. **程序不删除**：本模块不提供任何删除数据的接口（videos 记录的物理删除由用户手动完成）。

线程模型：每个操作独立开关连接，避免 Streamlit 多线程下的连接复用问题。
"""
from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

# 画面精华产物的「现行口径」版本号。重剪队列按它挑版本落后的行；
# `mover.is_pending()`（「待保全」子档的判据）也读它 —— 两处必须同一个值，
# 否则队列说要重剪、回收目录却说"可删"，机主会照着删掉还没保全的原片。
# mover 直接 import 本常量，selftest 有断言钉住。
#
# ⚠️ 换口径要**同时**改三处：这里 / highlighter.HL_VERSION /
#    config.json 的 highlight.version（剪辑器实际写的是 config 那个值）。
#    三处漂移的后果是单向且不可见的：config 比这里新 ⇒ 新剪出来的行全部被
#    判成"版本落后 ⇒ 待保全"，回收目录永远不会归零；反过来 ⇒ 该重剪的不重剪。
#
# hlv1 旧 / hlv2 删前保全 / hlv3 检出必留（孤立命中不再被吞，见 highlighter 注释）
HIGHLIGHT_TARGET_VERSION = "hlv3"

SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    id               INTEGER PRIMARY KEY,
    path             TEXT    UNIQUE NOT NULL,
    filename         TEXT,
    rel_dir          TEXT,
    year             TEXT,
    size_bytes       INTEGER,
    mtime            REAL,

    -- ffprobe 元数据
    duration_sec     REAL,
    width            INTEGER,
    height           INTEGER,
    video_codec      TEXT,
    has_audio        INTEGER,
    audio_energy     REAL,
    bitrate_kbps     INTEGER,
    meta_ok          INTEGER DEFAULT 0,
    meta_error       TEXT,

    -- L1 分析
    status           TEXT DEFAULT 'pending',
    batch_id         INTEGER,
    sampled_frames   INTEGER,
    has_person       INTEGER,
    person_conf      REAL,
    has_vehicle      INTEGER,
    vehicle_conf     REAL,
    has_animal       INTEGER,
    motion_area_ratio REAL,
    raw_detections   TEXT,
    frame_stats      TEXT,
    retention_score  REAL,
    verdict          TEXT,
    decision_trace   TEXT,
    analyzed_at      TEXT,
    analyze_error    TEXT,

    -- L2 VLM 复核：列保留仅为兼容旧库（早期预留），当前无任何代码写入/读取。
    vlm_has_person   INTEGER,
    vlm_has_vehicle  INTEGER,
    vlm_confidence   REAL,
    vlm_evidence     TEXT,
    vlm_agree        INTEGER,
    vlm_model        TEXT,
    vlm_reviewed_at  TEXT,
    final_score      REAL,

    -- L2.5 音频复核（有声事件捞回，见 docs/设计档案-10TB筛选系统.md §3.6）
    audio_checked      INTEGER,   -- 0 未复核 / 1 已复核 / -1 无音轨跳过
    speech_present     INTEGER,   -- VAD 是否检出人声（0/1）
    speech_ratio       REAL,      -- 语音时长占比
    speech_segments    TEXT,      -- VAD 分段 JSON（起止时间戳）
    transcript_text    TEXT,      -- 转写全文（给人看）
    transcript_segments TEXT,     -- 段级时间戳 JSON
    llm_useful         INTEGER,   -- 1 有用 / 0 无用 / 2 不确定
    llm_reason         TEXT,      -- LLM 结论理由
    audio_error        TEXT,      -- 复核失败原因（技术失败默认保留）
    audio_checked_at   TEXT,

    -- 人工处置
    human_decision   TEXT,
    human_at         TEXT,
    trash_path       TEXT,
    trash_tier       TEXT
);

CREATE TABLE IF NOT EXISTS batches (
    id          INTEGER PRIMARY KEY,
    name        TEXT,
    root_dir    TEXT,
    created_at  TEXT,
    status      TEXT,
    total       INTEGER DEFAULT 0,
    scanned     INTEGER DEFAULT 0,
    analyzed    INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS action_log (
    id         INTEGER PRIMARY KEY,
    video_id   INTEGER,
    action     TEXT,
    from_path  TEXT,
    to_path    TEXT,
    batch_id   INTEGER,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS feedback (
    id              INTEGER PRIMARY KEY,
    video_id        INTEGER,
    ai_verdict      TEXT,
    human_verdict   TEXT,
    reason_category TEXT,
    reason_detail   TEXT,
    suggestion      TEXT,
    evidence        TEXT,
    created_at      TEXT
);

CREATE INDEX IF NOT EXISTS idx_videos_status  ON videos(status);
CREATE INDEX IF NOT EXISTS idx_videos_score   ON videos(retention_score);
CREATE INDEX IF NOT EXISTS idx_videos_final   ON videos(final_score);
CREATE INDEX IF NOT EXISTS idx_videos_verdict ON videos(verdict);
CREATE INDEX IF NOT EXISTS idx_videos_year    ON videos(year);
CREATE INDEX IF NOT EXISTS idx_videos_human   ON videos(human_decision);
CREATE INDEX IF NOT EXISTS idx_videos_batch   ON videos(batch_id);
CREATE INDEX IF NOT EXISTS idx_feedback_cat   ON feedback(reason_category);
"""

# 状态流转：pending -> metadata -> analyzed -> (reviewed/keep/trash)
STATUS_PENDING = "pending"
STATUS_METADATA = "metadata"
STATUS_ANALYZED = "analyzed"
STATUS_KEEP = "keep"
STATUS_TRASH = "trash"
STATUS_ERROR = "error"

# L2.5 音频复核列。老库在 init_db 时用 PRAGMA 检查后 ALTER TABLE 补齐
# （SQLite 的 ALTER TABLE 不支持 ADD COLUMN IF NOT EXISTS，只能手动补）。
AUDIO_COLUMNS: Dict[str, str] = {
    "audio_checked": "INTEGER",
    "speech_present": "INTEGER",
    "speech_ratio": "REAL",
    "speech_segments": "TEXT",
    "transcript_text": "TEXT",
    "transcript_segments": "TEXT",
    "llm_useful": "INTEGER",
    "llm_reason": "TEXT",
    "audio_error": "TEXT",
    "audio_checked_at": "TEXT",
}

# L2.7 语音档案列：归档时间戳（幂等标记，配合 manifest.jsonl 双保险）
ARCHIVE_COLUMNS: Dict[str, str] = {
    "archived_at": "TEXT",
}

# L2.8 画面精华列：剪辑时间戳（幂等标记）+ 产物描述 + 失败原因
# highlight_clips 存 JSON 数组（每段起止/分级/文件路径），供 UI 直接回看。
#
# highlight_sent / highlight_coarse_hits 是「无精华」拆三义的尺子（Track 0）：
#   没有这两个数，"无精华 3780" 里到底有多少是"真复核过没有"、多少是"压根没跑检测"
#   根本分不出来 —— 而后者曾把 474 条"粗扫明明看到过目标"的视频静悄悄推进了回收目录。
# highlight_version 记录产物口径（hlv1/hlv2/hlv3），驱动重剪队列（Track C · C5）。
# 它就是本项目的**唯一纠偏机制**：产物口径一变、或发现某个版本曾静默吃掉产物，
# 就 bump 版本号 —— 已剪过的行版本落后即自动回到重剪队列，且剪完即幂等。
HIGHLIGHT_COLUMNS: Dict[str, str] = {
    "highlight_at": "TEXT",
    "highlight_count": "INTEGER",
    "highlight_sec": "REAL",
    "highlight_clips": "TEXT",
    "highlight_error": "TEXT",
    "highlight_sent": "INTEGER",          # 粗扫**实际送检**帧数；0 = 帧差预筛整片跳过
    "highlight_coarse_hits": "INTEGER",   # 粗扫检出命中数，>0 而产物为空 = 信号被丢弃
    "highlight_version": "TEXT",          # 产物口径版本（hlv1 旧 / hlv2 / hlv3）
    "highlight_best_grade": "TEXT",       # 产物里的最高级别（A>B>C）；仅 C = 纯档案
}


def _scope_sql(scope: str | None) -> str:
    """复核范围 → verdict 过滤 SQL 片段。默认 all（待清理 + 需人工确认）。"""
    return {"trash": "= 'trash'", "review": "= 'review'"} \
        .get(scope or "all", "IN ('trash','review')")


class Database:
    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    @contextmanager
    def conn(self):
        c = sqlite3.connect(self.db_path, timeout=30.0)
        c.row_factory = sqlite3.Row
        try:
            yield c
            c.commit()
        finally:
            c.close()

    def init_db(self) -> None:
        with self.conn() as c:
            c.executescript(SCHEMA)
            self._migrate(c)

    def _migrate(self, c) -> None:
        """老库补齐新增列（音频复核 / 归档字段）。已存在则跳过，幂等安全。"""
        existing = {r[1] for r in c.execute("PRAGMA table_info(videos)")}
        for col, decl in {**AUDIO_COLUMNS, **ARCHIVE_COLUMNS,
                          **HIGHLIGHT_COLUMNS}.items():
            if col not in existing:
                c.execute(f"ALTER TABLE videos ADD COLUMN {col} {decl}")

    # ------------------------------------------------------------ 写入
    def upsert_video_meta(self, path: str, meta: Dict[str, Any]) -> int:
        """插入或更新一条视频的元数据。返回 video id。"""
        with self.conn() as c:
            row = c.execute("SELECT id FROM videos WHERE path=?", (path,)).fetchone()
            if row:
                c.execute(
                    """UPDATE videos SET filename=?, rel_dir=?, year=?, size_bytes=?, mtime=?,
                       duration_sec=?, width=?, height=?, video_codec=?, has_audio=?,
                       audio_energy=?, bitrate_kbps=?, meta_ok=?, meta_error=?,
                       status=CASE WHEN status='pending' THEN 'metadata' ELSE status END
                       WHERE path=?""",
                    (
                        meta.get("filename"), meta.get("rel_dir"), meta.get("year"),
                        meta.get("size_bytes"), meta.get("mtime"),
                        meta.get("duration_sec"), meta.get("width"), meta.get("height"),
                        meta.get("video_codec"), meta.get("has_audio"),
                        meta.get("audio_energy"), meta.get("bitrate_kbps"),
                        meta.get("meta_ok", 0), meta.get("meta_error"),
                        path,
                    ),
                )
                return row["id"]
            c.execute(
                """INSERT INTO videos
                   (path, filename, rel_dir, year, size_bytes, mtime,
                    duration_sec, width, height, video_codec, has_audio,
                    audio_energy, bitrate_kbps, meta_ok, meta_error, status)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    path, meta.get("filename"), meta.get("rel_dir"), meta.get("year"),
                    meta.get("size_bytes"), meta.get("mtime"),
                    meta.get("duration_sec"), meta.get("width"), meta.get("height"),
                    meta.get("video_codec"), meta.get("has_audio"),
                    meta.get("audio_energy"), meta.get("bitrate_kbps"),
                    meta.get("meta_ok", 0), meta.get("meta_error"),
                    STATUS_METADATA if meta.get("meta_ok") else STATUS_ERROR,
                ),
            )
            return c.execute("SELECT last_insert_rowid() AS id").fetchone()["id"]

    def update_analysis(self, video_id: int, result: Dict[str, Any]) -> None:
        with self.conn() as c:
            c.execute(
                """UPDATE videos SET
                     status=?, sampled_frames=?, has_person=?, person_conf=?,
                     has_vehicle=?, vehicle_conf=?, has_animal=?, motion_area_ratio=?,
                     raw_detections=?, frame_stats=?, retention_score=?, verdict=?,
                     decision_trace=?, analyzed_at=?, analyze_error=?,
                     final_score=?
                   WHERE id=?""",
                (
                    result.get("status", STATUS_ANALYZED),
                    result.get("sampled_frames"),
                    result.get("has_person"), result.get("person_conf"),
                    result.get("has_vehicle"), result.get("vehicle_conf"),
                    result.get("has_animal"), result.get("motion_area_ratio"),
                    json.dumps(result.get("raw_detections", []), ensure_ascii=False),
                    json.dumps(result.get("frame_stats", {}), ensure_ascii=False),
                    result.get("retention_score"), result.get("verdict"),
                    json.dumps(result.get("decision_trace", {}), ensure_ascii=False),
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                    result.get("analyze_error"),
                    result.get("final_score", result.get("retention_score")),
                    video_id,
                ),
            )

    def update_audio_review(self, video_id: int, r: Dict[str, Any]) -> None:
        """写入 L2.5 音频复核结果。r 内 None 字段不清空（只覆盖本次有值的）。"""
        sets = []
        args: List[Any] = []
        for col in ("audio_checked", "speech_present", "speech_ratio",
                    "speech_segments", "transcript_text", "transcript_segments",
                    "llm_useful", "llm_reason", "audio_error"):
            if col in r and r[col] is not None:
                sets.append(f"{col}=?")
                args.append(r[col])
        sets.append("audio_checked_at=?")
        args.append(time.strftime("%Y-%m-%d %H:%M:%S"))
        args.append(video_id)
        with self.conn() as c:
            c.execute(f"UPDATE videos SET {', '.join(sets)} WHERE id=?", args)

    def mark_audio_skipped(self, video_id: int, reason: str = "no_audio") -> None:
        """无音轨/文件缺失等不可复核的片段：直接标记跳过，不阻塞处置。"""
        with self.conn() as c:
            c.execute(
                "UPDATE videos SET audio_checked=-1, audio_error=?, audio_checked_at=? "
                "WHERE id=?",
                (reason, time.strftime("%Y-%m-%d %H:%M:%S"), video_id),
            )

    def set_human_decision(
        self, video_id: int, decision: str,
        trash_path: str | None = None, trash_tier: str | None = None,
    ) -> None:
        """记录人工处置。decision: keep | trash

        直写、**不继承旧值**：曾用 COALESCE(?, trash_path) 导致「撤销移动 → 改判
        keep」后 trash_path 仍残留，行同时带 keep 与失效回收路径，后续处置翻转时
        清单/空间统计会错报该文件在回收目录。现在维护硬不变量：
        human_decision='keep' ⇒ trash_path/trash_tier 恒为 NULL；
        human_decision='trash' 且未给路径 ⇒ 仅逻辑标记（文件仍在原位）。
        """
        with self.conn() as c:
            c.execute(
                """UPDATE videos SET human_decision=?, human_at=?,
                   status=?, trash_path=?, trash_tier=?
                   WHERE id=?""",
                (
                    decision, time.strftime("%Y-%m-%d %H:%M:%S"),
                    STATUS_KEEP if decision == "keep" else STATUS_TRASH,
                    trash_path, trash_tier, video_id,
                ),
            )

    def log_action(self, video_id: int, action: str,
                   from_path: str, to_path: str, batch_id: int | None = None) -> None:
        with self.conn() as c:
            c.execute(
                """INSERT INTO action_log (video_id, action, from_path, to_path, batch_id, created_at)
                   VALUES (?,?,?,?,?,?)""",
                (video_id, action, from_path, to_path, batch_id,
                 time.strftime("%Y-%m-%d %H:%M:%S")),
            )

    def add_feedback(self, video_id: int, ai_verdict: str, human_verdict: str,
                     reason_category: str, reason_detail: str,
                     suggestion: str, evidence: dict | None = None) -> None:
        with self.conn() as c:
            c.execute(
                """INSERT INTO feedback
                   (video_id, ai_verdict, human_verdict, reason_category,
                    reason_detail, suggestion, evidence, created_at)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (video_id, ai_verdict, human_verdict, reason_category,
                 reason_detail, suggestion,
                 json.dumps(evidence or {}, ensure_ascii=False),
                 time.strftime("%Y-%m-%d %H:%M:%S")),
            )

    # ------------------------------------------------------------ 批次
    def create_batch(self, name: str, root_dir: str) -> int:
        with self.conn() as c:
            c.execute(
                """INSERT INTO batches (name, root_dir, created_at, status)
                   VALUES (?,?,?,?)""",
                (name, root_dir, time.strftime("%Y-%m-%d %H:%M:%S"), "scanning"),
            )
            return c.execute("SELECT last_insert_rowid() AS id").fetchone()["id"]

    def update_batch(self, batch_id: int, **kw) -> None:
        if not kw:
            return
        cols = ", ".join(f"{k}=?" for k in kw)
        with self.conn() as c:
            c.execute(f"UPDATE batches SET {cols} WHERE id=?", (*kw.values(), batch_id))

    def refresh_batch_progress(self, batch_id: int | None = None) -> None:
        """重算批次的 total / scanned / analyzed —— 进度条的唯一数据来源。

        为什么用「重算」而不是「累加」：分析可分多次、跨进程（CLI 与 UI 会交替跑），
        累加会重复计数，重算幂等且永远与 videos 表一致。
        analyzed 计入 analyzed/keep/trash 三种状态 —— 人工处置过的片段同样算已分析。
        """
        with self.conn() as c:
            if batch_id:
                ids = [batch_id]
            else:
                ids = [r[0] for r in c.execute("SELECT id FROM batches").fetchall()]
            for bid in ids:
                c.execute(
                    """UPDATE batches SET
                         total    = (SELECT COUNT(*) FROM videos WHERE batch_id=?),
                         scanned  = (SELECT COUNT(*) FROM videos WHERE batch_id=? AND meta_ok=1),
                         analyzed = (SELECT COUNT(*) FROM videos WHERE batch_id=?
                                     AND status IN ('analyzed','keep','trash'))
                       WHERE id=?""",
                    (bid, bid, bid, bid))

    def batch_progress(self, batch_id: int) -> Dict[str, int]:
        """批次实时进度（含「剩余多少」），供 UI 展示。"""
        with self.conn() as c:
            def one(q, *a):
                r = c.execute(q, a).fetchone()
                return r[0] if r else 0
            return {
                "total": one("SELECT COUNT(*) FROM videos WHERE batch_id=?", batch_id),
                "scanned": one("SELECT COUNT(*) FROM videos WHERE batch_id=? AND meta_ok=1", batch_id),
                "analyzed": one(
                    "SELECT COUNT(*) FROM videos WHERE batch_id=? "
                    "AND status IN ('analyzed','keep','trash')", batch_id),
                "pending": one(
                    "SELECT COUNT(*) FROM videos WHERE batch_id=? AND meta_ok=1 "
                    "AND status IN ('metadata','pending')", batch_id),
                "errors": one("SELECT COUNT(*) FROM videos WHERE batch_id=? AND status='error'", batch_id),
            }

    def get_batch(self, batch_id: int) -> Optional[sqlite3.Row]:
        with self.conn() as c:
            return c.execute("SELECT * FROM batches WHERE id=?", (batch_id,)).fetchone()

    def list_batches(self) -> List[sqlite3.Row]:
        with self.conn() as c:
            return c.execute("SELECT * FROM batches ORDER BY id DESC").fetchall()

    def set_video_batch(self, video_id: int, batch_id: int) -> None:
        """把单条视频绑定到批次（扫描过程中即时绑定，让进度条中途可见）。"""
        with self.conn() as c:
            c.execute("UPDATE videos SET batch_id=? WHERE id=?", (batch_id, video_id))

    def bind_batch(self, batch_id: int, year: str | None = None, root_dir: str | None = None) -> int:
        """把已扫描的视频绑定到批次。返回绑定数量。"""
        with self.conn() as c:
            if year:
                cur = c.execute("UPDATE videos SET batch_id=? WHERE year=? AND batch_id IS NULL",
                                (batch_id, year))
            elif root_dir:
                cur = c.execute(
                    "UPDATE videos SET batch_id=? WHERE (path LIKE ?) AND batch_id IS NULL",
                    (batch_id, root_dir.rstrip("\\/") + "%"))
            else:
                cur = c.execute("UPDATE videos SET batch_id=? WHERE batch_id IS NULL", (batch_id,))
            n = cur.rowcount
            c.execute("UPDATE batches SET total=(SELECT COUNT(*) FROM videos WHERE batch_id=?) WHERE id=?",
                      (batch_id, batch_id))
            return n

    # ------------------------------------------------------------ 查询
    def get_video(self, video_id: int) -> Optional[sqlite3.Row]:
        with self.conn() as c:
            return c.execute("SELECT * FROM videos WHERE id=?", (video_id,)).fetchone()

    def get_video_by_path(self, path: str) -> Optional[sqlite3.Row]:
        with self.conn() as c:
            return c.execute("SELECT * FROM videos WHERE path=?", (path,)).fetchone()

    def iter_pending_meta(self, batch_id: int | None = None, limit: int = 1000) -> List[sqlite3.Row]:
        sql = "SELECT * FROM videos WHERE meta_ok=0"
        args: List[Any] = []
        if batch_id:
            sql += " AND batch_id=?"
            args.append(batch_id)
        sql += " LIMIT ?"
        args.append(limit)
        with self.conn() as c:
            return c.execute(sql, args).fetchall()

    # ---------------------------------------------------------------- 重新分析
    # 改了判定阈值（YOLO 置信度、动静阈值…）后，已分析过的记录必须退回待分析才会重跑：
    # 分析队列只取 status IN ('metadata','pending')，analyzed/keep/trash 都不在里面。
    RESET_SCOPES = ("undecided", "failed", "all")

    def count_resettable(self, scope: str = "undecided",
                         batch_id: int | None = None) -> int:
        """可退回待分析的条数。scope: undecided / failed / all（见 reset_analysis）。"""
        sql, args = self._reset_where(scope, batch_id)
        with self.conn() as c:
            return c.execute("SELECT COUNT(*) FROM videos WHERE " + sql, args).fetchone()[0]

    def reset_analysis(self, scope: str = "undecided",
                       batch_id: int | None = None) -> int:
        """把已分析记录退回待分析状态（改参数后重跑用），返回重置条数。

        undecided  未人工处置的（analyzed/error）—— 默认，不动人工结论
        failed     仅分析失败的（status='error'）
        all        全部已分析，含已人工处置的 —— 会覆盖人工结论，UI 需二次确认

        只清 L1 分析产物；人工处置字段保留（那是人的决定，不因重分析而丢）。
        绝不触碰文件，也绝不删行。
        """
        if scope not in self.RESET_SCOPES:
            raise ValueError(f"未知 reset scope: {scope}")
        where, args = self._reset_where(scope, batch_id)
        with self.conn() as c:
            cur = c.execute(
                "UPDATE videos SET status=?, sampled_frames=0, has_person=NULL, "
                "person_conf=NULL, has_vehicle=NULL, vehicle_conf=NULL, "
                "has_animal=NULL, motion_area_ratio=NULL, raw_detections=NULL, "
                "frame_stats=NULL, retention_score=NULL, verdict=NULL, "
                "decision_trace=NULL, analyzed_at=NULL, analyze_error=NULL, "
                "final_score=NULL WHERE " + where,
                [STATUS_METADATA] + args)
            n = cur.rowcount
        if batch_id:
            self.refresh_batch_progress(batch_id)
        return n

    def _reset_where(self, scope: str, batch_id: int | None) -> tuple[str, list]:
        """reset/count 共用的 WHERE 片段（不含 WHERE 关键字）。"""
        base = "meta_ok=1 AND status NOT IN (?,?)"
        args: List[Any] = [STATUS_PENDING, STATUS_METADATA]
        if scope == "failed":
            base += " AND status=?"
            args.append(STATUS_ERROR)
        elif scope == "undecided":
            base += " AND human_decision IS NULL"
        # scope == "all"：不额外限制（含已人工处置的，由 UI 层把关确认）
        if batch_id:
            base += " AND batch_id=?"
            args.append(batch_id)
        return base, args

    def count_pending_analysis(self, batch_id: int | None = None) -> int:
        """待分析条数（与 iter_pending_analysis 同口径）。"""
        sql = ("SELECT COUNT(*) FROM videos WHERE status IN ('metadata','pending') "
               "AND meta_ok=1")
        args: List[Any] = []
        if batch_id:
            sql += " AND batch_id=?"
            args.append(batch_id)
        with self.conn() as c:
            return c.execute(sql, args).fetchone()[0]

    def iter_pending_analysis(self, batch_id: int | None = None, limit: int = 1000) -> List[sqlite3.Row]:
        sql = "SELECT * FROM videos WHERE status IN ('metadata','pending') AND meta_ok=1"
        args: List[Any] = []
        if batch_id:
            sql += " AND batch_id=?"
            args.append(batch_id)
        sql += " LIMIT ?"
        args.append(limit)
        with self.conn() as c:
            return c.execute(sql, args).fetchall()

    # 审核队列的谓词只有一处定义 —— 队列、计数、跳转定位三者共用，
    # 否则"显示 1000 条但翻到第 3 页就空"这类口径漂移会反复出现
    # （count_audio_pending 就踩过：计数与队列谓词不一致，报 0 实际 2322）。
    def _review_sql(self, only_unreviewed: bool = True, speech_only: bool = False,
                    person_only: bool = False, min_score: float | None = None,
                    max_score: float | None = None,
                    below_score: float | None = None,
                    batch_id: int | None = None,
                    hide_c_archive: bool = False) -> tuple[str, List[Any]]:
        """审核队列的 WHERE 片段（含前导 AND）与参数。"""
        sql = " AND human_decision IS NULL" if only_unreviewed else ""
        args: List[Any] = []
        if speech_only:
            sql += " AND audio_checked=1 AND speech_present=1"
        if person_only:
            sql += " AND has_person=1"
        if hide_c_archive:
            # Track C · C3：产物**只有 C 级**的视频是「档案」不是「待办」。
            # 它们已经剪完留底（Track C 生效后 C 级也出可播放 mp4），
            # 让人再逐条过一遍等于把"多留"变成"多看"，与减负相悖。
            # 只影响**列表展示**，不改放行/审核判定 —— 校验与处置逻辑一律不认这个开关。
            sql += " AND COALESCE(highlight_best_grade, '') <> 'C'"
        if min_score is not None:
            sql += " AND COALESCE(final_score, retention_score, 0) >= ?"
            args.append(float(min_score))
        if max_score is not None:
            sql += " AND COALESCE(final_score, retention_score, 0) <= ?"
            args.append(float(max_score))
        if below_score is not None:
            sql += " AND COALESCE(final_score, retention_score, 0) < ?"
            args.append(float(below_score))
        if batch_id:
            sql += " AND batch_id=?"
            args.append(batch_id)
        return sql, args

    def review_queue(self, batch_id: int | None = None, limit: int = 200,
                     only_unreviewed: bool = True, offset: int = 0,
                     speech_only: bool = False, person_only: bool = False,
                     min_score: float | None = None,
                     max_score: float | None = None,
                     hide_c_archive: bool = False) -> List[sqlite3.Row]:
        """审核队列：按最终分升序 —— 最可疑的排最前。offset 用于分页。

        speech_only=True 时只看「音频复核检出人声」的片段——画面被判删但画外
        可能有人声事件，清理前值得单独过一遍。

        person_only / min_score / max_score 是审核台的定位筛子：分数升序下，
        "跳到某个分数"等价于"数出有多少条低于它"，故 min/max 与跳转共用谓词。

        hide_c_archive=True 时滤掉「产物只有 C 级档案」的视频（Track C · C3）。
        """
        sql = ("SELECT * FROM videos WHERE verdict IN ('trash','review') AND meta_ok=1")
        frag, args = self._review_sql(only_unreviewed=only_unreviewed,
                                      speech_only=speech_only, person_only=person_only,
                                      min_score=min_score, max_score=max_score,
                                      batch_id=batch_id, hide_c_archive=hide_c_archive)
        sql += frag
        sql += " ORDER BY COALESCE(final_score, retention_score, 0) ASC, id ASC LIMIT ? OFFSET ?"
        args += [limit, int(offset or 0)]
        with self.conn() as c:
            return c.execute(sql, args).fetchall()

    def count_review_queue(self, batch_id: int | None = None,
                           only_unreviewed: bool = True,
                           speech_only: bool = False, person_only: bool = False,
                           min_score: float | None = None,
                           max_score: float | None = None,
                           hide_c_archive: bool = False) -> int:
        sql = ("SELECT COUNT(*) FROM videos WHERE verdict IN ('trash','review') AND meta_ok=1")
        frag, args = self._review_sql(only_unreviewed=only_unreviewed,
                                      speech_only=speech_only, person_only=person_only,
                                      min_score=min_score, max_score=max_score,
                                      batch_id=batch_id, hide_c_archive=hide_c_archive)
        sql += frag
        with self.conn() as c:
            return c.execute(sql, args).fetchone()[0]

    def count_review_below(self, score: float, batch_id: int | None = None,
                           only_unreviewed: bool = True,
                           speech_only: bool = False, person_only: bool = False,
                           hide_c_archive: bool = False) -> int:
        """队列里分数**低于** score 的条数 = 第一条「分数 ≥ score」的 0-based 位置。

        审核台按分数跳转靠它——别在 UI 里自己算，否则筛选条件一加就与队列错位。
        """
        sql = ("SELECT COUNT(*) FROM videos WHERE verdict IN ('trash','review') AND meta_ok=1")
        frag, args = self._review_sql(only_unreviewed=only_unreviewed,
                                      speech_only=speech_only, person_only=person_only,
                                      below_score=score, batch_id=batch_id,
                                      hide_c_archive=hide_c_archive)
        sql += frag
        with self.conn() as c:
            return c.execute(sql, args).fetchone()[0]

    def count_audio_pending(self, batch_id: int | None = None,
                            only_unreviewed: bool = True,
                            scope: str = "all") -> int:
        """待音频复核的**处置候选**数（有音轨 + 未复核）。

        scope: all = 待清理+需人工确认（默认） / trash = 仅待清理 / review = 仅需人工确认

        口径必须与 audio_review_queue **逐字一致**（同用 _pending_or_trashed_sql）：
        包括**已放行/已自动移入回收目录**但尚未复核的片段 —— 它们仍能补复核
        （review_one 有 trash_path 回落）。若这里写成"只看未处置"，会出现
        「队列有 100 条、计数是 0」的错报，任务进度条的分母也就跟着错。
        """
        verdict_sql = _scope_sql(scope)
        args: List[Any] = []
        if only_unreviewed:
            sql = (f"SELECT COUNT(*) FROM videos WHERE "
                   f"{self._pending_or_trashed_sql(verdict_sql)} "
                   "AND has_audio=1 AND audio_checked IS NULL")
        else:
            sql = (f"SELECT COUNT(*) FROM videos WHERE verdict {verdict_sql} "
                   "AND meta_ok=1 AND has_audio=1")
        if batch_id:
            sql += " AND batch_id=?"
            args.append(batch_id)
        with self.conn() as c:
            return c.execute(sql, args).fetchone()[0]

    def audio_review_queue(self, batch_id: int | None = None, limit: int = 1000,
                           offset: int = 0, scope: str = "all") -> List[sqlite3.Row]:
        """音频复核队列：候选按分数升序（最可疑/最该删的优先复核）。

        scope: all = 待清理+需人工确认（默认） / trash = 仅待清理 / review = 仅需人工确认。
        两类都覆盖 —— trash(待清理) 与 review(需人工确认) 均可经音频复核 + LLM 文字判断
        捞回画外有声事件。
        """
        verdict_sql = _scope_sql(scope)
        sql = (f"SELECT * FROM videos WHERE "
               f"{self._pending_or_trashed_sql(verdict_sql)} "
               "AND has_audio=1 AND audio_checked IS NULL")
        args: List[Any] = []
        if batch_id:
            sql += " AND batch_id=?"
            args.append(batch_id)
        sql += " ORDER BY COALESCE(final_score, retention_score, 0) ASC, id ASC LIMIT ? OFFSET ?"
        args += [limit, int(offset or 0)]
        with self.conn() as c:
            return c.execute(sql, args).fetchall()

    # ------------------------------------------------------------ 语音档案
    # 谓词 = 「先剪后删」处置候选（见 ADR 0001）：
    #   已移入回收目录（human_decision='trash'，人工/批量放行过）
    #   OR 未处置但 AI 判定待清理/需确认（human_decision IS NULL 且 verdict IN trash/review）
    # 放宽后 review 桶（0-30 分边界片段）也能归档 —— 画外音正是这类片段最容易藏内容的
    # 地方，且归档数据源是缓存 wav + 转写，不依赖视频本体，未移动也能归档。
    _PENDING_OR_TRASHED = (
        "(human_decision='trash' AND trash_path IS NOT NULL) "
        "OR (human_decision IS NULL AND verdict IN ('trash','review') AND meta_ok=1)"
    )

    def _pending_or_trashed_sql(self, verdict_sql: str | None = None) -> str:
        """处置候选谓词；给了 verdict_sql 时**只约束"未处置"那一半**。

        已处置（human_decision='trash'）的片段，verdict 仍然是 AI 当初的判定 ——
        放行不会改写它。若整条谓词都用 verdict 过滤，批量放行过的 review 档
        会从音频队列里漏掉（画外音再也补不上）。所以 scope 只作用于未处置半边。
        """
        if not verdict_sql:
            return f"({self._PENDING_OR_TRASHED})"
        return ("((human_decision='trash' AND trash_path IS NOT NULL) "
                f"OR (human_decision IS NULL AND verdict {verdict_sql} AND meta_ok=1))")

    def archive_queue(self, limit: int = 5000) -> List[sqlite3.Row]:
        """待归档队列：处置候选 + 确认为有效语音 + 尚未归档。

        谓词与 highlight_queue 共用 _PENDING_OR_TRASHED —— 归档与剪辑都面向同一
        批「即将被删除」的候选。已撤销移动（human_decision='keep'）自动退出队列。
        """
        # 谓词本身是 A OR B，必须整体加括号：否则 AND 优先级高于 OR，
        # speech_present/archived_at 过滤只作用于 B 分支，已移入回收的（A 分支）
        # 会绕过过滤 —— 曾导致已剪过的视频每次批处理都重剪。
        sql = (f"SELECT * FROM videos WHERE ({self._PENDING_OR_TRASHED}) "
               "AND speech_present=1 AND archived_at IS NULL "
               "ORDER BY id ASC LIMIT ?")
        with self.conn() as c:
            return c.execute(sql, (int(limit or 5000),)).fetchall()

    def count_archive_pending(self) -> int:
        """待归档条数（处置候选中的有效语音，未归档）。"""
        sql = (f"SELECT COUNT(*) FROM videos WHERE ({self._PENDING_OR_TRASHED}) "
               "AND speech_present=1 AND archived_at IS NULL")
        with self.conn() as c:
            return c.execute(sql).fetchone()[0]

    def count_archived(self) -> int:
        """已归档条数。"""
        with self.conn() as c:
            return c.execute(
                "SELECT COUNT(*) FROM videos WHERE archived_at IS NOT NULL").fetchone()[0]

    def mark_archived(self, video_id: int, ts: str) -> None:
        """标记已归档（幂等：重复标记无副作用，值被覆盖为最新时间）。"""
        with self.conn() as c:
            c.execute("UPDATE videos SET archived_at=? WHERE id=?", (ts, video_id))

    # ------------------------------------------------------------ 画面精华 L2.8
    # 队列谓词与 archive_queue 共用 _PENDING_OR_TRASHED：不只处理**已移入回收目录**
    # 的视频，也处理 review 桶（AI 判定需人工确认但尚未处置）—— 见 ADR 0001
    # 「先剪后删」：人工不再逐条审原片，机器高密度扫完直接产出精华，原片照删。
    # 未移入回收目录的视频直接从源路径（path）读，cut_one 有 trash_path→path 回落。
    def highlight_queue(self, limit: int = 5000,
                        batch_id: int | None = None,
                        cols: str = "*") -> List[sqlite3.Row]:
        """待剪辑队列：处置候选 + 未剪辑 + 无失败标记。

        highlight_error 非空的不进队列 —— 缺文件/解不开这类硬失败重试也没用，
        想重来请用 reset_highlight() 显式清除。

        ⚠️ 被这条谓词排除 ≠ 「不需要保全」。硬失败行和它一样是没落袋的，
        所以 mover.is_pending() 会把这两类**并起来**看待（见 pending_highlight_ids）。

        cols 只给外部交叉核对用（如 pending_highlight_ids 只要 id）：
        默认 * 会把 raw_detections / 转写这些大 JSON 一起拉出来。
        """
        sql = (f"SELECT {cols} FROM videos WHERE ({self._PENDING_OR_TRASHED}) "
               "AND highlight_at IS NULL AND highlight_error IS NULL")
        args: List[Any] = []
        if batch_id:
            sql += " AND batch_id=?"
            args.append(batch_id)
        sql += " ORDER BY id ASC LIMIT ?"
        args.append(int(limit or 5000))
        with self.conn() as c:
            return c.execute(sql, args).fetchall()

    def count_highlight_pending(self) -> int:
        with self.conn() as c:
            return c.execute(
                f"SELECT COUNT(*) FROM videos WHERE ({self._PENDING_OR_TRASHED}) "
                "AND highlight_at IS NULL AND highlight_error IS NULL").fetchone()[0]

    # ------------------------------------------------------------ 审核减负 · 放行候选
    # 放行候选（ADR 0003）= 判删候选里「机器没看到任何人物信号」的片段。
    # ADR 0002 用分数上界（≤10）间接表达这件事；一旦要把"无人、但被动物/音频
    # 叠加抬到 10 分以上"的片段也放行，就必须直接判定有没有人物信号。
    def release_queue(self, limit: int = 5000, batch_id: int | None = None,
                      threshold: float = 30.0,
                      require_no_person: bool = True,
                      cols: str = "*") -> List[sqlite3.Row]:
        """放行候选：未处置的判删候选，分数不高于 threshold 且无人物信号。

        meta_ok=1 保证分析确实跑过（不是没分析被当成 0 分）。
        require_no_person=True 时要求 has_person=0 **且** decision_trace 里没有
        「疑似人物未达阈值」—— 「模型看到了但没敢确认」正是漏检高发区，必须留给
        人或交给精华剪辑精扫复核，不能当"无人"批量放行。
        动物/车辆不构成排除条件：动物权重只有人物的三分之一，实测 0 辆。
        threshold=0 且 require_no_person=False 即回到 ADR 0002 之前的"仅 0 分桶"。

        cols 只给 count 用：near_miss 得在 Python 里读 decision_trace 判断，
        数数时没必要把 raw_detections / 转写这些大 JSON 一起拉出来（1.4w 行的库
        每次渲染审核台都会调一次 count_release）。
        """
        sql = (f"SELECT {cols} FROM videos WHERE verdict IN ('trash','review') "
               "AND meta_ok=1 AND human_decision IS NULL "
               "AND COALESCE(final_score, retention_score, 0) <= ?")
        args: List[Any] = [float(threshold)]
        if require_no_person:
            sql += " AND has_person=0"
        if batch_id:
            sql += " AND batch_id=?"
            args.append(batch_id)
        sql += " ORDER BY id ASC LIMIT ?"
        args.append(int(limit or 5000))
        with self.conn() as c:
            rows = c.execute(sql, args).fetchall()
        if not require_no_person:
            return rows
        out = []
        for r in rows:
            try:
                trace = json.loads(r["decision_trace"] or "{}")
            except (TypeError, ValueError):
                trace = {}
            if (trace.get("near_miss") or {}).get("person"):
                continue
            out.append(r)
        return out

    def count_release(self, batch_id: int | None = None,
                      threshold: float = 30.0,
                      require_no_person: bool = True) -> int:
        """放行候选条数。"""
        return len(self.release_queue(limit=1_000_000, batch_id=batch_id,
                                      threshold=threshold,
                                      require_no_person=require_no_person,
                                      cols="id, decision_trace"))

    def count_highlight_done(self) -> int:
        with self.conn() as c:
            return c.execute(
                "SELECT COUNT(*) FROM videos WHERE highlight_at IS NOT NULL").fetchone()[0]

    def highlight_stats(self) -> Dict[str, Any]:
        """精华剪辑汇总：已处理数 / 切出片段数 / 精华总时长 / 失败数。"""
        with self.conn() as c:
            r = c.execute(
                "SELECT COUNT(*) n, COALESCE(SUM(highlight_count),0) clips, "
                "COALESCE(SUM(highlight_sec),0) secs FROM videos "
                "WHERE highlight_at IS NOT NULL").fetchone()
            e = c.execute(
                "SELECT COUNT(*) FROM videos WHERE highlight_error IS NOT NULL").fetchone()[0]
        return {"done": r["n"], "clips": r["clips"], "secs": r["secs"], "failed": e}

    def mark_highlight(self, video_id: int, ts: str, count: int,
                       sec: float, clips: List[Dict],
                       sent: Optional[int] = None,
                       coarse_hits: Optional[int] = None,
                       version: Optional[str] = None) -> None:
        """写入剪辑结果。count=0 表示"确认无精华可留"，同样算处理完成。

        sent / coarse_hits / version 是 Track 0 的观测口径（见 HIGHLIGHT_COLUMNS）。
        传 None 时不清空旧值（老调用点不传就保持原样），避免半截写入把尺子抹掉。
        """
        sets = ["highlight_at=?", "highlight_count=?", "highlight_sec=?",
                "highlight_clips=?", "highlight_error=NULL"]
        args: List[Any] = [ts, int(count), float(sec),
                           json.dumps(clips, ensure_ascii=False)]
        # 最高级别由 clips 现算，调用方不用管；A>B>C，"仅 C" 即纯档案
        best = min((str(c.get("grade")) for c in clips if c.get("grade")),
                   default=None)
        sets.append("highlight_best_grade=?")
        args.append(best)
        for col, val in (("highlight_sent", sent),
                         ("highlight_coarse_hits", coarse_hits),
                         ("highlight_version", version)):
            if val is not None:
                sets.append(f"{col}=?")
                args.append(val)
        args.append(video_id)
        with self.conn() as c:
            c.execute(f"UPDATE videos SET {', '.join(sets)} WHERE id=?", args)

    def no_clip_breakdown(self) -> Dict[str, int]:
        """「无精华」拆三义（Track 0）——它是判断 Track C 有没有效的唯一尺子。

            已检测·确实没有 | 粗扫跑过检测、一帧命中都没有
            预筛跳过        | 帧差预筛判定画面没变，YOLO 一帧没跑（已接受的口径）
            看到过但被否决   | 粗扫有命中，最终却零产物 ← D2 的受害者，Track C 要救的就是它

        三者之和 == 无精华总数（selftest 有断言，防口径再次漂移）。
        注意三个分义必须**互斥**：`highlight_sent IS NULL`（回填前的历史行）只能落
        在 unknown 里，不能因为 COALESCE(NULL,0)=0 就同时被算成"预筛跳过" ——
        这正是 selftest 抓出来的一个重复计数 bug。
        """
        base = ("highlight_at IS NOT NULL AND highlight_error IS NULL "
                "AND (highlight_clips IS NULL OR highlight_clips='[]')")
        with self.conn() as c:
            r = c.execute(
                f"""SELECT
                      COALESCE(SUM(CASE WHEN highlight_sent = 0
                                        THEN 1 ELSE 0 END),0) AS skipped,
                      COALESCE(SUM(CASE WHEN highlight_sent > 0
                                         AND COALESCE(highlight_coarse_hits,0) = 0
                                        THEN 1 ELSE 0 END),0) AS detected_empty,
                      COALESCE(SUM(CASE WHEN highlight_sent > 0
                                         AND COALESCE(highlight_coarse_hits,0) > 0
                                        THEN 1 ELSE 0 END),0) AS dropped,
                      COALESCE(SUM(CASE WHEN highlight_sent IS NULL
                                        THEN 1 ELSE 0 END),0) AS unknown
                    FROM videos WHERE {base}""").fetchone()
        return {"detected_empty": r["detected_empty"], "skipped": r["skipped"],
                "dropped": r["dropped"], "unknown": r["unknown"],
                "total": r["detected_empty"] + r["skipped"] + r["dropped"] + r["unknown"]}

    def mark_highlight_error(self, video_id: int, reason: str) -> None:
        """记录硬失败（源缺失/解码失败）。置错后退出队列，避免每次重试白跑。"""
        with self.conn() as c:
            c.execute("UPDATE videos SET highlight_error=? WHERE id=?",
                      (reason[:500], video_id))

    def reset_highlight(self, scope: str = "failed") -> int:
        """清除精华剪辑标记，用于改参数后重跑。

        failed = 仅清失败标记（源缺失等硬错误）
        all    = 全部重来（已切出的片段文件不会被删除，新产物会加序号后缀）
        """
        with self.conn() as c:
            if scope == "failed":
                cur = c.execute(
                    "UPDATE videos SET highlight_error=NULL WHERE highlight_error IS NOT NULL")
            else:
                cur = c.execute(
                    "UPDATE videos SET highlight_at=NULL, highlight_count=NULL, "
                    "highlight_sec=NULL, highlight_clips=NULL, highlight_error=NULL")
            return cur.rowcount

    def list_highlight_rows(self, limit: int = 5000) -> List[sqlite3.Row]:
        """已出精华结果的记录（含 clips JSON），复核页用。"""
        with self.conn() as c:
            return c.execute(
                "SELECT id, filename, year, trash_path, highlight_at, highlight_count, "
                "highlight_sec, highlight_clips, highlight_error, highlight_sent, "
                "highlight_coarse_hits, highlight_version FROM videos "
                "WHERE highlight_at IS NOT NULL OR highlight_error IS NOT NULL "
                "ORDER BY id DESC LIMIT ?", (int(limit or 5000),)).fetchall()

    def recycle_scan_rows(self, limit: int = 200_000,
                          trashed_only: bool = True) -> List[sqlite3.Row]:
        """扫回收目录里的原片，取判定**保全状态**所需的最小列集。

        保全状态 = 已剪辑 / 待保全 / 无精华产物（判据在 mover.recycle_state，
        db 不重复实现）。判定要现算 highlight_clips，所以列里必须带它；
        还要带 highlight_at / highlight_error / highlight_version /
        highlight_coarse_hits —— 「待保全」就是照这四列算的。

        绝不 SELECT *：这张表每行还挂着 raw_detections、转写这些大 JSON，
        全库扫一遍能把内存吃满。

        trashed_only=True（默认）只看 trash_path 非空的行 —— 这正是"回收目录里
        有什么"的全部；仍在原位、尚未处置的行不用归位（下次 move_one 会落对）。
        """
        where = "WHERE trash_path IS NOT NULL" if trashed_only else ""
        with self.conn() as c:
            return c.execute(
                "SELECT id, filename, path, human_decision, trash_path, trash_tier, "
                "year, rel_dir, batch_id, highlight_at, highlight_error, "
                "highlight_version, highlight_coarse_hits, highlight_clips "
                f"FROM videos {where} ORDER BY id ASC LIMIT ?",
                (int(limit or 200_000),)).fetchall()

    def pending_highlight_ids(self) -> set:
        """全库「还没剪完」的 id 集合 —— 两条队列 ∪ 硬失败行。

        硬失败行（highlight_error 非空）当前不在任何队列里（要等 reset_highlight()
        才回去），但对"这条原片还能不能删"来说，它们和排队中的行一样是**没保全**
        的，所以必须并进来。少并这一类，回收目录就会告诉机主"这些可以删"。

        它的**唯一职责**是给 mover.is_pending() 的行内判断做交叉核对（selftest
        有断言）。运行时不走这里 —— 批量路径若每条都查一遍队列 SQL 太贵。
        """
        ids = {r["id"] for r in self.highlight_queue(limit=10 ** 9, cols="id")}
        ids |= {r["id"] for r in self.rehighlight_queue(
            limit=10 ** 9, only_dropped=True,
            cols="id, highlight_clips, highlight_coarse_hits")}
        with self.conn() as c:
            ids |= {r[0] for r in c.execute(
                "SELECT id FROM videos WHERE highlight_error IS NOT NULL "
                "AND human_decision IS NOT 'keep'")}
        return ids

    def rehighlight_queue(self, limit: int = 5000,
                          target_version: str = HIGHLIGHT_TARGET_VERSION,
                          only_dropped: bool = True,
                          cols: str = "*") -> List[sqlite3.Row]:
        """按产物口径重剪的队列（Track C · C5）。

        只认「已剪辑但版本不同」的行 —— 没剪过的交给 highlight_queue，
        版本已是目标值的跳过，天然幂等（中断后重跑不会重复劳动）。

        only_dropped=True 时只挑**收益最集中的那一批**：
          - 粗扫看到过目标、最终零产物（D2 受害者，删除后一个字都不留）
          - 只有静帧没有 mp4 的（D1/D4，C 级此前不切视频）
        这是收益最集中的那一批的谓词；False 则全量重剪。
        （别把"多少条"写进注释 —— 这个数是活的，2026-09-13 实测 1,446 条。）

        "有没有 mp4" 用解析后的 clips 判断而不是 SQL 里 LIKE 字符串 ——
        clips 是 JSON，靠 LIKE 匹配引号空格等于把格式当成契约，格式一变静默失效。

        ⚠️ 版本号既是这道门，也是本项目**唯一**的纠偏手段 —— 因为它是"事后按
        产物状态补救"的唯一入口。踩过的坑：hlv2 时代有两处逻辑会静默吃掉整段
        产物（见 highlighter.merge_two_levels / merge_events），产出行是
        `version='hlv2' + clips=[]`，被 `version <> target` 永久挡在队列外，
        回收目录据此判成"可直接删"（实测 630 条，其中 489 条已在回收目录里）。
        修法是**修掉那两处 + bump 到 hlv3**，而不是给谓词加症状分支：症状分支
        会自我循环（重剪后仍零产物 ⇒ 又被选中），bump 版本天生幂等。
        """
        sql = (f"SELECT {cols} FROM videos WHERE highlight_at IS NOT NULL "
               "AND highlight_error IS NULL "
               "AND COALESCE(highlight_version, 'hlv1') <> ? "
               "ORDER BY id ASC")
        with self.conn() as c:
            rows = c.execute(sql, (target_version,)).fetchall()
        if not only_dropped:
            return rows[:int(limit or 5000)]

        out: List[sqlite3.Row] = []
        for r in rows:
            try:
                clips = json.loads(r["highlight_clips"] or "[]")
            except (TypeError, ValueError):
                clips = []
            dropped = (not clips) and (r["highlight_coarse_hits"] or 0) > 0
            thumb_only = bool(clips) and not any(c.get("file") for c in clips)
            if dropped or thumb_only:
                out.append(r)
                if len(out) >= int(limit or 5000):
                    break
        return out

    def count_rehighlight_pending(self, target_version: str = HIGHLIGHT_TARGET_VERSION,
                                  only_dropped: bool = True) -> int:
        # 数数只用到 id / clips / coarse_hits —— 别把整行（含转写、raw_detections）
        # 读进内存，重剪队列经常是上千条。
        return len(self.rehighlight_queue(limit=1_000_000,
                                          target_version=target_version,
                                          only_dropped=only_dropped,
                                          cols="id, highlight_clips, highlight_coarse_hits"))

    def reset_highlight_version(self, version: str = "hlv1") -> int:
        """标记产物口径版本。用于回滚：置回 hlv1 即让重剪队列重新生效。"""
        with self.conn() as c:
            cur = c.execute("UPDATE videos SET highlight_version=? "
                            "WHERE highlight_at IS NOT NULL", (version,))
            return cur.rowcount

    def list_moved(self, batch_id: int | None = None, limit: int = 5000) -> List[sqlite3.Row]:
        """已移入回收目录、可撤销的文件。"""
        sql = ("SELECT * FROM videos WHERE human_decision='trash' "
               "AND trash_path IS NOT NULL")
        args: List[Any] = []
        if batch_id:
            sql += " AND batch_id=?"
            args.append(batch_id)
        sql += " ORDER BY id DESC LIMIT ?"
        args.append(limit)
        with self.conn() as c:
            return c.execute(sql, args).fetchall()

    def list_feedback(self, limit: int = 500) -> List[sqlite3.Row]:
        with self.conn() as c:
            return c.execute(
                "SELECT * FROM feedback ORDER BY id DESC LIMIT ?", (limit,)).fetchall()

    def stats(self, batch_id: int | None = None) -> Dict[str, int]:
        where, args = "", []
        if batch_id:
            where = " WHERE batch_id=?"
            args.append(batch_id)
        with self.conn() as c:
            def one(q):
                row = c.execute(q + (where or ""), args).fetchone()
                return row[0] if row else 0
            return {
                "total": one("SELECT COUNT(*) FROM videos"),
                "meta_ok": one("SELECT COUNT(*) FROM videos WHERE meta_ok=1"),
                "analyzed": one("SELECT COUNT(*) FROM videos WHERE status='analyzed'"),
                "verdict_trash": one("SELECT COUNT(*) FROM videos WHERE verdict='trash'"),
                "verdict_review": one("SELECT COUNT(*) FROM videos WHERE verdict='review'"),
                "verdict_keep": one("SELECT COUNT(*) FROM videos WHERE verdict='keep'"),
                "human_keep": one("SELECT COUNT(*) FROM videos WHERE human_decision='keep'"),
                "human_trash": one("SELECT COUNT(*) FROM videos WHERE human_decision='trash'"),
                "errors": one("SELECT COUNT(*) FROM videos WHERE status='error'"),
                "feedback": c.execute("SELECT COUNT(*) FROM feedback").fetchone()[0],
            }

    def feedback_summary(self) -> List[sqlite3.Row]:
        with self.conn() as c:
            return c.execute(
                """SELECT reason_category, COUNT(*) AS n
                   FROM feedback GROUP BY reason_category ORDER BY n DESC""").fetchall()

    def space_report(self, batch_id: int | None = None) -> Dict[str, Any]:
        """按年份与置信档位统计「已移入回收目录」的文件数与容量。

        这是人工删除前的核对依据：能直接看出哪一年、哪一档释放多少空间。
        """
        sql = """SELECT COALESCE(year,'_未分类') AS y, COALESCE(trash_tier,'?') AS tier,
                        COUNT(*) AS n,
                        COALESCE(SUM(size_bytes),0) AS bytes,
                        COALESCE(SUM(duration_sec),0) AS secs
                 FROM videos WHERE human_decision='trash'"""
        args: List[Any] = []
        if batch_id:
            sql += " AND batch_id=?"
            args.append(batch_id)
        sql += " GROUP BY y, tier ORDER BY y, tier"
        with self.conn() as c:
            rows = c.execute(sql, args).fetchall()
        by_year: Dict[str, Dict[str, Any]] = {}
        for r in rows:
            y = by_year.setdefault(r["y"], {"year": r["y"], "A": 0, "B": 0,
                                            "n": 0, "bytes": 0, "secs": 0.0})
            y[r["tier"]] = y.get(r["tier"], 0) + r["n"]
            y["n"] += r["n"]
            y["bytes"] += r["bytes"] or 0
            y["secs"] += r["secs"] or 0
        total = {
            "n": sum(v["n"] for v in by_year.values()),
            "bytes": sum(v["bytes"] for v in by_year.values()),
            "secs": sum(v["secs"] for v in by_year.values()),
            "A": sum(v.get("A", 0) for v in by_year.values()),
            "B": sum(v.get("B", 0) for v in by_year.values()),
        }
        return {"by_year": sorted(by_year.values(), key=lambda v: v["year"]), "total": total}

    def export_csv(self, out_path: str | Path, batch_id: int | None = None) -> Path:
        """导出待清理清单（供用户手动删除时核对）。"""
        import csv
        sql = """SELECT path, trash_path, trash_tier, verdict,
                        COALESCE(final_score, retention_score) AS score,
                        duration_sec, size_bytes, has_person, person_conf,
                        has_vehicle, motion_area_ratio, year
                 FROM videos WHERE human_decision='trash'"""
        args: List[Any] = []
        if batch_id:
            sql += " AND batch_id=?"
            args.append(batch_id)
        sql += " ORDER BY score ASC"
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with self.conn() as c:
            rows = c.execute(sql, args).fetchall()
        with out_path.open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["原路径", "回收后路径", "置信档位", "AI判定", "分数",
                        "时长(秒)", "大小(字节)", "检出人", "人的置信度",
                        "检出车", "前景面积占比", "年份"])
            for r in rows:
                w.writerow([r["path"], r["trash_path"] or "", r["trash_tier"] or "",
                            r["verdict"], r["score"], r["duration_sec"], r["size_bytes"],
                            r["has_person"], r["person_conf"], r["has_vehicle"],
                            r["motion_area_ratio"], r["year"]])
        return out_path
