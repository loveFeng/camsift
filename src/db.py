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
from typing import Any, Dict, Iterable, List, Optional

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

    -- L2 VLM 复核
    vlm_has_person   INTEGER,
    vlm_has_vehicle  INTEGER,
    vlm_confidence   REAL,
    vlm_evidence     TEXT,
    vlm_agree        INTEGER,
    vlm_model        TEXT,
    vlm_reviewed_at  TEXT,
    final_score      REAL,

    -- L2.5 音频复核（有声事件捞回，见开发计划 3.6）
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
        for col, decl in {**AUDIO_COLUMNS, **ARCHIVE_COLUMNS}.items():
            if col not in existing:
                c.execute(f"ALTER TABLE videos ADD COLUMN {col} {decl}")

    # ------------------------------------------------------------ 写入
    def upsert_video_meta(self, path: str, meta: Dict[str, Any]) -> int:
        """插入或更新一条视频的元数据。返回 video id。"""
        now = time.time()
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

    def update_vlm(self, video_id: int, r: Dict[str, Any]) -> None:
        with self.conn() as c:
            c.execute(
                """UPDATE videos SET vlm_has_person=?, vlm_has_vehicle=?, vlm_confidence=?,
                   vlm_evidence=?, vlm_agree=?, vlm_model=?, vlm_reviewed_at=?, final_score=?
                   WHERE id=?""",
                (
                    r.get("has_person"), r.get("has_vehicle"), r.get("confidence"),
                    r.get("evidence"), r.get("agree"), r.get("model"),
                    time.strftime("%Y-%m-%d %H:%M:%S"), r.get("final_score"),
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
        """记录人工处置。decision: keep | trash"""
        with self.conn() as c:
            c.execute(
                """UPDATE videos SET human_decision=?, human_at=?,
                   status=?, trash_path=COALESCE(?, trash_path),
                   trash_tier=COALESCE(?, trash_tier)
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

    def review_queue(self, batch_id: int | None = None, limit: int = 200,
                     only_unreviewed: bool = True, offset: int = 0,
                     speech_only: bool = False) -> List[sqlite3.Row]:
        """审核队列：按最终分升序 —— 最可疑的排最前。offset 用于分页。

        speech_only=True 时只看「音频复核检出人声」的片段——画面被判删但画外
        可能有人声事件，清理前值得单独过一遍。
        """
        sql = "SELECT * FROM videos WHERE verdict IN ('trash','review') AND meta_ok=1"
        args: List[Any] = []
        if only_unreviewed:
            sql += " AND human_decision IS NULL"
        if speech_only:
            sql += " AND audio_checked=1 AND speech_present=1"
        if batch_id:
            sql += " AND batch_id=?"
            args.append(batch_id)
        sql += " ORDER BY COALESCE(final_score, retention_score, 0) ASC, id ASC LIMIT ? OFFSET ?"
        args += [limit, int(offset or 0)]
        with self.conn() as c:
            return c.execute(sql, args).fetchall()

    def count_review_queue(self, batch_id: int | None = None,
                           only_unreviewed: bool = True,
                           speech_only: bool = False) -> int:
        sql = "SELECT COUNT(*) FROM videos WHERE verdict IN ('trash','review') AND meta_ok=1"
        args: List[Any] = []
        if only_unreviewed:
            sql += " AND human_decision IS NULL"
        if speech_only:
            sql += " AND audio_checked=1 AND speech_present=1"
        if batch_id:
            sql += " AND batch_id=?"
            args.append(batch_id)
        with self.conn() as c:
            return c.execute(sql, args).fetchone()[0]

    def count_audio_pending(self, batch_id: int | None = None,
                            only_unreviewed: bool = True,
                            scope: str = "all") -> int:
        """待音频复核的判删候选数（有音轨 + 未复核 + 尚未人工处置）。

        scope: all = 待清理+需人工确认（默认） / trash = 仅待清理 / review = 仅需人工确认
        """
        verdict_sql = _scope_sql(scope)
        sql = (f"SELECT COUNT(*) FROM videos WHERE verdict {verdict_sql} "
               "AND meta_ok=1 AND has_audio=1")
        args: List[Any] = []
        if only_unreviewed:
            sql += " AND audio_checked IS NULL AND human_decision IS NULL"
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
        sql = (f"SELECT * FROM videos WHERE verdict {verdict_sql} "
               "AND meta_ok=1 AND has_audio=1 AND audio_checked IS NULL "
               "AND human_decision IS NULL")
        args: List[Any] = []
        if batch_id:
            sql += " AND batch_id=?"
            args.append(batch_id)
        sql += " ORDER BY COALESCE(final_score, retention_score, 0) ASC, id ASC LIMIT ? OFFSET ?"
        args += [limit, int(offset or 0)]
        with self.conn() as c:
            return c.execute(sql, args).fetchall()

    # ------------------------------------------------------------ 语音档案
    def archive_queue(self, limit: int = 5000) -> List[sqlite3.Row]:
        """待归档队列：已移入回收目录 + 确认为有效语音 + 尚未归档。

        谓词与 list_moved 一致（human_decision='trash' 且 trash_path 非空），
        撤销移动后 human_decision 变 keep，自动退出队列。
        """
        sql = ("SELECT * FROM videos WHERE human_decision='trash' "
               "AND trash_path IS NOT NULL AND speech_present=1 "
               "AND archived_at IS NULL ORDER BY id ASC LIMIT ?")
        with self.conn() as c:
            return c.execute(sql, (int(limit or 5000),)).fetchall()

    def count_archive_pending(self) -> int:
        """待归档条数（已入回收目录的有效语音，未归档）。"""
        sql = ("SELECT COUNT(*) FROM videos WHERE human_decision='trash' "
               "AND trash_path IS NOT NULL AND speech_present=1 "
               "AND archived_at IS NULL")
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
