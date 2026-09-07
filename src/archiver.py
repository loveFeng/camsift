"""语音档案归档器（L2.7）。

把"已移入回收目录且确认为有效语音"的视频的音频资产永久收藏：
    16kHz 单声道 wav（从复核缓存复制，零转码）+ 同名转写 txt + manifest.jsonl 总清单

设计要点：
- **数据源是缓存 wav + 数据库转写**，不依赖视频本体 —— 视频被手动删除后仍可归档
- **永不删除/移动任何用户数据**：只做 copy，绝不碰源文件
- **幂等**：依赖 videos.archived_at 标记 + manifest.jsonl 双保险，重复执行只处理增量
- **红线**：归档目录绝不允许设在源盘/回收目录所在盘（会被手动清空）

```

{archive_dir}/2015/20151213211936_010.wav
{archive_dir}/2015/20151213211936_010.txt
{archive_dir}/manifest.jsonl
```
"""
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

ArchiverLog = Optional[Callable[[str], None]]
Progress = Optional[Callable[[int, int, str], None]]

MANIFEST_NAME = "manifest.jsonl"


class ArchiveError(Exception):
    """归档前置条件不满足（如目录在源盘、目录不可写）。"""


def same_drive(p1: str | Path, p2: str | Path) -> bool:
    """两路径是否在同一物理盘（按盘符判断）。任一路径不存在时按首字符比较。"""
    d1 = Path(p1).resolve().drive if Path(p1).exists() else Path(p1).drive
    d2 = Path(p2).resolve().drive if Path(p2).exists() else Path(p2).drive
    return (d1 or "").upper() == (d2 or "").upper()


def archive_dir_of(cfg: Dict[str, Any]) -> Path:
    """配置里的归档目录（未配置时回落默认）。"""
    return Path(cfg.get("archive", {}).get("dir") or "E:\\audio-archive")


def wav_cache_dir(cfg: Dict[str, Any]) -> Path:
    """音频复核缓存目录：work_dir/audio。"""
    work_dir = Path(cfg.get("work_dir") or ".")
    return work_dir / "audio"


def cached_wav(cfg: Dict[str, Any], video_id: int) -> Path:
    """某条视频的缓存 wav 路径（与 audioreview.extract_wav 命名一致）。"""
    return wav_cache_dir(cfg) / f"{video_id}.wav"


def check_ready(cfg: Dict[str, Any]) -> List[str]:
    """归档前置检查（空=就绪）。UI 自检与启动前调用。"""
    problems: List[str] = []
    adir = archive_dir_of(cfg)
    trash_root = (cfg.get("trash_root") or "").strip()
    source_root = (cfg.get("source_root") or "").strip()
    if trash_root and same_drive(adir, trash_root):
        problems.append(
            f"归档目录与回收目录同盘（{adir.drive}）—— 回收目录会被手动清空，"
            "档案必须放到另一块盘")
    if source_root and same_drive(adir, source_root):
        problems.append(
            f"归档目录与源目录同盘（{adir.drive}）—— 源盘上的视频会被删除，"
            "档案必须放到另一块盘")
    if str(adir).strip() in ("", "/", "\\"):
        problems.append("归档目录不能为空或盘符根目录")
    return problems


def _safe_stem(row: Dict[str, Any]) -> str:
    """归档文件名 stem：优先原视频文件名（含日期时间+通道，天然唯一带语义）。"""
    name = (row.get("filename") or "").strip() or Path(str(row.get("path") or "")).name
    stem = Path(name).stem or str(row.get("id"))
    # 去掉路径非法字符，避免跨设备拷贝出问题
    for ch in '<>:"/\\|?*':
        stem = stem.replace(ch, "_")
    return stem


def archive_one(cfg: Dict[str, Any], row: Dict[str, Any], adir: Path,
                ts: str) -> Dict[str, Any]:
    """归档单条。返回 {ok, wav, txt, stem, reason} —— 失败不抛异常，交由调用方统计。"""
    row = dict(row)
    vid = row.get("id")
    src_wav = cached_wav(cfg, vid)
    stem = _safe_stem(row)
    year = str(row.get("year") or "_未分类").strip() or "_未分类"
    out_dir = adir / year
    out_wav = out_dir / f"{stem}.wav"
    out_txt = out_dir / f"{stem}.txt"
    # 同名冲突（不同来源目录但文件名相同）→ 加数据库 id 后缀，绝不覆盖
    if out_wav.exists():
        out_wav = out_dir / f"{stem}_{vid}.wav"
        out_txt = out_dir / f"{stem}_{vid}.txt"

    if not src_wav.exists():
        return {"ok": False, "id": vid, "reason": "缓存 wav 缺失（可能已被清理）"}

    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_wav, out_wav)
    out_txt.write_text(row.get("transcript_text") or "", encoding="utf-8")

    record = {
        "id": vid,
        "src_path": row.get("path"),
        "trash_path": row.get("trash_path"),
        "year": row.get("year"),
        "duration_sec": row.get("duration_sec"),
        "speech_ratio": row.get("speech_ratio"),
        "speech_segments": row.get("speech_segments"),
        "transcript": row.get("transcript_text") or "",
        "llm_useful": row.get("llm_useful"),
        "llm_reason": row.get("llm_reason"),
        "archived_at": ts,
        "wav": str(out_wav),
        "txt": str(out_txt),
    }
    return {"ok": True, "id": vid, "stem": stem, "wav": out_wav, "txt": out_txt,
            "record": record}


def archive_pending(db, cfg: Dict[str, Any], limit: int = 5000,
                    progress: Progress = None, log: ArchiverLog = None,
                    stop_event=None) -> Dict[str, Any]:
    """归档所有待归档条目。返回统计 {done, skipped, failed, dir, manifest}。

    manifest.jsonl 追加写：每归档一条追加一行 JSON，作为训练元数据 / 检索索引 /
    幂等第二依据。
    """
    problems = check_ready(cfg)
    if problems:
        raise ArchiveError("；".join(problems))

    adir = archive_dir_of(cfg)
    adir.mkdir(parents=True, exist_ok=True)
    rows = db.archive_queue(limit=limit)
    total = len(rows)
    done = skipped = failed = 0
    manifest = adir / MANIFEST_NAME
    errors: List[str] = []

    with manifest.open("a", encoding="utf-8") as mf:
        for i, row in enumerate(rows, 1):
            if stop_event is not None and stop_event.is_set():
                if log:
                    log("归档已中断")
                break
            ts = datetime.now().isoformat(timespec="seconds")
            try:
                r = archive_one(cfg, row, adir, ts)
            except Exception as e:  # noqa: BLE001 —— 单条失败不拖垮批次
                failed += 1
                errors.append(f"id={row['id']}: {type(e).__name__}: {e}")
                if log:
                    log(f"[失败] id={row['id']} {type(e).__name__}: {e}")
                continue
            if not r.get("ok"):
                skipped += 1
                if log:
                    log(f"[跳过] id={row['id']} {r.get('reason')}")
                continue
            mf.write(json.dumps(r["record"], ensure_ascii=False) + "\n")
            mf.flush()
            db.mark_archived(r["id"], ts)
            done += 1
            if progress:
                progress(i, total, f"{r['stem']}.wav")
            if log:
                log(f"已归档 [{i}/{total}] {r['stem']}.wav")

    return {"done": done, "skipped": skipped, "failed": failed, "total": total,
            "dir": str(adir), "manifest": str(manifest), "errors": errors}
