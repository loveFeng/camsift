"""语音档案归档器（L2.7）。

把"已移入回收目录且确认为有效语音"的视频的音频资产永久收藏：
    16kHz 单声道 wav（从复核缓存复制，零转码）+ 同名转写 txt + manifest.jsonl 总清单

设计要点：
- **数据源是缓存 wav + 数据库转写**，不依赖视频本体 —— 视频被手动删除后仍可归档
- **永不删除/移动任何用户数据**：只做 copy，绝不碰源文件
- **幂等**：依赖 videos.archived_at 标记 + manifest.jsonl 双保险，重复执行只处理增量
- **目录同盘只是提示，不是限制**：归档目录与源盘/回收盘同盘时给出警告，但**允许执行**
  —— 删除是手动操作，用户清楚自己在删哪个文件夹，决定权留给人（见 check_warnings）

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

from .config import DEFAULT_CONFIG

ArchiverLog = Optional[Callable[[str], None]]
Progress = Optional[Callable[[int, int, str], None]]

MANIFEST_NAME = "manifest.jsonl"


class ArchiveError(Exception):
    """归档前置条件不满足（如目录在源盘、目录不可写）。"""


def same_drive(p1: str | Path, p2: str | Path) -> bool:
    """两路径是否在同一物理盘（按盘符判断）。

    resolve() 对不存在的路径也返回盘符（strict=False），所以"回收目录还没建"
    这种情形照样判得准；相对路径会先按 cwd 补全。
    """
    return Path(p1).resolve().drive.lower() == Path(p2).resolve().drive.lower()


def is_drive_root(p: str | Path) -> bool:
    """是否为盘符根目录（`G:\\` / `/`）或空路径 —— 这类目录不能当产出根目录。

    注意不能只比 `"/"`：Windows 的 `G:\\` 也是一个盘的根，单独漏掉它会导致
    用户把整个盘的根设成产出目录。
    """
    s = str(p).strip()
    if s in ("", "/", "\\"):
        return True
    try:
        return bool(Path(p).drive) and s.rstrip("\\/") == Path(p).drive
    except (OSError, ValueError):
        return False


def archive_dir_of(cfg: Dict[str, Any]) -> Path:
    """配置里的归档目录（未配置时回落到 config.DEFAULT_CONFIG 的那份默认）。"""
    return Path(cfg.get("archive", {}).get("dir") or DEFAULT_CONFIG["archive"]["dir"])


def wav_cache_dir(cfg: Dict[str, Any]) -> Path:
    """音频复核缓存目录：work_dir/audio。"""
    work_dir = Path(cfg.get("work_dir") or ".")
    return work_dir / "audio"


def cached_wav(cfg: Dict[str, Any], video_id: int) -> Path:
    """某条视频的缓存 wav 路径（与 audioreview.extract_wav 命名一致）。"""
    return wav_cache_dir(cfg) / f"{video_id}.wav"


def check_ready(cfg: Dict[str, Any]) -> List[str]:
    """归档**硬**前置检查（空=就绪，非空则禁止执行）。

    只放真正会让归档失败或破坏数据的条件。目录与源盘/回收盘同盘属于风险提示，
    不是错误——删除是手动操作，用户自己可控，所以由 check_warnings 负责。
    """
    problems: List[str] = []
    adir = archive_dir_of(cfg)
    if is_drive_root(adir):
        problems.append("归档目录不能为空或盘符根目录")
    return problems


def same_drive_warnings(out_dir: Path, cfg: Dict[str, Any], what: str) -> List[str]:
    """产出目录与源盘/回收盘同盘的**软**提示（归档与精华剪辑共用）。

    同盘不禁止：回收目录是被手动清空的，用户清楚自己在删哪个文件夹。
    但同盘确实意味着整盘出事时产出会一起没，所以提示到，决定权留给人。
    """
    warns: List[str] = []
    for key, other in (("trash_root", "回收目录"), ("source_root", "源目录")):
        root = (cfg.get(key) or "").strip()
        if root and same_drive(out_dir, root):
            warns.append(
                f"{what}与{other}同盘（{out_dir.drive}）—— {other}会被手动清空，"
                f"建议把{what}放到另一块盘（仅提示，不阻止执行）")
    return warns


def check_warnings(cfg: Dict[str, Any]) -> List[str]:
    """归档**软**风险提示（不影响执行，仅提醒）。"""
    return same_drive_warnings(archive_dir_of(cfg), cfg, "归档目录")


def safe_stem(row: Dict[str, Any]) -> str:
    """产出文件名 stem：优先原视频文件名（含日期时间+通道，天然唯一带语义）。

    语音档案与画面精华共用 —— 两边都要求"同名不同目录也不覆盖"，命名必须一致。
    """
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
    stem = safe_stem(row)
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
