"""
扫描器 —— 遍历目录 + ffprobe 提取元数据

设计要点
--------
1. **只读 header，不解码**：ffprobe 只解析容器头，单文件 0.1-0.3 秒。
2. **串行读盘**：源盘是 USB 机械盘，并发随机读会让磁头反复寻道，反而更慢。
3. **断点续跑**：已入库且 meta_ok=1 的文件默认跳过，随时中断随时继续。
4. **年份即批次**：从路径第一级目录解析 4 位年份，作为天然批次单位。

本模块不对源文件做任何写操作。
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Callable, Dict, Iterator, Optional

from .config import bin_path

VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".ts", ".flv", ".wmv", ".m4v", ".dav", ".h264", ".h265", ".hevc"}

_YEAR_RE = re.compile(r"(19|20)\d{2}")


def find_ffprobe(cfg: Dict) -> str:
    found = bin_path(cfg, "ffprobe_path", "ffprobe")
    if not found:
        raise RuntimeError("未找到 ffprobe，请在 config.json 的 runtime.ffprobe_path 指定路径")
    return found


def iter_video_files(root: str | Path, exts=VIDEO_EXTS) -> Iterator[Path]:
    root = Path(root)
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for fn in sorted(filenames):
            if Path(fn).suffix.lower() in exts:
                yield Path(dirpath) / fn


def parse_year(path: Path, root: Path) -> str:
    """从相对路径中解析年份 —— 优先取第一级目录，其次取文件名。"""
    try:
        rel = path.relative_to(root)
    except ValueError:
        rel = Path(path.name)
    parts = rel.parts
    for candidate in (parts[0] if parts else "", path.name):
        m = _YEAR_RE.search(candidate)
        if m:
            return m.group(0)
    # 退而求其次：路径中任意一段出现 4 位年份
    for part in parts[:-1]:
        m = _YEAR_RE.search(part)
        if m:
            return m.group(0)
    return ""


def probe(path: Path, ffprobe: str, timeout: int = 30) -> Dict:
    """调用 ffprobe 取元数据。只解析 header，不解码。"""
    cmd = [
        ffprobe, "-v", "quiet", "-print_format", "json",
        "-show_format", "-show_streams", str(path),
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, timeout=timeout)
        if out.returncode != 0:
            return {"meta_ok": 0, "meta_error": f"ffprobe exit {out.returncode}"}
        data = json.loads(out.stdout.decode("utf-8", "ignore") or "{}")
    except subprocess.TimeoutExpired:
        return {"meta_ok": 0, "meta_error": "ffprobe timeout"}
    except Exception as e:  # noqa: BLE001
        return {"meta_ok": 0, "meta_error": f"{type(e).__name__}: {e}"}

    fmt = data.get("format", {}) or {}
    info: Dict = {
        "meta_ok": 1,
        "duration_sec": _f(fmt.get("duration")),
        "size_bytes": int(fmt.get("size") or path.stat().st_size),
        "bitrate_kbps": int(_f(fmt.get("bit_rate") or 0) / 1000) if fmt.get("bit_rate") else 0,
        "has_audio": 0,
        "video_codec": "",
        "width": 0,
        "height": 0,
    }
    for st in data.get("streams", []) or []:
        if st.get("codec_type") == "video":
            info["video_codec"] = st.get("codec_name") or ""
            info["width"] = int(st.get("width") or 0)
            info["height"] = int(st.get("height") or 0)
            # 有些容器 format.duration 缺失，用视频流时长兜底
            if not info["duration_sec"]:
                info["duration_sec"] = _f(st.get("duration"))
        elif st.get("codec_type") == "audio":
            info["has_audio"] = 1
    return info


def _f(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def audio_energy(path: Path, ffmpeg: str = "ffmpeg", timeout: int = 60) -> Optional[float]:
    """用 volumedetect 测平均音量。无音轨或失败返回 None。

    只在 has_audio=1 时调用，避免无谓的解码开销。
    """
    cmd = [ffmpeg, "-v", "quiet", "-i", str(path), "-af", "volumedetect", "-f", "null", "-"]
    try:
        out = subprocess.run(cmd, capture_output=True, timeout=timeout)
        txt = out.stderr.decode("utf-8", "ignore")
        m = re.search(r"mean_volume:\s*([-\d.]+)\s*dB", txt)
        if m:
            return float(m.group(1))
    except Exception:  # noqa: BLE001
        pass
    return None


class Scanner:
    def __init__(self, db, cfg: Dict):
        self.db = db
        self.cfg = cfg
        self.ffprobe = find_ffprobe(cfg)
        self.ffmpeg = bin_path(cfg, "ffmpeg_path", "ffmpeg") or "ffmpeg"

    def scan(
        self,
        root: str | Path,
        batch_name: str | None = None,
        limit: int | None = None,
        skip_existing: bool = True,
        with_audio_energy: bool = False,
        progress: Optional[Callable[[int, int, str], None]] = None,
        log: Optional[Callable[[str], None]] = None,
        stop_event=None,
        refresh_every: int = 50,
    ) -> Dict:
        """扫描目录并把元数据入库。返回统计字典。

        stop_event: threading.Event，置位后在当前文件处理完立即退出。
        refresh_every: 每处理 N 个文件回写一次批次进度，让 UI 进度条中途可见。
        """
        root = Path(root)
        if not root.exists():
            raise FileNotFoundError(f"源目录不存在：{root}")

        batch_id = self.db.create_batch(batch_name or root.name, str(root))
        if log:
            log(f"创建批次 #{batch_id}  {root}")

        done = skipped = failed = stopped = 0
        t0 = time.time()
        for i, path in enumerate(iter_video_files(root)):
            if stop_event is not None and stop_event.is_set():
                stopped = True
                break
            if limit and done >= limit:
                break
            spath = str(path)

            if skip_existing:
                row = self.db.get_video_by_path(spath)
                if row and row["meta_ok"]:
                    skipped += 1
                    continue

            info = probe(path, self.ffprobe)
            info["filename"] = path.name
            info["rel_dir"] = str(path.parent.relative_to(root)) if path.parent != root else ""
            info["year"] = parse_year(path, root)
            try:
                info["mtime"] = path.stat().st_mtime
            except OSError:
                info["mtime"] = 0

            if info.get("meta_ok") and with_audio_energy and info.get("has_audio"):
                db_val = audio_energy(path, self.ffmpeg)
                info["audio_energy"] = db_val

            vid = self.db.upsert_video_meta(spath, info)
            self.db.set_video_batch(vid, batch_id)
            if info.get("meta_ok"):
                done += 1
            else:
                failed += 1
                if log:
                    log(f"[元数据失败] {path.name}: {info.get('meta_error')}")

            if (i + 1) % max(1, refresh_every) == 0:
                self.db.refresh_batch_progress(batch_id)
                if progress:
                    progress(done, done + skipped + failed, path.name)

        n_bound = self.db.bind_batch(batch_id, root_dir=str(root))
        self.db.refresh_batch_progress(batch_id)
        self.db.update_batch(batch_id, status="stopped" if stopped else "scanned")
        if log:
            tail = "（已手动停止）" if stopped else ""
            log(f"扫描完成：新增 {done}，跳过 {skipped}，失败 {failed}，"
                f"绑定批次 {n_bound} 条，用时 {time.time() - t0:.1f}s{tail}")
        return {"batch_id": batch_id, "added": done, "skipped": skipped,
                "failed": failed, "stopped": stopped,
                "seconds": time.time() - t0}
