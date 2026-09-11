"""画面精华剪辑器（L2.8）。

删除视频前的最后一道保险
------------------------
音频侧：L2.7 语音档案把"有效语音"存成 wav + 转写。
画面侧：本模块负责 —— 把画面里还有点东西的几秒剪下来留存。

两边都做完，原视频才敢真正删除。

为什么要两级检测
----------------
L1 判定阶段每段只抽 12-16 帧 —— 43 秒视频里只有 2.5% 的画面被看过。
小孩身形小、进出画面快，或者被门框家具挡住只剩半个身子，这种密度必然漏。
人工审核看的就是这 16 帧拼的缩略图，所以"人已经看过一遍"兜不住这个漏。

但把全片拉到 1280 精扫要 17 小时，不现实。所以分两级：

    第一级 粗扫   3fps @640  阈值 0.15   全片    —— 负责不漏，误检交给下一级
    第二级 精扫   4fps @1280 阈值 0.25   仅候选窗 —— 负责看清，小目标全靠它

粗扫阈值压到 0.15 是故意的：误检的代价是被精扫过滤掉（多花点算力），
漏检的代价是家庭影像永久消失。这两件事不对等。

三级产出（用分级目录代替人工筛选）
----------------------------------
全自动必然产生垃圾，所以不让人来筛，而是把"确定性"表达在目录结构里：

    A_确认/  精扫确认有目标，且 ≥8s 或多人同框   → mp4（含原音）+ 代表帧
    B_短暂/  精扫确认有目标，2-8s                → mp4（含原音）+ 代表帧
    C_疑似/  粗扫看到但精扫没确认，或只有大幅动静 → mp4（含原音）+ 多张代表帧

**C 级也切 mp4**（Track C · ADR 0006）：本模块的定位是"原片被删之前的最后一道保全"，
静帧只能证明"某一秒有过东西"，而"删除后再也找不到"要的是能回放的整段。
C 段不进人工审核队列（它是**档案**不是**待办**），所以"多留"不会变成"多看"。

流水线
------
    粗扫抽帧 → CLAHE → 帧差预筛 → 检测(0.15) → 候选窗 → 精扫抽帧 → 检测(0.25)
    → 事件聚合 → 分级 → A/B 重编码切割 / C 抽代表帧 → manifest

三个必须遵守的约束
------------------
1. **重编码切割**：监控 H.264 的 GOP 很长，-c copy 会把切点吸附到关键帧
   （误差数秒，开头还可能花屏）。必须重编码换帧级精确。
2. **保持源帧率**：15fps 素材强行拔到 30fps 只会徒增体积，没有任何新信息。
3. **同盘只是提示，不是限制**：输出目录与源盘 / 回收目录同盘时给出警告，但**允许执行**
   —— 删除是手动操作，用户清楚自己在删哪个文件夹，决定权留给人（见 check_warnings）。
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .archiver import MANIFEST_NAME, is_drive_root, safe_stem, same_drive_warnings
from .config import DEFAULT_CONFIG, bin_path

Log = Optional[Callable[[str], None]]
Progress = Optional[Callable[[int, int, str], None]]

# 文件名里的真实拍摄时间：20150619095035_010.mp4 → 2015-06-19 09:50:35
_TS_RE = re.compile(r"(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})")

# 产出分级目录名
GRADE_DIRS = {"A": "A_确认", "B": "B_短暂", "C": "C_疑似"}

# hit 来源等级：2 = 精扫/粗扫确认，1 = 疑似（低置信度 / 纯动静）
LV_CONFIRMED = 2
LV_SUSPECT = 1

# 产物口径版本（写进 videos.highlight_version，驱动重剪队列）：
#   hlv1 = 旧口径：粗扫疑似被精扫否决后**直接丢弃**；C 级只留一张静帧
#   hlv2 = 删前保全口径：粗扫疑似降级保留；C 级也切可播放 mp4
# 换口径时改这里（或配置 highlight.version），已剪过的行版本落后即自动进重剪队列。
HL_VERSION_LEGACY = "hlv1"
HL_VERSION = "hlv2"

# C 级每段留几张代表帧（配置 highlight.c_thumbs 的默认值）。
# 一张图证明不了"这 30 秒里发生过什么"，而多抽两张的成本约等于一次 ffprobe。
DEFAULT_C_THUMBS = 3


class HighlightError(Exception):
    """前置条件不满足（目录不合法 / 缺 ffmpeg / 缺模型）。"""


# --------------------------------------------------------------------------- 就绪检查
def highlight_dir_of(cfg: Dict[str, Any]) -> Path:
    return Path(cfg.get("highlight", {}).get("dir")
                or DEFAULT_CONFIG["highlight"]["dir"])


def _bin(cfg: Dict[str, Any], key: str, name: str) -> str:
    found = bin_path(cfg, key, name)
    if not found:
        raise HighlightError(f"未找到 {name}，请在配置里指定 runtime.{key}")
    return found


def check_ready(cfg: Dict[str, Any]) -> List[str]:
    """**硬**前置检查（空列表 = 就绪，非空则禁止执行）。

    只放真正跑不起来的条件（缺 ffmpeg / 缺模型 / 目录非法）。
    目录与源盘/回收盘同盘是风险提示不是错误，由 check_warnings 负责。
    """
    problems: List[str] = []
    hdir = highlight_dir_of(cfg)

    if is_drive_root(hdir):
        problems.append("精华目录不能为空或盘符根目录")
    for key, name in (("ffmpeg_path", "ffmpeg"), ("ffprobe_path", "ffprobe")):
        try:
            _bin(cfg, key, name)
        except HighlightError as e:
            problems.append(str(e))
    model = Path(cfg.get("analysis", {}).get("yolo_model_path") or "")
    if cfg.get("analysis", {}).get("yolo_enabled", True) and not model.exists():
        problems.append(f"未找到 YOLO 模型：{model}（可运行 scripts/download_model.py 下载）")
    return problems


def check_warnings(cfg: Dict[str, Any]) -> List[str]:
    """**软**风险提示（不影响执行，仅提醒）—— 与语音归档同一口径。"""
    return same_drive_warnings(highlight_dir_of(cfg), cfg, "精华目录")


# --------------------------------------------------------------------------- 工具
def parse_shot_time(stem: str) -> Optional[datetime]:
    """从文件名解析拍摄时刻 —— 给片段打上真实时间，家庭影像留念很值钱。"""
    m = _TS_RE.search(stem or "")
    if not m:
        return None
    try:
        return datetime(*[int(g) for g in m.groups()])
    except ValueError:
        return None


def probe_av(path: Path, ffprobe: str, timeout: int = 30) -> Dict[str, Any]:
    """取时长 / 帧率 / 是否有音轨。只解析 header，不解码。"""
    cmd = [ffprobe, "-v", "quiet", "-print_format", "json",
           "-show_format", "-show_streams", str(path)]
    out = {"duration": 0.0, "fps": 0.0, "has_audio": False}
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=timeout)
        data = json.loads(r.stdout.decode("utf-8", "ignore") or "{}")
    except Exception:  # noqa: BLE001
        return out

    try:
        out["duration"] = float((data.get("format") or {}).get("duration") or 0)
    except (TypeError, ValueError):
        pass
    for st in data.get("streams", []) or []:
        if st.get("codec_type") == "audio":
            out["has_audio"] = True
        elif st.get("codec_type") == "video":
            if not out["duration"]:
                try:
                    out["duration"] = float(st.get("duration") or 0)
                except (TypeError, ValueError):
                    pass
            rate = st.get("avg_frame_rate") or st.get("r_frame_rate") or "0/0"
            try:
                num, den = str(rate).split("/")
                if float(den or 0):
                    out["fps"] = float(num) / float(den)
            except (ValueError, ZeroDivisionError):
                pass
    return out


def _enhance(frame: np.ndarray, enabled: bool) -> np.ndarray:
    """CLAHE 对比度增强（作用于 LAB 的 L 通道，不动颜色）。

    红外/夜视画面低对比高噪点，人体作为热源本就比背景亮，
    拉开信噪比后检测器能明显多捡回一批目标。
    """
    if not enabled:
        return frame
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def _diff_ratio(a: np.ndarray, b: np.ndarray) -> float:
    """两帧之间变化像素的面积占比（降采样 + 高斯去噪后比较）。"""
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    ga = cv2.GaussianBlur(cv2.resize(ga, (320, 180)), (5, 5), 0)
    gb = cv2.GaussianBlur(cv2.resize(gb, (320, 180)), (5, 5), 0)
    d = cv2.absdiff(ga, gb)
    _, mask = cv2.threshold(d, 25, 255, cv2.THRESH_BINARY)
    return float(mask.sum()) / 255.0 / float(mask.shape[0] * mask.shape[1])


# --------------------------------------------------------------------------- 聚合
def merge_events(
    hits: List[Dict[str, Any]],
    duration: float,
    gap_merge: float = 3.0,
    min_duration: float = 1.0,
    pre_pad: float = 2.0,
    post_pad: float = 3.0,
    max_clips: int = 8,
    grade_a_seconds: float = 8.0,
    grade_a_persons: int = 2,
) -> List[Dict[str, Any]]:
    """把离散的"有内容帧"聚合成可切割的事件区间并分级。

    hits: [{t, conf, n_person, max_area, level}]，按时间升序。
    返回事件列表，按时间升序，含 start/end/dur/grade/score。
    """
    if not hits:
        return []

    # 1) 按间隔合并 —— 抵消遮挡、漏检抖动造成的断断续续
    groups: List[List[Dict[str, Any]]] = [[hits[0]]]
    for h in hits[1:]:
        if h["t"] - groups[-1][-1]["t"] <= gap_merge:
            groups[-1].append(h)
        else:
            groups.append([h])

    events: List[Dict[str, Any]] = []
    for g in groups:
        start, end = g[0]["t"], g[-1]["t"]
        if end - start < min_duration:
            continue        # 一闪而过，八成是误检
        level = max(x["level"] for x in g)
        conf = max(x["conf"] for x in g)
        n_person = max(x["n_person"] for x in g)
        best = max(g, key=lambda x: (x["level"], x["conf"]))

        # 2) 前后缓冲：避免人物"半路冒出来"、动作没收尾
        s = max(0.0, start - pre_pad)
        e = min(duration, end + post_pad)
        if e - s <= 0:
            continue
        dur = e - s

        # 3) 分级 —— 全自动下用目录表达确定性，而不是让人来筛
        grade = ("A" if (dur >= grade_a_seconds or n_person >= grade_a_persons)
                 else "B") if level >= LV_CONFIRMED else "C"
        score = (conf * 100.0 + min(dur, 30.0) * 2.0 + len(g) * 3.0
                 + (20.0 if n_person >= 2 else 0.0)
                 + (50.0 if level >= LV_CONFIRMED else 0.0))

        events.append({
            "start": round(s, 3), "end": round(e, 3), "dur": round(dur, 2),
            "level": level, "grade": grade, "score": round(score, 1),
            "conf": round(conf, 4), "n_person": n_person,
            "frames": len(g), "best_t": round(best["t"], 3),
        })

    if not events:
        return []

    # 4) 缓冲后可能重叠，再并一次（并完取最高等级）
    events.sort(key=lambda x: x["start"])
    merged: List[Dict[str, Any]] = [events[0]]
    for ev in events[1:]:
        last = merged[-1]
        if ev["start"] <= last["end"]:
            last["end"] = max(last["end"], ev["end"])
            last["dur"] = round(last["end"] - last["start"], 2)
            last["level"] = max(last["level"], ev["level"])
            last["n_person"] = max(last["n_person"], ev["n_person"])
            last["frames"] += ev["frames"]
            if ev["conf"] > last["conf"]:
                last["conf"] = ev["conf"]
                last["best_t"] = ev["best_t"]
            last["score"] = round(max(last["score"], ev["score"]), 1)
            last["grade"] = ("A" if (last["dur"] >= grade_a_seconds
                                     or last["n_person"] >= grade_a_persons)
                             else "B") if last["level"] >= LV_CONFIRMED else "C"
        else:
            merged.append(ev)

    # 5) 按价值排序取前 N，再按时间排回（输出顺序符合观看习惯）
    merged.sort(key=lambda x: (-x["score"], x["start"]))
    picked = merged[:max(1, int(max_clips))]
    picked.sort(key=lambda x: x["start"])
    return picked


def build_windows(hits: List[Dict[str, Any]], duration: float,
                  gap: float = 4.0, pad: float = 2.0) -> List[Tuple[float, float]]:
    """把粗扫候选帧合并成待精扫的时间窗（含前后缓冲）。"""
    if not hits:
        return []
    wins: List[List[float]] = [[hits[0]["t"], hits[0]["t"]]]
    for h in hits[1:]:
        if h["t"] - wins[-1][1] <= gap:
            wins[-1][1] = h["t"]
        else:
            wins.append([h["t"], h["t"]])
    out = []
    for a, b in wins:
        s, e = max(0.0, a - pad), min(duration, b + pad)
        if e > s:
            out.append((s, e))
    return out


def merge_two_levels(c_hits: List[Dict[str, Any]], f_hits: List[Dict[str, Any]],
                     c_diffs: List[float], c_fps: float,
                     scanned: List[Tuple[float, float]], fb_thr: float,
                     fine_on: bool,
                     keep_coarse_suspect: bool = True) -> Tuple[List[Dict[str, Any]], int]:
    """合并粗扫（640/0.15）与精扫（1280/0.25）两级的检出。返回 (hits, 降级保留条数)。

    这是 Track C 的核心（ADR 0006）——**精扫是"升级权"，不是"否决权"**：

        精扫确认              → 升为 A/B
        精扫没确认但粗扫看到   → 降级保留为 C 疑似（keep_coarse_suspect=False 时退回旧行为：丢弃）
        超预算没精扫到的窗     → 有动静则保留为 C 疑似
        只有动静没有目标       → C 疑似（精扫只认目标不认动静）

    为什么旧行为（丢弃）是错的：低照度素材上 1280 会把置信度压低
    （实测同一帧 0.42 → 0.19），"精扫没确认"不等于"没有"。
    旧口径因此让 474 条视频"粗扫看到过目标、最终零产物"，删除后一个字都不留。
    """
    hits = list(f_hits) if fine_on else list(c_hits)
    downgraded = 0
    if fine_on and c_hits:
        for x in c_hits:
            idx = min(int(x["t"] * c_fps), len(c_diffs) - 1) if c_diffs else 0
            if x.get("motion_only"):
                hits.append(dict(x, level=LV_SUSPECT))
                continue
            scanned_here = bool(scanned) and any(ws <= x["t"] <= we for ws, we in scanned)
            if not scanned_here:
                if c_diffs and c_diffs[idx] > fb_thr:
                    hits.append(dict(x, level=LV_SUSPECT))
                continue
            if keep_coarse_suspect:
                downgraded += 1
                hits.append(dict(x, level=LV_SUSPECT))
    hits.sort(key=lambda x: x["t"])
    return hits, downgraded


# --------------------------------------------------------------------------- 主流程
class HighlightCutter:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.h = cfg.get("highlight", {}) or {}
        self.ffmpeg = _bin(cfg, "ffmpeg_path", "ffmpeg")
        self.ffprobe = _bin(cfg, "ffprobe_path", "ffprobe")
        self.work_dir = Path(cfg.get("work_dir") or ".")
        self.coarse = self.fine = None
        if cfg.get("analysis", {}).get("yolo_enabled", True):
            from .analyzer import YoloDetector
            path = cfg["analysis"].get("yolo_model_path", "")
            nms = float(cfg["analysis"].get("nms_threshold", 0.45))
            self.coarse = YoloDetector(path, int(self.h.get("coarse_size", 640)),
                                       float(self.h.get("coarse_conf", 0.15)), nms)
            self.fine = YoloDetector(path, int(self.h.get("fine_size", 1280)),
                                     float(self.h.get("fine_conf", 0.25)), nms)
            if not self.coarse.available:
                self.coarse = self.fine = None

    # ------------------------------------------------------------------ 抽帧
    @property
    def hl_version(self) -> str:
        """当前产物口径版本。做成属性而不是 __init__ 里的字段 ——
        子类/测试常常只 set 一个 self.h，字段式赋值会让它们全部 AttributeError。"""
        return str(self.h.get("version") or HL_VERSION)

    def _extract(self, src: Path, out_dir: Path, fps: float, width: int,
                 dur: Optional[float] = None, start: float = 0.0) -> List[Path]:
        """抽帧。返回按序 jpg 列表，第 i 张对应全局时刻 start + i/fps。"""
        if out_dir.exists():
            shutil.rmtree(out_dir, ignore_errors=True)  # AUDIT-OK: 仅清理自创临时抽帧目录
        out_dir.mkdir(parents=True, exist_ok=True)
        pattern = str(out_dir / "f_%05d.jpg")
        n = int(max(1, min(3000, (dur or 60) * fps + 2)))
        cmd = [self.ffmpeg, "-v", "error"]
        if start > 0:
            cmd += ["-ss", f"{start:.3f}"]      # 输入 seek：快，解码时精确对齐
        cmd += ["-i", str(src)]
        if dur:
            cmd += ["-t", f"{dur:.3f}"]
        cmd += ["-vf", f"fps={fps:.4f},scale={int(width)}:-2",
                "-frames:v", str(n), "-q:v", "3", pattern]
        subprocess.run(cmd, capture_output=True, timeout=900)
        return sorted(out_dir.glob("f_*.jpg"))

    # ------------------------------------------------------------------ 单级检测
    def _scan(self, frames: List[Path], fps: float, t0: float, duration: float,
              detector, conf: float, prefilter: bool, keep: set,
              fb_thr: float) -> Tuple[List[Dict[str, Any]], List[float], int]:
        """增强 → 预筛 → 检测。返回 (hits, 每帧动静比, 实际送检帧数)。"""
        h = self.h
        clahe = bool(h.get("clahe", True))
        mthr = float(h.get("motion_threshold", 0.004))
        dilate = int(h.get("prefilter_dilate", 1))
        fallback = bool(h.get("motion_fallback", True))
        suspect = float(h.get("suspect_conf", 0.15))

        imgs: List[np.ndarray] = []
        for p in frames:
            img = cv2.imread(str(p))
            if img is not None:
                imgs.append(img)
        n = len(imgs)
        if n == 0:
            return [], [], 0

        # --- 帧差预筛：固定机位下背景恒定，"画面没变"的帧没必要送检测 ---
        active = [True] * n
        diffs = [0.0] * n
        if prefilter and n >= 2:
            raw = [False] * n
            for i in range(1, n):
                d = _diff_ratio(imgs[i - 1], imgs[i])
                diffs[i] = d
                diffs[i - 1] = max(diffs[i - 1], d)
                if d > mthr:
                    raw[i] = raw[i - 1] = True
            active = list(raw)
            if dilate > 0:      # 向两侧扩展，补回入场/出场那一两帧
                for i, v in enumerate(raw):
                    if v:
                        for j in range(max(0, i - dilate), min(n, i + dilate + 1)):
                            active[j] = True

        hits: List[Dict[str, Any]] = []
        sent = 0

        def _motion_hit(i: int) -> Optional[Dict[str, Any]]:
            if not (fallback and diffs[i] > fb_thr):
                return None
            return {"t": min(t0 + i / fps, max(0.0, duration)), "conf": 0.0,
                    "n_person": 0, "max_area": diffs[i], "level": LV_SUSPECT,
                    "motion_only": True}

        for i, img in enumerate(imgs):
            if not active[i]:
                mh = _motion_hit(i)
                if mh:
                    hits.append(mh)
                continue
            if detector is None:
                mh = _motion_hit(i)
                if mh:
                    hits.append(mh)
                continue

            sent += 1
            final, _raw = detector.detect(_enhance(img, clahe), raw_min=0.10)
            wanted = [d for d in final if d["class"] in keep]
            if not wanted:
                mh = _motion_hit(i)
                if mh:
                    hits.append(mh)
                continue
            best = max(wanted, key=lambda d: d["conf"])
            if best["conf"] >= conf:
                level = LV_CONFIRMED
            elif best["conf"] >= suspect:
                level = LV_SUSPECT
            else:
                mh = _motion_hit(i)
                if mh:
                    hits.append(mh)
                continue
            hits.append({
                "t": min(t0 + i / fps, max(0.0, duration)),
                "conf": best["conf"],
                "n_person": sum(1 for d in wanted if d["class"] == "person"),
                "max_area": max(d["area_ratio"] for d in wanted),
                "level": level, "motion_only": False,
            })
        return hits, diffs, sent

    # ------------------------------------------------------------------ 切割
    def _cut(self, src: Path, ev: Dict[str, Any], out: Path, fps: float,
             has_audio: bool) -> bool:
        """重编码切出单段。

        为什么必须重编码：监控 H.264 的 GOP 往往 5-10 秒一个关键帧，
        -c copy 只能把切点吸附到最近关键帧，误差可达数秒且开头可能花屏。
        """
        h = self.h
        cmd = [self.ffmpeg, "-v", "error", "-y",
               "-ss", f"{ev['start']:.3f}", "-i", str(src), "-t", f"{ev['dur']:.3f}",
               "-c:v", "libx264", "-crf", str(int(h.get("crf", 20))),
               "-preset", str(h.get("preset", "veryfast"))]
        # 源帧率即输出帧率 —— 15fps 素材强行拔到 30fps 只会徒增体积，没有新信息
        if fps and fps > 0:
            gop = max(1, int(round(fps * float(h.get("gop_seconds", 2)))))
            cmd += ["-r", f"{fps:g}", "-g", str(gop)]
        if has_audio:
            cmd += ["-c:a", "aac", "-b:a", str(h.get("audio_bitrate", "128k"))]
        else:
            cmd += ["-an"]
        cmd += ["-movflags", "+faststart", str(out)]
        r = subprocess.run(cmd, capture_output=True, timeout=900)
        return r.returncode == 0 and out.exists() and out.stat().st_size > 0

    def _thumb(self, src: Path, t: float, out: Path) -> bool:
        """抽代表帧：取检测置信度最高的那一瞬间。C 级时这张图就是唯一产物。"""
        out.parent.mkdir(parents=True, exist_ok=True)
        cmd = [self.ffmpeg, "-v", "error", "-y", "-ss", f"{max(0.0, t):.3f}",
               "-i", str(src), "-frames:v", "1",
               "-vf", f"scale={int(self.h.get('thumb_width', 640))}:-2",
               "-q:v", "4", str(out)]
        r = subprocess.run(cmd, capture_output=True, timeout=120)
        return r.returncode == 0 and out.exists()

    @staticmethod
    def _thumb_times(ev: Dict[str, Any], n: int) -> List[float]:
        """一个事件要几张代表帧时，取哪些时刻。

        第一张永远是**最高置信度**那一刻（最可能有内容的证据帧），
        其余在本段时长内均匀铺开 —— 单张图证明不了"这 30 秒里发生过什么"，
        三张才能大致还原，而多抽两张的成本约等于一次 ffprobe。
        """
        n = max(1, int(n))
        best = float(ev.get("best_t", ev["start"]))
        if n == 1:
            return [best]
        s, e = float(ev["start"]), float(ev["end"])
        times = [best]
        for k in range(1, n):
            t = s + (e - s) * (k / n)
            if all(abs(t - x) > 0.2 for x in times):
                times.append(t)
            if len(times) >= n:
                break
        while len(times) < n:           # 段太短，铺不满就贴着 best 前后微偏移
            times.append(best + 0.3 * len(times))
        return times[:n]

    # ------------------------------------------------------------------ 单条
    def cut_one(self, row: Dict[str, Any], hdir: Path, ts: str,
                log: Log = None) -> Dict[str, Any]:
        """处理单个视频。返回 {ok, id, stem, clips, sec, reason, stats}。"""
        row = dict(row)
        vid = row.get("id")
        t_start = time.time()
        # 已移入回收目录后用回收路径（源路径此时已失效）
        src = None
        for key in ("trash_path", "path"):
            v = row.get(key)
            if v and Path(v).exists():
                src = Path(v)
                break
        if src is None:
            return {"ok": False, "id": vid, "reason": "视频文件已不存在（可能已手动删除）"}

        info = probe_av(src, self.ffprobe)
        duration = info["duration"] or float(row.get("duration_sec") or 0)
        if duration <= 0:
            return {"ok": False, "id": vid, "reason": "无法读取时长（文件损坏？）"}

        h = self.h
        stem = safe_stem(row)
        year = str(row.get("year") or "_未分类").strip() or "_未分类"
        keep: set = set()
        if h.get("keep_person", True):
            keep.add("person")
        if h.get("keep_vehicle", False):
            keep |= {"car", "truck", "bus", "motorcycle", "bicycle", "train"}
        if h.get("keep_animal", False):
            keep |= {"cat", "dog", "bird", "horse", "sheep", "cow"}
        fb_thr = float(h.get("motion_fallback_threshold", 0.05))

        tmp = self.work_dir / "highlight_frames" / str(vid)
        stats: Dict[str, Any] = {"duration": round(duration, 1)}
        try:
            # ---------------- 第一级：粗扫全片 ----------------
            c_fps = float(h.get("coarse_fps", 3.0))
            c_frames = self._extract(src, tmp / "coarse", c_fps,
                                     int(h.get("coarse_size", 640)), duration)
            if not c_frames:
                return {"ok": False, "id": vid, "reason": "抽帧失败（无法解码）"}
            stats["coarse_frames"] = len(c_frames)
            c_hits, c_diffs, c_sent = self._scan(
                c_frames, c_fps, 0.0, duration, self.coarse,
                float(h.get("coarse_conf", 0.15)),
                bool(h.get("motion_prefilter", True)), keep, fb_thr)
            stats["coarse_sent"] = c_sent
            stats["coarse_hits"] = len(c_hits)

            # ---------------- 第二级：精扫候选窗 ----------------
            f_hits: List[Dict[str, Any]] = []
            fine_on = bool(h.get("fine_enabled", True)) and self.fine is not None
            windows: List[Tuple[float, float]] = []
            if fine_on and c_hits:
                # 只有"真有动静"处的粗扫检出才值得精扫 —— 静态画面上的低置信度
                # 检出几乎一定是噪点，精扫也是浪费
                seeds = [x for x in c_hits
                         if not x.get("motion_only")
                         or c_diffs[min(int(x["t"] * c_fps), len(c_diffs) - 1)] > fb_thr]
                windows = build_windows(
                    seeds, duration, gap=float(h.get("gap_merge", 3.0)) + 1.0,
                    pad=float(h.get("fine_window_pad", 2.0)))
                # 预算控制：精扫帧数不超过粗扫的 max_fine_ratio 倍，超了只精扫高分窗
                f_fps = float(h.get("fine_fps", 4.0))
                budget = max(30, int(len(c_frames) * float(h.get("max_fine_ratio", 1.5))))
                windows.sort(key=lambda w: -max(
                    (x["conf"] for x in seeds if w[0] <= x["t"] <= w[1]), default=0.0))
                used = 0
                scanned: List[Tuple[float, float]] = []
                stats["windows"] = len(windows)
                stats["fine_sent"] = 0
                for i, (ws, we) in enumerate(windows):
                    wd = we - ws
                    if used + wd * f_fps > budget:
                        continue        # 超预算的窗不精扫 —— 其粗扫检出按疑似保留
                    used += wd * f_fps
                    scanned.append((ws, we))
                    frames = self._extract(src, tmp / f"w{i}", f_fps,
                                           int(h.get("fine_size", 1280)), wd, ws)
                    if not frames:
                        continue
                    got, _d, sent = self._scan(
                        frames, f_fps, ws, duration, self.fine,
                        float(h.get("fine_conf", 0.25)), False, keep, fb_thr)
                    stats["fine_sent"] += sent
                    f_hits.extend(got)
                stats["fine_frames"] = int(used)
                stats["windows_scanned"] = len(scanned)

            # ---------------- 合并两级结果 ----------------
            # 规则全在 merge_two_levels 里（可单测）：精扫是"升级权"不是"否决权"，
            # 粗扫看到而精扫没确认的目标降级保留为 C 疑似，不再丢弃。
            keep_cs = bool(h.get("keep_coarse_suspect", True))
            hits, downgraded = merge_two_levels(
                c_hits, f_hits, c_diffs, c_fps, scanned, fb_thr, fine_on, keep_cs)
            stats["coarse_downgraded"] = downgraded
            stats["hits"] = len(hits)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)     # AUDIT-OK: 清理自创临时抽帧目录

        events = merge_events(
            hits, duration,
            gap_merge=float(h.get("gap_merge", 3.0)),
            min_duration=float(h.get("min_duration", 1.0)),
            pre_pad=float(h.get("pre_pad", 2.0)),
            post_pad=float(h.get("post_pad", 3.0)),
            max_clips=int(h.get("max_clips_per_video", 8)),
            grade_a_seconds=float(h.get("grade_a_seconds", 8.0)),
            grade_a_persons=int(h.get("grade_a_persons", 2)),
        )
        stats["events"] = len(events)
        stats["grades"] = {g: sum(1 for e in events if e["grade"] == g)
                           for g in ("A", "B", "C")}
        stats["secs"] = round(time.time() - t_start, 1)

        if not events:
            return {"ok": True, "id": vid, "stem": stem, "clips": [], "sec": 0.0,
                    "duration": duration, "year": year, "stats": stats,
                    "note": "无精华（画面确实没有值得留的内容）"}

        shot = parse_shot_time(stem)
        src_fps = info["fps"] or 0.0
        has_audio = bool(info["has_audio"])
        n_thumb_ab = int(h.get("thumb_per_clip", 1))
        n_thumb_c = int(h.get("c_thumbs", DEFAULT_C_THUMBS))
        cut_c = bool(h.get("c_cut_mp4", True))
        clips: List[Dict[str, Any]] = []

        for i, ev in enumerate(events, 1):
            grade = ev["grade"]
            # 片段名带真实时刻：文件名含拍摄时间戳，加上事件偏移即为绝对时间
            label = ((shot + timedelta(seconds=ev["start"])).strftime("%H%M%S")
                     if shot else f"{int(ev['start'] // 60):02d}{int(ev['start'] % 60):02d}00")
            tag = f"{stem}_E{i:02d}_{grade}_{label}_{int(round(ev['dur']))}s"
            out_dir = hdir / year / GRADE_DIRS[grade]
            out_dir.mkdir(parents=True, exist_ok=True)

            # 代表帧：C 级多留几张 —— 它可能**只**有图（没切视频）或没有更多上下文，
            # 一张图证明不了这 30 秒里发生过什么（Track C · C4）
            want_thumbs = n_thumb_c if grade == "C" else n_thumb_ab
            thumbs: List[str] = []
            if want_thumbs > 0:
                for k, tt in enumerate(self._thumb_times(ev, want_thumbs), 1):
                    tp = out_dir / "_thumbs" / f"{tag}{'' if k == 1 else f'_{k}'}.jpg"
                    if self._thumb(src, tt, tp):
                        thumbs.append(str(tp))
            thumb = thumbs[0] if thumbs else None

            # C 级也切可播放 mp4（Track C · C2）—— 静帧只能证明"某一秒有过东西"，
            # 而"删除后再也找不到"要的是能回放的整段。靠 c_cut_mp4 一键退回旧行为。
            mp4 = None
            if grade in ("A", "B") or (grade == "C" and cut_c):
                mp4 = out_dir / f"{tag}.mp4"
                if mp4.exists():            # 重跑时绝不覆盖，加 id 后缀
                    mp4 = out_dir / f"{tag}_{vid}.mp4"
                if not self._cut(src, ev, mp4, src_fps, has_audio):
                    if log:
                        log(f"[切段失败] {tag}")
                    mp4 = None

            if mp4 is None and not thumbs:
                continue

            clips.append({
                "index": i, "grade": grade,
                "start": ev["start"], "end": ev["end"], "dur": ev["dur"],
                "score": ev["score"], "conf": ev["conf"],
                "n_person": ev["n_person"], "level": ev["level"],
                "shot_at": (shot + timedelta(seconds=ev["start"])).isoformat()
                           if shot else None,
                "file": str(mp4) if mp4 else None,
                "thumb": thumb, "thumbs": thumbs,
                "ver": self.hl_version,
            })

        sec = sum(c["dur"] for c in clips if c["file"])
        stats["cut_secs"] = round(sec, 1)
        return {"ok": True, "id": vid, "stem": stem, "clips": clips, "sec": sec,
                "duration": duration, "src": str(src), "year": year,
                "kept_ratio": round(sec / duration, 3) if duration else 0,
                "stats": stats}

    # ------------------------------------------------------------------ 批量
    def run_batch(self, db, limit: int = 5000, progress: Progress = None,
                  log: Log = None, stop_event=None,
                  batch_id: Optional[int] = None,
                  rehighlight: bool = False,
                  only_dropped: bool = True) -> Dict[str, Any]:
        """剪辑所有待处理视频。返回统计。

        审核减负（ADR 0001 先剪后删）：剪辑产物留底后，自动对「未处置」的候选
        完成处置闭环 —— 精华占比过高自动保留原片；其余（含确认无精华）自动移入
        回收目录。人工不再需要逐条审核原片，只看 A/B 级精华产物即可。
        处置动作集中在 _dispose_one，任一步失败只记这一条，不拖垮整批。

        rehighlight=True 时改为跑**重剪队列**（Track C · C5）：只挑已剪过、
        产物口径版本落后的行。批量大小按"收益最集中优先"，
        only_dropped=True 即第一批（D2/D4 命中：零产物 + 只有静帧的），
        False 则全量重剪。幂等 —— 重剪完版本号更新，再跑就是空的。
        """
        problems = check_ready(self.cfg)
        if problems:
            raise HighlightError("；".join(problems))

        hdir = highlight_dir_of(self.cfg)
        hdir.mkdir(parents=True, exist_ok=True)
        if rehighlight:
            rows = db.rehighlight_queue(limit=limit, target_version=self.hl_version,
                                        only_dropped=only_dropped)
        else:
            rows = db.highlight_queue(limit=limit, batch_id=batch_id)
        total = len(rows)
        done = empty = failed = 0
        rescued = upgraded = 0
        n_clips = n_frames = 0
        n_sec = 0.0
        grades = {"A": 0, "B": 0, "C": 0}
        disposed = {"keep_full": 0, "undo_keep": 0, "moved": 0,
                    "not_disposed": 0}
        manifest = hdir / MANIFEST_NAME
        errors: List[str] = []

        with manifest.open("a", encoding="utf-8") as mf:
            for i, row in enumerate(rows, 1):
                if stop_event is not None and stop_event.is_set():
                    if log:
                        log("精华剪辑已中断")
                    break
                row = dict(row)      # sqlite3.Row → dict：后续统一用 .get()，
                                     # 否则 Row 没有 .get() 会 AttributeError
                vid = row["id"]
                ts = datetime.now().isoformat(timespec="seconds")
                try:
                    r = self.cut_one(row, hdir, ts, log=log)

                    if not r.get("ok"):
                        failed += 1
                        db.mark_highlight_error(vid, str(r.get("reason")))
                        errors.append(f"id={vid}: {r.get('reason')}")
                        if log:
                            log(f"[跳过] id={vid} {r.get('reason')}")
                        continue

                    clips = r.get("clips", [])
                    stats = r.get("stats", {}) or {}
                    if rehighlight:
                        # 捞回率：重剪前"零产物"的现在有产物了 / 原来只有静帧的现在有视频了
                        try:
                            prev = json.loads(row.get("highlight_clips") or "[]")
                        except (TypeError, ValueError):
                            prev = []
                        if clips and not prev:
                            rescued += 1
                        elif (clips and any(c.get("file") for c in clips)
                              and not any(c.get("file") for c in prev)):
                            upgraded += 1
                    db.mark_highlight(
                        r["id"], ts, len(clips), r.get("sec", 0.0), clips,
                        sent=stats.get("coarse_sent"),
                        coarse_hits=stats.get("coarse_hits"),
                        version=self.hl_version)
                    mf.write(json.dumps({
                        "id": r["id"], "stem": r.get("stem"), "year": r.get("year"),
                        "src": r.get("src"), "trash_path": row.get("trash_path"),
                        "duration_sec": r.get("duration"),
                        "kept_ratio": r.get("kept_ratio"),
                        "version": self.hl_version,
                        "stats": stats,
                        "clips": clips, "cut_at": ts,
                    }, ensure_ascii=False) + "\n")
                    mf.flush()

                    # —— 处置闭环：剪完留底后，机器替人工完成 keep/trash 决定 ——
                    act, note = self._dispose_one(db, row, r, log=log)
                    disposed[act] = disposed.get(act, 0) + 1
                    if note and log:
                        log(f"[处置] id={vid} {note}")
                except Exception as e:  # noqa: BLE001 单条失败不拖垮批次
                    # 写库/写 manifest/处置 都在 try 内：任一环节出错都只记这一条，
                    # 不能让一条坏数据炸掉整批（1372 段跑几小时，代价太高）
                    failed += 1
                    db.mark_highlight_error(vid, f"{type(e).__name__}: {e}")
                    errors.append(f"id={vid}: {type(e).__name__}: {e}")
                    if log:
                        log(f"[失败] id={vid} {type(e).__name__}: {e}")
                    continue

                for c in clips:
                    grades[c["grade"]] = grades.get(c["grade"], 0) + 1
                    if c["file"]:
                        n_clips += 1
                    if c["thumb"]:
                        n_frames += 1
                n_sec += r.get("sec", 0.0)
                if clips:
                    done += 1
                else:
                    empty += 1
                if progress:
                    progress(i, total, f"{r.get('stem')} · {len(clips)} 段")
                if log:
                    tail = f"{len(clips)} 段 / {r.get('sec', 0):.1f}s" if clips else "无精华"
                    log(f"已剪 [{i}/{total}] {r.get('stem')} → {tail}")

        return {"done": done, "empty": empty, "failed": failed, "total": total,
                "clips": n_clips, "frames": n_frames, "grades": grades,
                "secs": round(n_sec, 1),
                "rescued": rescued, "upgraded": upgraded,
                "rehighlight": bool(rehighlight),
                "version": self.hl_version,
                "disposed": disposed,
                "dir": str(hdir), "manifest": str(manifest), "errors": errors}

    def _dispose_one(self, db, row: Dict[str, Any], r: Dict[str, Any],
                     log: Log = None) -> tuple:
        """剪辑完成后的自动处置（审核减负 · 先剪后删）。

        只处理「机器该替人拍板」的情况，人工已表态的一律不碰：
        - human_decision IS NULL（未处置）：精华占比 > keep_full_ratio →
          自动保留原片；否则（含确认无精华）→ 自动移入回收目录。
        - human_decision='trash'（已移入回收，如 0 分桶批量放行）但剪出高占比 →
          机器从误删边缘捞回：撤销移动，改判保留。
        - human_decision='keep' 或剪辑失败的：不处置。

        返回 (action, note)：keep_full / undo_keep / moved / not_disposed / skip。
        """
        from .mover import decide_tier, move_one, undo_move
        h = self.h
        kr = float(r.get("kept_ratio") or 0.0)
        kfr = float(h.get("keep_full_ratio", 0.6))
        vid = row["id"]
        decision = row.get("human_decision")

        # 剪辑失败或机器判定为空但已处置过 —— 保持人工结论
        if not r.get("ok"):
            return "skip", ""
        # 人工已保留 → 无需处置（剪出来的产物只是额外保险）
        if decision == "keep":
            return "not_disposed", "人工已保留，产物留底不处置"

        auto = bool(self.cfg.get("highlight", {}).get("auto_dispose", True))
        keep_full = kr > kfr

        if decision == "trash" and keep_full:
            # 已放行但整片都有内容 —— 0 分桶批量放行也可能误杀，机器捞回
            ok, msg = undo_move(db, row, log)
            if ok:
                if log:
                    log(f"↩ 撤销移动并保留（精华占比 {kr:.0%} > {kfr:.0%}）：{msg}")
                return "undo_keep", f"精华占比 {kr:.0%} 过高，已撤销移动改判保留"
            return "not_disposed", f"想捞回但撤销失败：{msg}"

        if decision is None and not auto:
            return "not_disposed", "auto_dispose 关闭，留给人工处置"

        if decision is None:
            if keep_full:
                db.set_human_decision(vid, "keep")
                if log:
                    log(f"🟢 自动保留（精华占比 {kr:.0%} > {kfr:.0%}，整片都有内容）")
                return "keep_full", f"精华占比 {kr:.0%} 过高，自动整片保留"

            # —— 这里曾经有一道"有音轨未复核就不许移入"的守卫，已撤掉 ——
            # 守卫的前提是"音频复核队列只认未处置的视频"，而队列早已放宽为
            # 「处置候选」（见 db._PENDING_OR_TRASHED）：移入回收目录后依然能补复核
            # （audioreview.review_one 有 trash_path 回落），补完照常落成语音档案。
            # 所以音频走的是**事后补**这条线，不是"拦住"。两条路径（本函数与
            # mover.move_all）必须同口径 —— 半拦半不拦会让一键执行的终点不可预期
            # （ADR 0004 · Consequences）。
            # 唯一的代价：回收目录被**手动清空**之前必须把音频复核+归档跑完。
            # 命令式的东西靠顺序保证 —— 一键执行的链路里音频复核排在剪辑之前。

            # 剪完留底（含确认无精华）→ 移入回收目录，交由人工手动清空
            tier = decide_tier(row, self.cfg)
            ok, msg = move_one(db, row, self.cfg, tier, log)
            if ok:
                return "moved", "精华已留底，自动移入回收目录"
            return "not_disposed", f"想移入回收但失败：{msg}（产物已留底，可人工处理）"

        return "not_disposed", f"已处置为 {decision}，不重复处置"
