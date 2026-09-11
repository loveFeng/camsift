"""
分析引擎（L1）—— 抽帧 + 动静检测 + YOLO 目标检测 + 打分 + 生成判定解释

设计要点
--------
1. **保存未过阈值的原始检出**（raw_detections）
   只存 has_person=0 的话，事后永远无法知道"模型其实看到了人、只是置信度 0.38
   没过 0.5 阈值"。误判归因完全依赖这份数据，是本项目的关键设计。

2. **判定解释链**（decision_trace）
   每一项打分的来源、数值、是否触发阈值都记录成结构化原因，
   UI 据此回答"为什么判定要删/要留"。

3. **解不开的片段默认保留**
   私有编码（H.265+/Smart H.265）解不开时绝不判为垃圾 —— 不能因技术失败而误删。

4. **无模型优雅降级**
   没有 ONNX 模型时仍可运行，退化为"动静 + 元数据"模式，只是分数不含目标检测项。
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import cv2
import numpy as np

from .config import COCO_CLASSES, THUMB_DIR, bin_path


# ---------------------------------------------------------------------------- 抽帧
def extract_frames(
    path: Path,
    n_frames: int,
    out_dir: Path,
    ffmpeg: str = "ffmpeg",
    width: int = 640,
    duration: float | None = None,
    timeout: int = 180,
    hwaccel: str = "auto",
) -> List[Path]:
    """用 ffmpeg 均匀抽出 n_frames 帧。

    优先按 duration 计算 fps 均匀抽样；duration 未知时退化为每 5 秒一帧。
    hwaccel 让 ffmpeg 自行选择（有 N 卡时走 CUVID，没有则回落软解）。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(out_dir / "f_%03d.jpg")
    for f in out_dir.glob("f_*.jpg"):
        f.unlink()

    if duration and duration > 0:
        fps = max(0.02, min(30.0, n_frames / max(duration, 0.1)))
        vf = f"fps={fps:.4f},scale={width}:-2"
    else:
        vf = f"fps=0.2,scale={width}:-2"

    cmd = [ffmpeg, "-v", "error"]
    if hwaccel and hwaccel != "none":
        cmd += ["-hwaccel", hwaccel]
    cmd += ["-i", str(path), "-vf", vf, "-frames:v", str(n_frames), "-q:v", "3", pattern]

    subprocess.run(cmd, capture_output=True, timeout=timeout)
    return sorted(out_dir.glob("f_*.jpg"))


# ---------------------------------------------------------------------------- 动静
def motion_ratio(frames: List[np.ndarray]) -> float:
    """稀疏帧之间的最大变化面积占比。

    抽帧稀疏（非逐帧）时用帧差比 MOG2 更合适 —— MOG2 需要连续帧才能建立背景模型。
    """
    if len(frames) < 2:
        return 0.0
    best = 0.0
    prev = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    prev = cv2.GaussianBlur(prev, (5, 5), 0)
    h, w = prev.shape
    total = float(h * w)
    for f in frames[1:]:
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        g = cv2.GaussianBlur(g, (5, 5), 0)
        diff = cv2.absdiff(prev, g)
        _, mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        best = max(best, float(mask.sum()) / 255.0 / total)
        prev = g
    return round(best, 4)


# ---------------------------------------------------------------------------- 画面特征
def frame_stats(frames: List[np.ndarray]) -> Dict[str, Any]:
    """刻画画面特征，供误判归因使用（夜视/红外/分辨率/模糊）。"""
    if not frames:
        return {}
    f0 = frames[0]
    h, w = f0.shape[:2]
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]

    brightness = float(np.mean([g.mean() for g in grays]))
    contrast = float(np.mean([g.std() for g in grays]))
    # 拉普拉斯方差：越小越模糊
    blur = float(np.mean([cv2.Laplacian(g, cv2.CV_64F).var() for g in grays]))

    # 饱和度低 + 亮度异常 → 疑似红外/夜视（红外常为灰度或单色调）
    hsv = cv2.cvtColor(f0, cv2.COLOR_BGR2HSV)
    saturation = float(hsv[:, :, 1].mean())

    return {
        "width": w, "height": h,
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "blur_var": round(blur, 2),
        "saturation": round(saturation, 2),
        "is_low_light": bool(brightness < 45),
        "is_low_saturation": bool(saturation < 30),
        "is_blurry": bool(blur < 100),
    }


# ---------------------------------------------------------------------------- YOLO
class YoloDetector:
    """用 OpenCV DNN 加载 YOLOv8 ONNX —— 不依赖 torch / ultralytics，模型仅 ~12MB。"""

    def __init__(self, model_path: str, input_size: int = 640,
                 conf_threshold: float = 0.5, nms_threshold: float = 0.45):
        self.model_path = Path(model_path)
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self._net = None

    @property
    def available(self) -> bool:
        return self.model_path.exists()

    @property
    def net(self):
        if self._net is None:
            if not self.available:
                raise FileNotFoundError(f"模型文件不存在：{self.model_path}")
            self._net = cv2.dnn.readNetFromONNX(str(self.model_path))
        return self._net

    def detect(self, frame: np.ndarray, raw_min: float = 0.15):
        """返回 (过阈值检出, 全部原始检出)。

        全部原始检出包含低于阈值但高于 raw_min 的项 —— 误判归因的关键数据。
        """
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (self.input_size, self.input_size),
                                     swapRB=True, crop=False)
        self.net.setInput(blob)
        out = self.net.forward()  # (1, 84, 8400)
        preds = out[0].T  # (8400, 84)

        sx, sy = w / self.input_size, h / self.input_size
        boxes, scores, class_ids = [], [], []
        for row in preds:
            cx, cy, bw, bh = row[:4]
            cls_scores = row[4:]
            cid = int(np.argmax(cls_scores))
            conf = float(cls_scores[cid])
            if conf < raw_min:
                continue
            x1 = max(0.0, (cx - bw / 2) * sx)
            y1 = max(0.0, (cy - bh / 2) * sy)
            x2 = min(float(w), (cx + bw / 2) * sx)
            y2 = min(float(h), (cy + bh / 2) * sy)
            boxes.append([x1, y1, x2 - x1, y2 - y1])
            scores.append(conf)
            class_ids.append(cid)

        raw = [
            {
                "class": COCO_CLASSES[c] if 0 <= c < len(COCO_CLASSES) else str(c),
                "class_id": int(c),
                "conf": round(float(s), 4),
                "box": [round(float(v), 1) for v in b],
                "area_ratio": round(float(b[2]) * float(b[3]) / float(w * h), 5),
            }
            for b, s, c in zip(boxes, scores, class_ids)
        ]
        if not boxes:
            return [], raw

        keep_ids = cv2.dnn.NMSBoxes(boxes, scores, self.conf_threshold, self.nms_threshold)
        keep: List[int] = []
        if len(keep_ids) > 0:
            keep = [int(i) for i in np.array(keep_ids).flatten()]
        final = [raw[i] for i in keep]
        return final, raw


# ---------------------------------------------------------------------------- 打分与解释
def _class_of(name: str, cfg: Dict) -> str:
    classes = cfg.get("classes", {})
    for group, names in classes.items():
        if name in names:
            return group
    return "other"


def build_decision(
    detections_all: List[Dict],
    motion: float,
    audio_energy: Optional[float],
    has_audio: bool,
    fstats: Dict,
    cfg: Dict,
    yolo_on: bool,
) -> Dict[str, Any]:
    """计算保留价值分，并生成完整的解释链。"""
    sc = cfg.get("scoring", {})
    vd = cfg.get("verdict", {})
    thr = cfg["analysis"]["yolo_conf_threshold"]
    raw_min = cfg["analysis"].get("yolo_keep_raw_above", 0.15)

    # 按帧聚合：取每个类别的最高置信度
    best: Dict[str, Dict] = {}
    for d in detections_all:
        grp = _class_of(d["class"], cfg)
        if grp == "other":
            continue
        if d["conf"] >= thr:
            if grp not in best or d["conf"] > best[grp]["conf"]:
                best[grp] = d

    # 同样统计"未达阈值但值得一提"的检出
    near: Dict[str, Dict] = {}
    for d in detections_all:
        grp = _class_of(d["class"], cfg)
        if grp == "other" or d["conf"] >= thr:
            continue
        if d["conf"] < raw_min:
            continue
        if grp not in near or d["conf"] > near[grp]["conf"]:
            near[grp] = d

    reasons: List[Dict[str, Any]] = []
    score = 0.0

    # ---- 人物（权重最高）
    if "person" in best:
        d = best["person"]
        add = sc["person_weight"] * d["conf"]
        score += add
        reasons.append({
            "factor": "人物", "result": "检出", "contribution": round(add, 1),
            "detail": f"检出人物，置信度 {d['conf']:.2f}，画面占比 {d['area_ratio']*100:.1f}%",
            "level": "keep",
        })
    elif "person" in near:
        d = near["person"]
        reasons.append({
            "factor": "人物", "result": "疑似未达阈值", "contribution": 0,
            "detail": (f"存在疑似人物但置信度仅 {d['conf']:.2f}，未达判定阈值 {thr:.2f}。"
                       f"这是误判高发区，建议人工确认"),
            "level": "warn",
        })
    else:
        reasons.append({
            "factor": "人物", "result": "未检出", "contribution": 0,
            "detail": ("未检测到人物" if yolo_on else "未启用目标检测模型，该项未参与判定"),
            "level": "trash",
        })

    # ---- 车辆
    if "vehicle" in best:
        d = best["vehicle"]
        add = sc["vehicle_weight"] * d["conf"]
        score += add
        reasons.append({
            "factor": "车辆", "result": "检出", "contribution": round(add, 1),
            "detail": f"检出 {d['class']}，置信度 {d['conf']:.2f}，画面占比 {d['area_ratio']*100:.1f}%",
            "level": "keep",
        })
    elif "vehicle" in near:
        d = near["vehicle"]
        reasons.append({
            "factor": "车辆", "result": "疑似未达阈值", "contribution": 0,
            "detail": f"存在疑似 {d['class']} 但置信度仅 {d['conf']:.2f}，未达阈值 {thr:.2f}",
            "level": "warn",
        })
    else:
        reasons.append({
            "factor": "车辆", "result": "未检出", "contribution": 0,
            "detail": "未检测到车辆" if yolo_on else "未启用目标检测模型",
            "level": "trash",
        })

    # ---- 动物
    if "animal" in best:
        d = best["animal"]
        add = sc["animal_weight"] * d["conf"]
        score += add
        reasons.append({
            "factor": "动物", "result": "检出", "contribution": round(add, 1),
            "detail": f"检出 {d['class']}，置信度 {d['conf']:.2f}",
            "level": "keep",
        })
    else:
        reasons.append({
            "factor": "动物", "result": "未检出", "contribution": 0,
            "detail": "未检测到动物", "level": "neutral",
        })

    # ---- 动静
    m_thr = sc["motion_area_threshold"]
    if motion > m_thr:
        score += sc["motion_area_weight"]
        reasons.append({
            "factor": "画面动静", "result": "显著", "contribution": sc["motion_area_weight"],
            "detail": f"画面变化区域占比 {motion*100:.1f}%，超过阈值 {m_thr*100:.0f}%",
            "level": "keep",
        })
    else:
        reasons.append({
            "factor": "画面动静", "result": "微弱", "contribution": 0,
            "detail": (f"画面变化区域占比仅 {motion*100:.1f}%，低于阈值 {m_thr*100:.0f}%，"
                       f"符合环境干扰（树叶/光影）特征"),
            "level": "trash",
        })

    # ---- 音频
    if has_audio and audio_energy is not None:
        # audio_energy 是 dB（负值），越接近 0 越响
        if audio_energy > float(sc.get("audio_db_threshold", -50.0)):
            score += sc["audio_weight"]
            reasons.append({
                "factor": "音频", "result": "有声音", "contribution": sc["audio_weight"],
                "detail": f"平均音量 {audio_energy:.1f} dB，检测到声音活动",
                "level": "keep",
            })
        else:
            reasons.append({
                "factor": "音频", "result": "接近静音", "contribution": 0,
                "detail": f"平均音量 {audio_energy:.1f} dB，基本无声",
                "level": "neutral",
            })
    elif has_audio:
        reasons.append({
            "factor": "音频", "result": "未分析", "contribution": 0,
            "detail": "存在音轨但未做音量分析（可在扫描时开启）", "level": "neutral",
        })
    else:
        reasons.append({
            "factor": "音频", "result": "无音轨", "contribution": 0,
            "detail": "该摄像头不录音，音频维度不参与判定", "level": "neutral",
        })

    score = round(score, 2)

    # ---- 结论
    if score <= vd["trash_threshold"]:
        verdict = "trash"
    elif score < vd["review_threshold"]:
        verdict = "review"
    else:
        verdict = "keep"

    summary = _summarize(verdict, score, best, near, motion, fstats)

    return {
        "score": score,
        "verdict": verdict,
        "reasons": reasons,
        "summary": summary,
        "has_person": 1 if "person" in best else 0,
        "person_conf": round(best["person"]["conf"], 4) if "person" in best else (
            round(near["person"]["conf"], 4) if "person" in near else 0.0),
        "has_vehicle": 1 if "vehicle" in best else 0,
        "vehicle_conf": round(best["vehicle"]["conf"], 4) if "vehicle" in best else 0.0,
        "has_animal": 1 if "animal" in best else 0,
        "near_miss": {k: round(v["conf"], 4) for k, v in near.items()},
    }


def _summarize(verdict, score, best, near, motion, fstats) -> str:
    if verdict == "keep":
        what = []
        if "person" in best:
            what.append("人物")
        if "vehicle" in best:
            what.append("车辆")
        if "animal" in best:
            what.append("动物")
        return f"画面检出{'、'.join(what) or '有效内容'}，判定保留（得分 {score}）。"
    if verdict == "review":
        hint = ""
        if near:
            hint = f"存在未达阈值的疑似目标（{'、'.join(near.keys())}），建议人工确认。"
        return f"自动判定不明确（得分 {score}）。{hint}"
    # trash
    extra = []
    if fstats.get("is_low_light"):
        extra.append("画面偏暗（疑似夜间/红外）")
    if fstats.get("is_blurry"):
        extra.append("画面偏模糊")
    tail = f"，且{'、'.join(extra)}，请留意漏检可能" if extra else ""
    return (f"未检出人物或车辆，画面变化区域仅 {motion*100:.1f}%，"
            f"判定为环境干扰导致的误触发录制{tail}。")


# ---------------------------------------------------------------------------- 缩略图
def save_thumbnail(frames: List[np.ndarray], video_id: int,
                   cfg: Dict, detections: List[Dict] | None = None) -> Path | None:
    """把抽出的帧拼成网格图存到本地 SSD，供 UI 快速浏览（不碰源盘）。"""
    if not frames:
        return None
    a = cfg.get("analysis", {})
    cols = int(a.get("grid_cols", 3))
    tw = int(a.get("thumb_width", 320))
    rows = (len(frames) + cols - 1) // cols

    th = int(frames[0].shape[0] * tw / max(frames[0].shape[1], 1))
    tiles = []
    for f in frames:
        im = f.copy()
        if detections:
            im = _draw(im, detections, f.shape)
        tiles.append(cv2.resize(im, (tw, th)))

    # 补齐空格
    while len(tiles) < rows * cols:
        tiles.append(np.full((th, tw, 3), 245, dtype=np.uint8))

    grid_rows = [np.hstack(tiles[r * cols:(r + 1) * cols]) for r in range(rows)]
    grid = np.vstack(grid_rows)

    THUMB_DIR.mkdir(parents=True, exist_ok=True)
    out = THUMB_DIR / f"{video_id}_grid.jpg"
    cv2.imwrite(str(out), grid, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return out


def _draw(img: np.ndarray, dets: List[Dict], shape) -> np.ndarray:
    """在缩略图上画检出框，让人一眼看出模型看到了什么。"""
    for d in dets:
        if d["conf"] < 0.3:
            continue
        x, y, w, h = [int(v) for v in d["box"]]
        label = f"{d['class']} {d['conf']:.2f}"
        color = (60, 160, 60) if d["conf"] >= 0.5 else (60, 140, 220)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label, (x, max(y - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, color, 1, cv2.LINE_AA)
    return img


# ---------------------------------------------------------------------------- 主流程
class Analyzer:
    def __init__(self, db, cfg: Dict):
        self.db = db
        self.cfg = cfg
        a = cfg.get("analysis", {})
        self.n_frames = int(a.get("sample_frames", 6))
        self.detector = YoloDetector(
            a.get("yolo_model_path", ""),
            int(a.get("yolo_input_size", 640)),
            float(a.get("yolo_conf_threshold", 0.5)),
            float(a.get("nms_threshold", 0.45)),
        )
        self.ffmpeg = bin_path(cfg, "ffmpeg_path", "ffmpeg") or "ffmpeg"
        self.yolo_on = bool(a.get("yolo_enabled", True)) and self.detector.available

    def analyze_one(self, row, keep_thumb: bool = True) -> Dict[str, Any]:
        path = Path(row["path"])
        video_id = row["id"]
        result: Dict[str, Any] = {
            "status": "analyzed", "sampled_frames": 0,
            "has_person": 0, "person_conf": 0.0, "has_vehicle": 0,
            "vehicle_conf": 0.0, "has_animal": 0, "motion_area_ratio": 0.0,
            "raw_detections": [], "frame_stats": {},
            "retention_score": 0.0, "verdict": "review",
            "decision_trace": {}, "analyze_error": None,
        }

        tmp = Path(tempfile.mkdtemp(prefix="vs_"))
        try:
            if not path.exists():
                result.update(status="error", analyze_error="文件不存在")
                return result

            files = extract_frames(
                path, self.n_frames, tmp, self.ffmpeg,
                duration=row["duration_sec"],
                hwaccel=self.cfg.get("analysis", {}).get("decode_backend", "auto"),
            )
            if not files:
                # 解不开 —— 默认保留，绝不因技术失败判为垃圾
                result.update(
                    status="error",
                    analyze_error="无法解码（可能是 H.265+ 私有编码或文件损坏）",
                    verdict="keep", retention_score=100.0,
                    decision_trace={"score": 100.0, "verdict": "keep", "reasons": [{
                        "factor": "解码", "result": "失败", "contribution": 100,
                        "detail": "无法解码该片段，出于安全考虑一律保留，绝不因技术失败而误删",
                        "level": "warn"}],
                        "summary": "无法解码，出于安全一律保留。"},
                )
                return result

            frames = [cv2.imread(str(f)) for f in files]
            frames = [f for f in frames if f is not None]
            if not frames:
                result.update(status="error", analyze_error="抽帧后无法读取图像",
                              verdict="keep", retention_score=100.0)
                return result

            result["sampled_frames"] = len(frames)

            # 动静 + 画面特征
            motion = motion_ratio(frames)
            fstats = frame_stats(frames)
            result["motion_area_ratio"] = motion
            result["frame_stats"] = fstats

            # 目标检测
            all_dets: List[Dict] = []
            final_dets: List[Dict] = []
            if self.yolo_on:
                for f in frames:
                    fin, raw = self.detector.detect(f)
                    final_dets.extend(fin)
                    all_dets.extend(raw)
                # 原始检出去重限流，避免 JSON 过大
                all_dets.sort(key=lambda d: -d["conf"])
                all_dets = all_dets[:60]
            result["raw_detections"] = all_dets

            # 打分 + 解释
            dec = build_decision(
                final_dets if self.yolo_on else [],
                motion,
                row["audio_energy"] if row["audio_energy"] is not None else None,
                bool(row["has_audio"]),
                fstats, self.cfg, self.yolo_on,
            )
            result.update(
                retention_score=dec["score"], verdict=dec["verdict"],
                decision_trace=dec, has_person=dec["has_person"],
                person_conf=dec["person_conf"], has_vehicle=dec["has_vehicle"],
                vehicle_conf=dec["vehicle_conf"], has_animal=dec["has_animal"],
            )
            result["final_score"] = dec["score"]

            if keep_thumb:
                save_thumbnail(frames, video_id, self.cfg, final_dets[:12])

        except subprocess.TimeoutExpired:
            result.update(status="error", analyze_error="解码超时",
                          verdict="keep", retention_score=100.0)
        except Exception as e:  # noqa: BLE001
            result.update(status="error", analyze_error=f"{type(e).__name__}: {e}",
                          verdict="keep", retention_score=100.0)
        finally:
            # 仅清理本函数自己用 tempfile 创建的临时抽帧目录，从不触碰用户视频源文件。
            # 程序对源数据只有「读取」和「分级移动」两种操作，永不删除。
            shutil.rmtree(tmp, ignore_errors=True)  # AUDIT-OK: 仅清理自创临时目录

        return result

    def run(self, batch_id: int | None = None, limit: int | None = None,
            progress: Callable[[int, int, str], None] | None = None,
            log: Callable[[str], None] | None = None,
            stop_event=None, refresh_every: int = 10) -> Dict:
        """批量分析。

        stop_event: threading.Event，置位后在当前文件处理完立即退出（不丢已完成的写入）。
        refresh_every: 每处理 N 条回写一次批次进度 —— 进度条的可见性来源。
        """
        t0 = time.time()
        done = err = processed = 0
        stopped = False
        self.db.refresh_batch_progress(batch_id)

        while True:
            if stop_event is not None and stop_event.is_set():
                stopped = True
                break
            remain = None if limit is None else max(0, limit - done)
            if remain == 0:
                break
            rows = self.db.iter_pending_analysis(batch_id, limit=min(remain or 200, 200))
            if not rows:
                break
            for row in rows:
                if stop_event is not None and stop_event.is_set():
                    stopped = True
                    break
                r = self.analyze_one(row)
                self.db.update_analysis(row["id"], r)
                if r["status"] == "error":
                    err += 1
                    if log:
                        log(f"[分析失败] {row['filename']}: {r.get('analyze_error')}")
                else:
                    done += 1
                processed += 1
                if processed % max(1, refresh_every) == 0:
                    self.db.refresh_batch_progress(batch_id)
                    if progress:
                        progress(done + err, done + err, row["filename"])
            if stopped:
                break

        self.db.refresh_batch_progress(batch_id)
        if log:
            tail = "（已手动停止）" if stopped else ""
            log(f"分析完成：成功 {done}，失败 {err}（失败项一律保留），"
                f"用时 {time.time() - t0:.1f}s{tail}")
        return {"analyzed": done, "errors": err, "stopped": stopped}
