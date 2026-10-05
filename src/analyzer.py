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


def _class_stats(dets: List[Dict], grp: str, cfg: Dict,
                 n_frames: int) -> Dict[str, Any]:
    """某一类目标在**多帧**上的统计 —— 质量分公式的输入。

    现状的问题：`build_decision` 只对"单帧最高 conf"加权，而 `best[grp]`
    取的是 `max(conf)` —— 一个人只在 1 帧被检出、和 15 帧全被检出，得分**完全一样**。
    加上 `person_weight=50` 与 `review_threshold=30`，只要 conf ≥ 0.6 就直接 keep。

    这里统计六个量：
        frames   至少一帧检出（conf ≥ thr）的**帧数** —— 连续出现比闪现更可信
        persist  frames / n_frames —— 一致性系数。**分母用实际抽到的帧数**
                 （n_frames），不用配置值：短片或部分解码失败时按配置算会低估
        conf_top 单帧最高 conf（展示 + F2 保底用）
        conf_med 各命中帧 conf 的**中位数** —— 抗单帧误检的主力
        area_max 该类最大面积占比（远小判据）
        area_med 面积中位数（抗单帧大框误检）

    输入的 dets 必须带 `d["frame"]` 帧序号（analyze_one 里打的）。
    """
    thr = float(cfg["analysis"]["yolo_conf_threshold"])
    hits = []
    for d in dets:
        if _class_of(d["class"], cfg) != grp:
            continue
        if d["conf"] < thr:
            continue
        hits.append(d)
    total = max(1, int(n_frames or 0))
    if not hits:
        return {"frames": 0, "persist": 0.0, "conf_top": 0.0, "conf_med": 0.0,
                "area_max": 0.0, "area_med": 0.0}
    confs = sorted(float(d["conf"]) for d in hits)
    areas = sorted(float(d.get("area_ratio", 0.0)) for d in hits)
    mid = len(confs) // 2
    med_conf = confs[mid] if len(confs) % 2 else (confs[mid - 1] + confs[mid]) / 2
    mid_a = len(areas) // 2
    med_area = areas[mid_a] if len(areas) % 2 else (areas[mid_a - 1] + areas[mid_a]) / 2
    frames = len({int(d.get("frame", -1)) for d in hits})
    return {
        "frames": frames,
        "persist": min(1.0, frames / total),
        "conf_top": round(float(confs[-1]), 4),
        "conf_med": round(float(med_conf), 4),
        "area_max": round(float(areas[-1]), 5),
        "area_med": round(float(med_area), 5),
    }


def _score_quality_v2(dets_all: List[Dict], motion: float,
                      audio_energy: Optional[float], has_audio: bool,
                      fstats: Dict, cfg: Dict, yolo_on: bool,
                      n_frames: int, thr: float, raw_min: float,
                      ) -> Dict[str, Any]:
    """质量分公式（`scoring.mode = "quality_v2"`）。

    三步：**多帧一致性聚合 → noisy-OR 内容分 → 质量乘数**。

    为什么用乘数不用减法：内容分 0-100、扣分 0-45，减法会让
    `max(0, 小内容 − 大扣分) = 0` 直接落 trash —— 那等于"质量差否决内容"。
    乘数最低 0.5、永不清零，语义正确：**质量差只降低置信，不否决内容**。

    为什么用 noisy-OR 不用裸加法：裸加法可以无限累加（三个弱信号叠加就刷满），
    noisy-OR 天然饱和、逐项可解释、且天然落在 0-100。
    """
    sc = cfg.get("scoring", {})
    vd = cfg.get("verdict", {})
    cons = sc.get("consistency", {})
    size = sc.get("size", {})
    qual = sc.get("quality", {})
    cont = sc.get("content", {})

    # ---- Step 1: 多帧一致性 ----
    st = {g: _class_stats(dets_all, g, cfg, n_frames)
          for g in ("person", "vehicle", "animal")}

    # near_miss 仍按"单帧最高 conf"口径 —— decide_tier / release_queue /
    # feedback 四处消费者依赖它，一个字都不能改。公式 v2 只**新增**统计量。
    near: Dict[str, float] = {}
    for d in dets_all:
        grp = _class_of(d["class"], cfg)
        if grp == "other" or d["conf"] >= thr or d["conf"] < raw_min:
            continue
        near[grp] = max(near.get(grp, 0.0), float(d["conf"]))

    p_full = float(cons.get("persist_full", 0.4))
    a_ref = size.get("area_ref") or {}
    reasons: List[Dict[str, Any]] = []

    def _component(grp: str, label: str, w: float) -> float:
        s = st[grp]
        if s["frames"] == 0:
            return 0.0
        b_persist = (float(cons.get("persist_floor", 0.6))
                     + (1.0 - float(cons.get("persist_floor", 0.6)))
                     * min(1.0, s["persist"] / p_full))
        ref = float(a_ref.get(grp, 0.05)) or 0.05
        b_size = (float(size.get("size_floor", 0.5))
                  + (1.0 - float(size.get("size_floor", 0.5)))
                  * min(1.0, s["area_med"] / ref))
        c = float(w) * s["conf_med"] * b_persist * b_size
        reasons.append({
            "factor": label, "result": "检出",
            "contribution": round(c * 100, 1),
            "detail": (f"置信度中位数 {s['conf_med']:.2f}（峰值 {s['conf_top']:.2f}）、"
                       f"{s['frames']}/{n_frames} 帧出现、面积占比中位数 "
                       f"{s['area_med']*100:.1f}%（最大 {s['area_max']*100:.1f}%）"),
            "level": "keep",
        })
        return c

    c_person = _component("person", "人物", cont.get("w_person", 0.55))
    c_vehicle = _component("vehicle", "车辆", cont.get("w_vehicle", 0.45))
    c_animal = _component("animal", "动物", cont.get("w_animal", 0.20))
    for g, label in (("person", "人物"), ("vehicle", "车辆"), ("animal", "动物")):
        if st[g]["frames"] == 0:
            reasons.append({
                "factor": label, "result": "未检出", "contribution": 0,
                "detail": ("未检测到" + ("人物" if g == "person" else label)
                           if yolo_on else "未启用目标检测模型，该项未参与判定"),
                "level": "trash" if g != "animal" else "neutral",
            })
        elif g in near:
            reasons.append({
                "factor": label, "result": "另有未达阈值检出", "contribution": 0,
                "detail": f"另有置信度 {near[g]:.2f} 的疑似检出未达判定阈值 {thr:.2f}",
                "level": "warn",
            })

    # 动静 / 音频：只作为**弱信号**参与内容分（不再一次性 +10/+10）
    m_thr = float(sc.get("motion_area_threshold", 0.05))
    c_motion = float(cont.get("w_motion", 0.15)) if motion > m_thr else 0.0
    c_audio = 0.0
    if has_audio and audio_energy is not None:
        if audio_energy > float(sc.get("audio_db_threshold", -50.0)):
            c_audio = float(cont.get("w_audio", 0.15))
    reasons.append({
        "factor": "画面动静",
        "result": "显著" if motion > m_thr else "微弱",
        "contribution": round(c_motion * 100, 1),
        "detail": (f"画面变化区域占比 {motion*100:.1f}%（阈值 {m_thr*100:.0f}%）"
                   + ("" if motion > m_thr else "，符合环境干扰（树叶/光影）特征")),
        "level": "keep" if motion > m_thr else "trash",
    })
    reasons.append({
        "factor": "音频",
        "result": "有声音" if c_audio else ("接近静音" if has_audio else "无音轨"),
        "contribution": round(c_audio * 100, 1),
        "detail": (f"平均音量 {audio_energy:.1f} dB" if audio_energy is not None
                   else ("存在音轨但未做音量分析" if has_audio else "该摄像头不录音")),
        "level": "keep" if c_audio else "neutral",
    })

    # ---- Step 2: noisy-OR 饱和聚合 → 内容分 0-100 ----
    surv = 1.0
    for c in (c_person, c_vehicle, c_animal, c_motion, c_audio):
        if c > 0:
            surv *= (1.0 - min(0.99, c))
    content = round(100.0 * (1.0 - surv), 2)

    # ---- Step 3: 质量乘数 ----
    p = {}
    penalty = 0.0
    p["is_blurry"] = bool(fstats.get("is_blurry")) and float(
        qual.get("w_blurry", 25))
    p["is_low_light"] = bool(fstats.get("is_low_light")) and float(
        qual.get("w_low_light", 20))
    p["is_low_saturation"] = bool(fstats.get("is_low_saturation")) and float(
        qual.get("w_low_sat", 10))
    # 分辨率过低 = 远小目标的温床
    w_low_res = float(qual.get("w_low_res", 15))
    h = int(fstats.get("height") or 0)
    p["is_low_res"] = (0 < h < int(qual.get("min_height", 480))) and w_low_res
    p["is_tiny_person"] = (0 < st["person"]["area_max"] < float(
        qual.get("tiny_area", 0.01))) and float(qual.get("w_tiny_person", 10))
    p["is_single_frame_person"] = (st["person"]["frames"] == 1) and float(
        qual.get("w_single_frame", 10))
    cap = float(qual.get("penalty_cap", 45))
    penalty = min(cap, sum(v for v in p.values() if v))
    quality = round(100.0 - penalty, 2)
    mult = float(qual.get("mult_floor", 0.5)) + (
        1.0 - float(qual.get("mult_floor", 0.5))) * (quality / 100.0)
    score = round(content * mult, 2)

    # ---- 保底规则（Floor）与 Veto ----
    # Floor 命中 ⇒ verdict 至少 review，**绝不 trash**。
    # 它不制造积压：review 桶在 _PENDING_OR_TRASHED 的 B 分支 ⇒ 进 highlight_queue
    # ⇒ 剪完由 _dispose_one 按 kept_ratio 自动定去留。"保底"不等于"等人"。
    fl = qual.get("floor", {}) or {}
    tiny = float(qual.get("tiny_area", 0.01))
    # F0：**任何**一帧检出人物（conf ≥ 判定阈值）⇒ 至少 review，**绝不 trash**。
    #
    # 这条是全部 Floor 里最朴素也最硬的一条，理由只有一句：
    # **"看到了人"就不该判"这里没有价值"。** 尤其在糊/夜视画面里，
    # 远处一个小目标的检出更可能是真人而不是噪点 —— 这正是 L2.8 精华剪辑
    # （8 倍采样密度重扫）存在的意义。Veto 绝不能取消它，否则
    # "糊 + 远小 + 单帧"这个最典型的漏检场景会被判 trash 直接进回收目录。
    floor = {
        "F0_any_person": st["person"]["frames"] >= 1,
        "F1_multi_frame_person": (st["person"]["frames"] >= 2
                                  and st["person"]["conf_med"] >= float(
                                      fl.get("f1_conf_med", 0.5))),
        "F2_big_target": (st["person"]["conf_top"] >= float(fl.get("f2_conf", 0.6))
                          and st["person"]["area_max"] >= float(
                              fl.get("f2_area", 0.05))),
        "F3_big_motion": motion >= float(fl.get("f3_motion", 0.15)),
        "F4_weak_person": st["person"]["frames"] >= int(fl.get("f4_frames", 2)),
        "F5_majority_person": st["person"]["persist"] >= float(
            fl.get("f5_persist", 0.5)),
        "F6_big_vehicle": (st["vehicle"]["conf_top"] >= float(fl.get("f6_conf", 0.7))
                           and st["vehicle"]["area_max"] >= float(
                               fl.get("f6_area", 0.05))),
    }
    # Veto：**只**取消弱保底 F4/F5 —— 它们靠"多帧/过半帧"这种**间接**证据，
    # 而糊画面的帧间差异不可靠，多半是噪点抖动。
    # F0/F1/F2/F3/F6 仍有效，其中 F0 更是任何情况下都不取消（见上）。
    # 单向：只能取消弱保底，不能取消强保底。
    veto = bool(
        (fstats.get("is_blurry") and fstats.get("is_low_light"))
        or (fstats.get("is_low_light") and fstats.get("is_low_saturation"))
        or (fstats.get("is_blurry") and p.get("is_tiny_person"))
    )
    if veto:
        floor["F4_weak_person"] = False
        floor["F5_majority_person"] = False
    strong = any(floor[k] for k in ("F0_any_person", "F1_multi_frame_person",
                                    "F2_big_target", "F3_big_motion",
                                    "F6_big_vehicle"))

    # ---- 结论 ----
    if score <= vd["trash_threshold"]:
        verdict = "trash"
    elif score < vd["review_threshold"]:
        verdict = "review"
    else:
        verdict = "keep"
    if verdict == "trash" and (strong or any(floor.values())):
        verdict = "review"

    hit = [k for k, v in floor.items() if v]
    reasons.append({
        "factor": "质量与一致性", "result": "已评估", "contribution": 0,
        "detail": (f"内容分 {content:.1f} × 质量系数 {mult:.2f}（扣分 {penalty:.0f}："
                   f"{'、'.join(k for k, v in p.items() if v) or '无'}）"
                   + (f"；触发保底 {'、'.join(hit)}" if hit else "")
                   + ("（画面质量差，弱保底已取消）" if veto else "")),
        "level": "warn" if hit else "neutral",
    })

    summary = _summarize_v2(verdict, score, content, quality, st, floor, veto)
    return {
        "score": score,
        "verdict": verdict,
        "reasons": reasons,
        "summary": summary,
        "content_score": content,
        "quality_score": quality,
        "quality_penalty": round(penalty, 1),
        "quality_mult": round(mult, 3),
        "consistency": {g: st[g] for g in st},
        "floor": {**floor, "veto": veto, "strong": strong},
        "score_kind": "quality_v2",
        "l1_version": "qv2",
        "has_person": 1 if st["person"]["frames"] else 0,
        "person_conf": round(st["person"]["conf_top"], 4) if st["person"]["frames"] else (
            round(near.get("person", 0.0), 4) if "person" in near else 0.0),
        "has_vehicle": 1 if st["vehicle"]["frames"] else 0,
        "vehicle_conf": round(st["vehicle"]["conf_top"], 4) if st["vehicle"]["frames"] else 0.0,
        "has_animal": 1 if st["animal"]["frames"] else 0,
        "near_miss": {k: round(v, 4) for k, v in near.items()},
    }


def _summarize_v2(verdict, score, content, quality, st, floor, veto) -> str:
    """质量分口径的结论文案。与 _summarize 分开写，避免两套口径混在一段话里。"""
    hit = [k for k, v in floor.items() if v]
    tail = f"（保底规则 {'、'.join(hit)} 生效）" if hit else ""
    if verdict == "keep":
        what = [g for g in ("person", "vehicle", "animal") if st[g]["frames"]]
        return (f"内容分 {content:.1f} × 质量系数 → {score:.1f}，"
                f"稳定检出 {'、'.join(what) or '有效内容'}，判定保留{tail}。")
    if verdict == "review":
        return (f"内容分 {content:.1f}、质量 {quality:.1f} → {score:.1f}，"
                f"不足以判定保留，交给高密度重扫复核{tail}。")
    return (f"内容分 {content:.1f}、质量 {quality:.1f} → {score:.1f}，"
            f"画面无有效目标，判为环境干扰{('（' + ('糊且' if veto else '') + '弱信号已折价）' if veto else '')}。")


def build_decision(
    detections_all: List[Dict],
    motion: float,
    audio_energy: Optional[float],
    has_audio: bool,
    fstats: Dict,
    cfg: Dict,
    yolo_on: bool,
    n_frames: int = 0,
) -> Dict[str, Any]:
    """计算保留价值分，并生成完整的解释链。

    scoring.mode = "quality_v2"（2026-10-04 起）走**质量分**口径：
    多帧一致性 + noisy-OR 内容分 × 质量乘数 + 六条保底规则。
    改 "legacy" 即回退到下面的旧公式，**不需要 git revert、不需要重跑分析**。

    n_frames：实际抽到的帧数，是 `persist` 的分母。旧公式没有这个概念，
    但 v2 的一致性判据离开它就没有意义（见 _class_stats 的说明）。
    """
    mode = str(cfg.get("scoring", {}).get("mode", "legacy") or "legacy")
    if mode == "quality_v2":
        thr = cfg["analysis"]["yolo_conf_threshold"]
        raw_min = cfg["analysis"].get("yolo_keep_raw_above", 0.15)
        return _score_quality_v2(
            detections_all, motion, audio_energy, has_audio, fstats,
            cfg, yolo_on, n_frames, thr, raw_min,
        )
    return _build_decision_legacy(
        detections_all, motion, audio_energy, has_audio, fstats, cfg, yolo_on)


def _build_decision_legacy(
    detections_all: List[Dict],
    motion: float,
    audio_energy: Optional[float],
    has_audio: bool,
    fstats: Dict,
    cfg: Dict,
    yolo_on: bool,
) -> Dict[str, Any]:
    """**旧**公式（`scoring.mode = "legacy"`），原样保留作回滚出口。

    只接受"过阈的单帧最高 conf"：一个人只在 1 帧被检出、和 15 帧全被检出，
    得分完全一样；加上 person_weight=50 与 review_threshold=30，
    conf ≥ 0.6 就直接 keep。保留这段是为了让 `mode` 切换能立刻对比，
    不是为了继续用它。
    """
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
                for fi, f in enumerate(frames):
                    fin, raw = self.detector.detect(f)
                    # ★ 帧序号：质量分公式的 `persist`（一致性）与 `frames` 靠它。
                    # 旧公式不需要帧维度，所以这里以前完全没有。
                    for d in fin:
                        d["frame"] = fi
                    for d in raw:
                        d["frame"] = fi
                    final_dets.extend(fin)
                    all_dets.extend(raw)
                # ⚠️ 截断**只影响写库的 raw_detections**，不能影响统计。
                # 原来这里 `all_dets = all_dets[:60]` 之后才把 all_dets 交给
                # build_decision —— 一帧 5 个检出就能把后面所有帧挤掉，
                # 于是"多帧一致性"被系统性低估：单帧误检看起来像多帧确认。
                # 现在统计用**完整**的 all_dets_full，落库仍限流。
                all_dets_full = list(all_dets)
                all_dets.sort(key=lambda d: -d["conf"])
                all_dets = all_dets[:60]
            else:
                all_dets_full = []
            result["raw_detections"] = all_dets

            # 打分 + 解释。v2 需要完整检出集（多帧一致性）+ 实际帧数（persist 分母）
            dec = build_decision(
                all_dets_full if self.yolo_on else [],
                motion,
                row["audio_energy"] if row["audio_energy"] is not None else None,
                bool(row["has_audio"]),
                fstats, self.cfg, self.yolo_on,
                n_frames=len(frames),
            )
            result.update(
                retention_score=dec["score"], verdict=dec["verdict"],
                decision_trace=dec, has_person=dec["has_person"],
                person_conf=dec["person_conf"], has_vehicle=dec["has_vehicle"],
                vehicle_conf=dec["vehicle_conf"], has_animal=dec["has_animal"],
            )
            # v2 的新字段（legacy 模式下为 None，update_analysis 会跳过）
            for _k in ("l1_version", "content_score", "quality_score",
                       "person_frames", "person_persist", "person_area_max"):
                result[_k] = dec.get(_k)
            if dec.get("l1_version") is None:
                result.pop("l1_version", None)
            # final_score 与 retention_score 恒等，但它是**对外的判定结果**：
            # 12 处 SQL 写 COALESCE(final_score, retention_score, 0)，selftest
            # 也直接读它。保留是为了维持既有读取契约，不是维护第二套口径。
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
