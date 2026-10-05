"""
误判归因 —— 用户把「AI 判定要清理」的片段标记为「需要保留」时，自动分析判错原因

为什么能归因
------------
因为 analyzer 保存了**未过阈值的原始检出**（raw_detections）和画面特征（frame_stats）。
如果当初只存 has_person=0，那么"模型其实看到了人、只是置信度 0.38 没过 0.5 阈值"
这个信息就永久丢失了，事后根本无从分析。这是本模块存在的前提。

归因类别
--------
- no_yolo           未启用目标检测模型，判定只靠动静，必然不准
- low_confidence    模型看到了目标但置信度未达阈值（最可操作：调阈值）
- night_vision      夜视/低照度导致检测能力下降
- occlusion_blur    画面模糊/遮挡
- motion_only       有明显动静却无目标检出（远处/画外/未知类别）
- unknown           未能归类
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

REASON_LABELS = {
    "no_yolo": "未启用目标检测模型",
    "low_confidence": "目标置信度未达阈值",
    "night_vision": "夜视 / 低照度",
    "occlusion_blur": "画面模糊 / 遮挡",
    "motion_only": "有动静但无目标检出",
    "unknown": "原因不明",
}


def _loads(s: Optional[str], default):
    if not s:
        return default
    try:
        return json.loads(s)
    except Exception:  # noqa: BLE001
        return default


def analyze_misjudgment(row, cfg: Dict, yolo_on: bool = True) -> Dict[str, Any]:
    """分析单条误判。

    row: videos 表的一行（需含 raw_detections / frame_stats / decision_trace）
    返回: {category, label, detail, suggestion, evidence}
    """
    a = cfg.get("analysis", {})
    thr = float(a.get("yolo_conf_threshold", 0.5))
    raw_min = float(a.get("yolo_keep_raw_above", 0.15))

    raw: List[Dict] = _loads(row["raw_detections"], [])
    fstats: Dict = _loads(row["frame_stats"], {})
    trace: Dict = _loads(row["decision_trace"], {})

    classes = cfg.get("classes", {})
    interesting = set(classes.get("person", [])) | set(classes.get("vehicle", [])) \
        | set(classes.get("animal", []))

    motion = row["motion_area_ratio"] or 0.0
    score = trace.get("score", row["retention_score"] or 0)

    # ---------- 1. 未启用模型
    if not yolo_on:
        return {
            "category": "no_yolo", "label": REASON_LABELS["no_yolo"],
            "detail": "分析时未启用目标检测模型，判定只依赖画面动静，无法识别具体内容。",
            "suggestion": "下载 yolov8n.onnx 放到 models/ 目录后重新分析该片段。",
            "evidence": {"motion_area_ratio": motion, "score": score},
        }

    # ---------- 2. 未达阈值的目标检出（最具体、最可操作）
    near = [d for d in raw if d.get("class") in interesting and raw_min <= d["conf"] < thr]
    if near:
        near.sort(key=lambda d: -d["conf"])
        top = near[0]
        return {
            "category": "low_confidence", "label": REASON_LABELS["low_confidence"],
            "detail": (f"模型其实检测到了「{top['class']}」，置信度 {top['conf']:.2f}，"
                       f"但未达到判定阈值 {thr:.2f}，因此被判为无目标。"
                       f"共 {len(near)} 个未达阈值的疑似目标。"),
            "suggestion": (f"可将 yolo_conf_threshold 下调至 {max(0.15, top['conf'] - 0.05):.2f} 附近，"
                           f"或对置信度处于 {raw_min:.2f}~{thr:.2f} 区间的片段强制人工复核。"
                           f"注意：下调阈值会提高误留率（把垃圾判成有用）。"),
            "evidence": {
                "near_misses": [{"class": d["class"], "conf": d["conf"],
                                 "area_ratio": d.get("area_ratio")} for d in near[:5]],
                "threshold": thr, "motion_area_ratio": motion,
            },
        }

    # ---------- 3. 夜视 / 低照度
    if fstats.get("is_low_light") or fstats.get("is_low_saturation"):
        return {
            "category": "night_vision", "label": REASON_LABELS["night_vision"],
            "detail": (f"画面亮度 {fstats.get('brightness', 0):.1f}、"
                       f"饱和度 {fstats.get('saturation', 0):.1f}，"
                       f"属于夜视/红外场景，模型在低照度下检测能力显著下降。"),
            "suggestion": ("对 is_low_light 的片段单独降低判定阈值，或直接标记为"
                           "「必须人工审核」—— 夜间是漏检高发场景。"),
            "evidence": {"frame_stats": fstats, "motion_area_ratio": motion},
        }

    # ---------- 4. 模糊
    if fstats.get("is_blurry"):
        return {
            "category": "occlusion_blur", "label": REASON_LABELS["occlusion_blur"],
            "detail": f"画面清晰度指标 blur_var={fstats.get('blur_var', 0):.1f} 偏低，存在模糊或遮挡。",
            "suggestion": "模糊片段建议提高抽帧数与分辨率，或直接人工审核，不宜依赖自动判定。",
            "evidence": {"frame_stats": fstats},
        }

    # ---------- 5. 有动静但完全没检出
    if motion >= 0.05:
        return {
            "category": "motion_only", "label": REASON_LABELS["motion_only"],
            "detail": (f"画面变化区域达 {motion*100:.1f}%，说明确实有东西在动，"
                       f"但模型未识别为人物/车辆/动物。可能是远处目标、画外目标、"
                       f"或模型不认识的类别（如大型物体、包裹、设备）。"),
            "suggestion": ("若这类误判频繁，说明存在自定义关注类别，"
                           "建议在 config.json 的 classes 中补充，或对该场景训练专用模型。"),
            "evidence": {"motion_area_ratio": motion, "raw_detections": raw[:5]},
        }

    # ---------- 6. 未知
    return {
        "category": "unknown", "label": REASON_LABELS["unknown"],
        "detail": "未匹配到已知误判模式，需要人工查看画面后补充归类。",
        "suggestion": "记录该样本，积累到一定数量后再分析是否存在新的误判模式。",
        "evidence": {"score": score, "motion_area_ratio": motion,
                     "frame_stats": fstats, "raw_detections": raw[:5]},
    }


def record_feedback(db, row, human_verdict: str, cfg: Dict,
                    yolo_on: bool = True, note: str = "") -> Dict[str, Any]:
    """记录一条人工反馈并归因。human_verdict: keep | trash"""
    ai_verdict = row["verdict"] or "trash"
    if human_verdict == ai_verdict:
        # 人与 AI 一致，不是误判，只记录处置
        db.set_human_decision(row["id"], human_verdict, decided_by="human")
        return {"is_misjudgment": False, "category": "agreed"}

    db.set_human_decision(row["id"], human_verdict, decided_by="human")

    # 只有"AI 判清理、人判保留"才是危险误判，需要归因
    if ai_verdict == "trash" and human_verdict == "keep":
        r = analyze_misjudgment(row, cfg, yolo_on)
        if note:
            r["detail"] = f"{r['detail']}（用户备注：{note}）"
        db.add_feedback(row["id"], ai_verdict, human_verdict,
                        r["category"], r["detail"], r["suggestion"], r["evidence"])
        return {"is_misjudgment": True, **r}

    # 反向误判：AI 判保留、人判清理 —— 说明清理不够干净，也记录
    db.add_feedback(
        row["id"], ai_verdict, human_verdict, "false_keep",
        f"AI 判定为 {ai_verdict}（得分 {row['retention_score']}），但人工认为应当清理。",
        "若此类误判增多，说明阈值偏保守，可上调 trash_threshold 以提高清理率。",
        {"score": row["retention_score"], "note": note},
    )
    return {"is_misjudgment": True, "category": "false_keep",
            "label": "误留（清理不够干净）",
            "detail": f"AI 判定保留（得分 {row['retention_score']}），人工认为应清理。",
            "suggestion": "若此类增多，说明阈值偏保守，可考虑上调判定阈值。",
            "evidence": {}}


def summarize_feedback(db) -> Dict[str, Any]:
    """汇总所有误判反馈，给出整体优化建议 —— 把人工审核变成可调参的依据。"""
    rows = db.list_feedback(limit=5000)
    total = len(rows)
    if not total:
        return {"total": 0, "distribution": [], "suggestions": []}

    dist: Dict[str, int] = {}
    for r in rows:
        dist[r["reason_category"]] = dist.get(r["reason_category"], 0) + 1

    distribution = sorted(
        [{"category": k, "label": REASON_LABELS.get(k, k), "count": v,
          "ratio": round(v / total * 100, 1)} for k, v in dist.items()],
        key=lambda x: -x["count"],
    )

    suggestions: List[str] = []
    top = distribution[0]

    # 针对"未达阈值"给出具体阈值建议
    lc = [r for r in rows if r["reason_category"] == "low_confidence"]
    if lc:
        confs: List[float] = []
        for r in lc:
            try:
                ev = json.loads(r["evidence"] or "{}")
                for nm in ev.get("near_misses", [])[:3]:
                    confs.append(float(nm["conf"]))
            except Exception:  # noqa: BLE001
                pass
        if confs:
            confs.sort()
            p10 = confs[max(0, int(len(confs) * 0.1))]
            suggestions.append(
                f"有 {len(lc)} 例因置信度未达阈值而漏检（占 {len(lc)/total*100:.0f}%），"
                f"这些样本的目标置信度集中在 {min(confs):.2f}~{max(confs):.2f}。"
                f"建议把 yolo_conf_threshold 下调到 {max(0.15, round(p10 - 0.05, 2))} 附近，"
                f"代价是误留率上升（更多垃圾需要人工过目）。"
            )

    if dist.get("night_vision", 0) >= max(3, total * 0.2):
        suggestions.append(
            f"夜视场景误判 {dist['night_vision']} 例（占 {dist['night_vision']/total*100:.0f}%）。"
            f"建议对 is_low_light 的片段单独降低阈值，或直接标记为「必须人工审核」。"
        )
    if dist.get("no_yolo", 0) > 0:
        suggestions.append("存在因未启用目标检测模型导致的误判，建议下载 yolov8n.onnx 后重跑。")
    if dist.get("motion_only", 0) >= max(3, total * 0.15):
        suggestions.append(
            f"「有动静但无目标」误判 {dist['motion_only']} 例。"
            f"建议检查是否存在未被 classes 覆盖的关注类别，补充后重跑。"
        )
    if dist.get("unknown", 0) >= max(5, total * 0.3):
        suggestions.append("未归类误判占比偏高，说明可能存在新的误判模式，建议抽样人工查看后补充规则。")

    if not suggestions:
        suggestions.append(f"当前误判以「{top['label']}」为主，样本量还较少，建议继续积累后再调整阈值。")

    return {"total": total, "distribution": distribution, "suggestions": suggestions}
