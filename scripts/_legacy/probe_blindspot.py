"""盲区补检对照实验（一次性诊断，只读，不改生产数据）。

背景：精华剪辑的「帧差预筛」把画面没变化的帧整段跳过、不送检测。对全片静止的
片段，YOLO 一帧都没跑就判了「无精华」——之后还会自动移入回收目录。本脚本挑出
这批片段，按指定 fps/分辨率**真的送检一遍**，看捞回多少，用来判断"提高检测标准"
值不值。

抽帧走 ffmpeg 管道直读帧（不落盘）：一是不留临时文件，二是避免批量删除保护
（每视频清一次抽帧目录会触发 SAFE_DELETE_BULK_CONFIRM_REQUIRED）。

⚠️ 已知口径差异（重要）：生产链路的 `_extract` 落 JPEG 再 cv2.imread，本脚本走
rawvideo 管道。同一帧两者像素均差仅 ~2.4/255（JPEG 振铃级别），person 置信度却
能 0 → 0.42 地翻转 —— 说明模型在 0.15~0.4 的低置信区几乎处于噪声判决状态，
比较方案时务必固定同一条抽帧路径。

用法：
    ./.venv/Scripts/python.exe scripts/probe_blindspot.py --fps 1 --width 640 --input 640
    ./.venv/Scripts/python.exe scripts/probe_blindspot.py --fps 1 --width 1280 --input 1280 --limit 30
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from src.analyzer import YoloDetector  # noqa: E402
from src.config import load_config  # noqa: E402
from src.db import Database  # noqa: E402


def probe_size(ffprobe: str, src: Path):
    out = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", str(src)],
        capture_output=True, text=True, timeout=60).stdout.strip()
    try:
        w, h = [int(x) for x in out.split(",")[:2]]
        return w, h
    except Exception:
        return None


def detect_stream(ffmpeg: str, ffprobe: str, src: Path, fps: float, width: int,
                  dur: float, det: YoloDetector):
    """返回 (帧数, 有人的帧数, ≥0.25 的帧数, 最大人置信度, 最大置信度那一帧)。"""
    sw, sh = probe_size(ffprobe, src) or (width, width)
    th = max(2, int(round(sh * width / sw / 2)) * 2)
    cmd = [ffmpeg, "-v", "error", "-i", str(src), "-t", f"{dur:.3f}",
           "-vf", f"fps={fps:.6f},scale={width}:{th}",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    fsz = width * th * 3
    n = n_p = n_p25 = 0
    best = (0.0, None)
    try:
        while True:
            raw = p.stdout.read(fsz)
            if not raw or len(raw) < fsz:
                break
            img = np.frombuffer(raw, np.uint8).reshape(th, width, 3)
            n += 1
            final, _ = det.detect(img, raw_min=0.10)
            ps = [d["conf"] for d in final if d["class"] == "person"]
            if ps:
                mx = max(ps)
                n_p += 1
                if mx >= 0.25:
                    n_p25 += 1
                if mx > best[0]:
                    best = (mx, img.copy())
    finally:
        p.stdout.close()
        p.wait(timeout=30)
    return n, n_p, n_p25, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--input", type=int, default=640, help="模型输入边长")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="data/probe")
    args = ap.parse_args()

    cfg = load_config()
    db = Database(cfg["runtime"]["sqlite_path"])
    ffmpeg = cfg.get("runtime", {}).get("ffmpeg_path") or "ffmpeg"
    ffprobe = cfg.get("runtime", {}).get("ffprobe_path") or "ffprobe"
    model = cfg["analysis"].get("yolo_model_path") or "data/models/yolov8n.onnx"
    det = YoloDetector(model, args.input, 0.15, 0.45)

    manifest = Path(cfg["highlight"]["dir"]) / "manifest.jsonl"
    mf = {}
    for line in manifest.open(encoding="utf-8"):
        line = line.strip()
        if line:
            try:
                r = json.loads(line)
                mf[r["id"]] = r
            except Exception:
                pass

    with db.conn() as c:
        rows = {r["id"]: dict(r) for r in c.execute("SELECT * FROM videos").fetchall()}

    targets = []
    for vid, m in mf.items():
        if (m.get("stats") or {}).get("coarse_sent") != 0:
            continue
        row = rows.get(vid)
        if not row or not row.get("has_person"):
            continue
        src = next((Path(p) for p in (m.get("src"), row.get("trash_path"))
                    if p and Path(p).exists()), None)
        if src is None:
            continue
        targets.append((vid, row, m, src))
    targets.sort(key=lambda x: -(x[1].get("person_conf") or 0))
    if args.limit:
        targets = targets[:args.limit]

    out_root = Path(args.out) / f"i{args.input}_f{args.fps:g}_w{args.width}"
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"盲区样本 {len(targets)} 条  fps={args.fps}  抽帧宽={args.width}  模型输入={args.input}",
          flush=True)

    results = []
    t_start = time.time()
    for k, (vid, row, m, src) in enumerate(targets, 1):
        dur = float(m.get("duration_sec") or 60)
        n, n_p, n_p25, best = detect_stream(ffmpeg, ffprobe, src, args.fps,
                                            args.width, dur, det)
        conf = round(best[0], 3)
        if best[1] is not None:
            cv2.imwrite(str(out_root / f"{vid}_p{conf:.2f}.jpg"), best[1])
        results.append({"id": vid, "person_conf_L1": round(row.get("person_conf") or 0, 3),
                        "frames": n, "frames_person": n_p, "frames_person_25": n_p25,
                        "max_person_conf": conf, "dur": round(dur, 1)})
        flag = "★有人" if conf >= 0.25 else ("·低置信" if conf >= 0.15 else "  无人")
        print(f"[{k}/{len(targets)}] id={vid} conf_L1={row.get('person_conf'):.2f} "
              f"帧={n} 人帧={n_p} max={conf:.2f} {flag}", flush=True)

    (out_root / "result.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in results), encoding="utf-8")
    n_any = sum(1 for r in results if r["max_person_conf"] >= 0.15)
    n_hit = sum(1 for r in results if r["max_person_conf"] >= 0.25)
    print(f"\n=== 结果（{len(results)} 条，{time.time() - t_start:.0f}s）===")
    print(f"  检出人（≥0.15）：{n_any} 条  ({100 * n_any / max(1, len(results)):.0f}%)")
    print(f"  确认人（≥0.25）：{n_hit} 条  ({100 * n_hit / max(1, len(results)):.0f}%)")
    print(f"  产物目录：{out_root}")


if __name__ == "__main__":
    main()
