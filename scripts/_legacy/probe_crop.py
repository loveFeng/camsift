"""盲区检出目视核验：把「补检检出的人」画框放大存图，判断是真漏检还是误报。

配套 probe_blindspot.py 使用。只读，不写生产数据。

用法：
    ./.venv/Scripts/python.exe scripts/probe_crop.py 11999 9073 11976 --input 640
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402

from src.analyzer import YoloDetector  # noqa: E402
from src.config import load_config  # noqa: E402
from src.db import Database  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="+", type=int)
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--input", type=int, default=640)
    ap.add_argument("--out", default="data/probe")
    args = ap.parse_args()

    cfg = load_config()
    db = Database(cfg["runtime"]["sqlite_path"])
    ffmpeg = cfg.get("runtime", {}).get("ffmpeg_path") or "ffmpeg"
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

    out_root = Path(args.out) / f"verify_i{args.input}"
    out_root.mkdir(parents=True, exist_ok=True)

    for vid in args.ids:
        with db.conn() as c:
            row = dict(c.execute("SELECT * FROM videos WHERE id=?", (vid,)).fetchone())
        m = mf.get(vid, {})
        src = next((Path(p) for p in (m.get("src"), row.get("trash_path"),
                                      row.get("path")) if p and Path(p).exists()), None)
        if src is None:
            print(f"id={vid} 源文件找不到"); continue
        dur = float(m.get("duration_sec") or row.get("duration_sec") or 60)
        tmp = out_root / "_frames" / str(vid)
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)   # AUDIT-OK: 仅清理自创临时抽帧目录
        tmp.mkdir(parents=True, exist_ok=True)
        subprocess.run([ffmpeg, "-v", "error", "-i", str(src), "-t", f"{dur:.3f}",
                        "-vf", f"fps={args.fps:g},scale={args.width}:-2",
                        "-q:v", "3", str(tmp / "f_%05d.jpg")],
                       capture_output=True, timeout=900)
        frames = sorted(tmp.glob("f_*.jpg"))
        best = None
        for fp in frames:
            img = cv2.imread(str(fp))
            if img is None:
                continue
            final, _raw = det.detect(img, raw_min=0.10)
            ps = [d for d in final if d["class"] == "person"]
            if ps and (best is None or max(d["conf"] for d in ps) > best[0]):
                best = (max(d["conf"] for d in ps), img, ps, fp.name)
        if best is None:
            print(f"id={vid} 无检出"); continue
        conf, img, ps, fname = best
        ann = img.copy()
        for d in ps:
            x, y, w, h = [int(v) for v in d["box"]]
            cv2.rectangle(ann, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(ann, f'{d["conf"]:.2f}', (x, max(12, y - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.imwrite(str(out_root / f"{vid}_full_{conf:.2f}.jpg"), ann)
        # 最大人框放大 3 倍单独存 —— 用来判断"是人还是椅子/杂物"
        big = max(ps, key=lambda d: d["area_ratio"])
        x, y, w, h = [int(v) for v in big["box"]]
        pad = int(0.35 * max(w, h))
        crop = img[max(0, y - pad):min(img.shape[0], y + h + pad),
                   max(0, x - pad):min(img.shape[1], x + w + pad)]
        if crop.size:
            up = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(str(out_root / f"{vid}_crop_{big['conf']:.2f}.jpg"), up)
        print(f"id={vid} conf={conf:.2f} 人框={len(ps)} 帧={fname} → {out_root}")
        shutil.rmtree(tmp, ignore_errors=True)   # AUDIT-OK: 仅清理自创临时抽帧目录


if __name__ == "__main__":
    main()
