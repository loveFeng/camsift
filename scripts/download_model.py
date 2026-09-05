"""
下载 YOLOv8-nano ONNX 模型（约 12MB）。

用 OpenCV DNN 直接加载，不需要 torch / ultralytics，
因此在 8GB GPU 或纯 CPU 环境都能跑，部署成本极低。

用法：
    python scripts/download_model.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import MODEL_DIR, load_config  # noqa: E402

# Ultralytics 官方 assets 仓库。注意：yolov8n.onnx 只在 v8.4.0 及之后提供，
# v8.3.0 只有 yolo11 系列 —— 这也是早期版本下载 404 的原因。
URLS = [
    "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.onnx",
    "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n.onnx",
    "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.onnx",
    "https://hf-mirror.com/Ultralytics/yolov8/resolve/main/yolov8n.onnx",
]


def verify(path: Path) -> bool:
    """验证 ONNX 能被 OpenCV DNN 加载，并检查输出维度是否为 YOLOv8 格式 (1,84,8400)。"""
    try:
        import cv2
        import numpy as np
        net = cv2.dnn.readNetFromONNX(str(path))
        blob = np.zeros((1, 3, 640, 640), dtype=np.float32)
        net.setInput(blob)
        out = net.forward()
        print(f"  模型可用，输出形状 {out.shape}")
        if out.ndim == 3 and out.shape[1] in (84, 85):
            return True
        print(f"  警告：输出形状 {out.shape} 与预期 (1,84,8400) 不符，后处理可能需要调整")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  模型校验失败：{type(e).__name__}: {e}")
        return False


def download(url: str, dest: Path, retries: int = 6) -> bool:
    """带断点续传与重试的下载。

    某些网络环境（如代理/CDN 中转）会在传输中途掐断连接，
    单纯重试会从头开始、反复失败；断点续传能一截一截把文件拼完。
    """
    import time
    import urllib.request

    for attempt in range(1, retries + 1):
        have = dest.stat().st_size if dest.exists() else 0
        headers = {"User-Agent": "Mozilla/5.0"}
        if have > 0:
            headers["Range"] = f"bytes={have}-"
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=90) as r:
                # 服务器忽略 Range 时会返回 200，此时必须从头写
                if r.status == 200 and have > 0:
                    have = 0
                total = int(r.headers.get("Content-Length") or 0)
                end = (have + total) if total else 0
                mode = "ab" if have > 0 else "wb"
                if attempt == 1:
                    print(f"下载中：{url}")
                with dest.open(mode) as f:
                    while True:
                        chunk = r.read(32768)
                        if not chunk:
                            break
                        f.write(chunk)
                        have += len(chunk)
                        if end:
                            print(f"\r  {have}/{end} ({have*100//end}%)", end="", flush=True)
            print()
            if end and have >= end:
                return True
            if not end and have > 1_000_000:
                return True
        except Exception as e:  # noqa: BLE001
            cur = dest.stat().st_size if dest.exists() else 0
            print(f"\n  第 {attempt}/{retries} 次中断（{type(e).__name__}），"
                  f"已下载 {cur/1048576:.1f} MB，2 秒后从断点继续...")
            time.sleep(2)
    return False


def main() -> int:
    cfg = load_config()
    dest = Path(cfg["analysis"].get("yolo_model_path") or (MODEL_DIR / "yolov8n.onnx"))
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and dest.stat().st_size > 1_000_000:
        print(f"模型已存在：{dest}（{dest.stat().st_size/1048576:.1f} MB）")
        return 0

    for url in URLS:
        if download(url, dest) and verify(dest):
            print(f"完成：{dest}（{dest.stat().st_size/1048576:.1f} MB）")
            print("现在运行分析即可启用目标检测。")
            return 0
        dest.unlink(missing_ok=True)

    print("自动下载失败。请手动下载 yolov8n.onnx 并放到：")
    print(f"  {dest}")
    print(f"  候选地址：{URLS[0]}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
