"""
下载 YOLO 的 ONNX 模型，并用 OpenCV DNN **逐输入尺寸**校验能不能真跑起来。

用 OpenCV DNN 直接加载，不需要 torch / ultralytics，
因此在 8GB GPU 或纯 CPU 环境都能跑，部署成本极低。

用法：
    python scripts/download_model.py                    # 默认 yolov8n → 配置里的 yolo_model_path
    python scripts/download_model.py --spec yolov8s      # 候选模型，落到同目录的 yolov8s.onnx
    python scripts/download_model.py --spec yolo11s --force

要换模型，必须先过两道门（都是实测踩过的）
--------------------------------------------
1. **ONNX 得能被 OpenCV DNN 装载**。官方 `yolo11s.onnx`（v8.4.0）在 OpenCV 5.0.0 上
   装载就失败（onnx importer 报错），根本进不到推理。
2. **每个会用到的输入尺寸都得能 forward**。官方 `yolov8s.onnx` 把 reshape 写死在
   640，喂 `highlight.fine_size`（1280）会崩在 reshape2_layer 的 `outTotal == inpTotal`
   —— 精扫档会整段挂掉。只在 640 上试一次，会把这种模型判成"合格"。

模型来源：Ultralytics 官方 assets 仓库。**v8.4.0 同时提供 yolov8 与 yolo11 两族的
n/s/m ONNX**；v8.3.0 没有 `.onnx` 附件（早期版本下载 404 就是这个原因）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import MODEL_DIR, load_config  # noqa: E402

SPECS = ("yolov8n", "yolov8s", "yolov8m", "yolo11n", "yolo11s", "yolo11m")
RELEASE = "v8.4.0"


def urls_for(spec: str) -> list:
    base = f"https://github.com/ultralytics/assets/releases/download/{RELEASE}"
    return [f"{base}/{spec}.onnx"]


def verify(path: Path, sizes=None) -> bool:
    """验证 ONNX 能被 OpenCV DNN 装载，且在**每个会用到的输入尺寸**上都能 forward。

    sizes 默认取配置里真正会用到的两个：`analysis.yolo_input_size`（粗扫）
    与 `highlight.fine_size`（精扫）。缺任意一个都说明这模型不能上生产 ——
    精扫档是"负责确认"的那一级，它挂掉等于整条精华链路失效。
    """
    if sizes is None:
        cfg = load_config()
        sizes = [int(cfg["analysis"].get("yolo_input_size", 640)),
                 int(cfg.get("highlight", {}).get("fine_size", 1280))]
    try:
        import cv2
        import numpy as np
    except Exception as e:  # noqa: BLE001
        # 不能当"通过"：这是"换模型上生产"的闸门，验不了就得说验不了，
        # 否则 main 会照常打印"校验通过"，把没验过的模型顶上生产。
        print(f"  ✗ 缺 cv2/numpy，无法校验：{e}")
        print("    先装依赖再跑（本项目必备：pip install opencv-python numpy）。")
        return False

    try:
        net = cv2.dnn.readNetFromONNX(str(path))
    except Exception as e:  # noqa: BLE001
        print(f"  ✗ 装载失败（OpenCV DNN 读不了这个 ONNX）：{type(e).__name__}: {e}")
        return False

    ok = True
    for size in sizes:
        try:
            net.setInput(np.zeros((1, 3, size, size), dtype=np.float32))
            out = net.forward()
        except Exception as e:  # noqa: BLE001
            ok = False
            print(f"  ✗ 输入 {size}: {type(e).__name__}: {str(e)[:110]}")
            print(f"    → 该模型在 {size} 上跑不了，多半是导出时把 shape 写死了。"
                  f"精扫档用 fine_size，换模型前必须先解决这个。")
            continue
        print(f"  ✓ 输入 {size}: 输出 {out.shape}")
        if not (out.ndim == 3 and out.shape[1] in (84, 85)):
            print("    警告：输出形状与 YOLO 的 (1,84,N) 不符，后处理可能要调整")
    return ok


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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="下载并用 OpenCV DNN 校验 YOLO 的 ONNX 模型")
    ap.add_argument("--spec", default="yolov8n", choices=SPECS,
                    help="要哪个规格（默认 yolov8n，即当前生产用的那个）")
    ap.add_argument("--dest", default="",
                    help="落到哪个文件；默认：yolov8n → 配置里的 yolo_model_path，"
                         "其它规格 → 与它同目录的 <spec>.onnx（不动生产模型）")
    ap.add_argument("--force", action="store_true", help="已存在也重新下载")
    args = ap.parse_args(argv)

    cfg = load_config()
    prod = Path(cfg["analysis"].get("yolo_model_path") or (MODEL_DIR / "yolov8n.onnx"))
    if args.dest:
        dest = Path(args.dest)
    elif args.spec == "yolov8n":
        dest = prod
    else:
        dest = prod.with_name(f"{args.spec}.onnx")
    dest.parent.mkdir(parents=True, exist_ok=True)

    print(f"{'--dest 覆写' if args.dest else '规格 ' + args.spec} → {dest}")
    if dest.exists() and dest.stat().st_size > 1_000_000 and not args.force:
        print(f"  已存在（{dest.stat().st_size/1048576:.1f} MB），只校验不重下"
              f"（要重下加 --force）")
        ok = verify(dest)
        print("  校验通过：两个输入尺寸都能跑。" if ok
              else "  ⚠ 校验未通过 —— 这个模型**不能**直接上生产，理由见上。")
        return 0 if ok else 1

    url = urls_for(args.spec)[0]
    if not download(url, dest):
        print("自动下载失败。请手动下载并放到：")
        print(f"  {dest}")
        print(f"  候选地址：{url}")
        return 1

    size_mb = dest.stat().st_size / 1048576
    print(f"下载完成：{dest}（{size_mb:.2f} MB）")
    if verify(dest):
        print("校验通过：两个输入尺寸都能跑。")
        if dest == prod:
            print("现在运行分析即可启用目标检测。")
        else:
            print("这是**候选**模型，尚未接入生产 —— 生产读的是 "
                  "config.json 的 analysis.yolo_model_path，要换请自行改。")
        return 0

    # 下载完整但用不了：留着别删（可能要重新导出），但把话说清楚
    print("⚠ 文件已下载完整，但**校验未通过**，直接换上去会让精扫档整段失效。")
    print(f"  文件保留在：{dest}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
