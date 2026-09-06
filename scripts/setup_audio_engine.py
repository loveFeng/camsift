"""
L2.5 音频复核引擎依赖安装（FunASR 系）。

背景坑位：funasr 1.4.x 在 requirements 里硬性 pin `numpy<2`，而 Python 3.13
下 numpy 1.26 没有 wheel，直接 `pip install funasr` 会尝试源码编译 numpy →
缺编译器直接失败。因此这里用 `--no-deps` 安装 funasr，再手动补齐其运行依赖
（这些依赖的最新版均兼容 numpy 2.x）。

用法：
    python scripts/setup_audio_engine.py

前置：torch 已装 CUDA 版（如 torch 2.14.0+cu126）。脚本会检查并提示。
模型下载另跑：python scripts/download_audio_models.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PY = sys.executable
INDEX = "https://pypi.tuna.tsinghua.edu.cn/simple"

# funasr 本体（--no-deps 避开 numpy<2 pin）
# 其余：modelscope(模型下载/API) / librosa+soundfile+scipy(音频IO/特征)
#       kaldi-native-fbank(特征) / sentencepiece(词表) / editdistance+jamo(评估，funasr import 需要)
FUNASR_DEPS = [
    "modelscope", "kaldi-native-fbank", "librosa", "soundfile",
    "sentencepiece", "editdistance", "jamo", "scipy",
]


def _run(args: list[str]) -> bool:
    print(f"\n$ python -m pip {' '.join(args)}")
    r = subprocess.run([PY, "-m", "pip", *args])
    return r.returncode == 0


def torch_cuda_ok() -> bool:
    try:
        import torch
        ok = bool(torch.cuda.is_available())
        print(f"torch {torch.__version__} | CUDA 可用: {ok}")
        return ok
    except ImportError:
        print("torch 未安装")
        return False


def main() -> int:
    print("=" * 62)
    print("L2.5 音频复核引擎依赖安装")
    print("=" * 62)

    if not torch_cuda_ok():
        print("\n[提示] 当前无 CUDA 版 torch。FunASR 可以用 CPU 跑（慢），")
        print("       批量复核建议 GPU。若需 GPU 加速，请先执行：")
        print("       pip install torch --index-url "
              "https://download.pytorch.org/whl/cu126")
        print("       脚本将继续安装 FunASR 依赖（CPU 也能用）。")

    steps = [
        (["install", "funasr==1.4.14", "--no-deps", "-i", INDEX], "安装 funasr（--no-deps 避开 numpy<2 pin）"),
    ]
    for pkg_group in [FUNASR_DEPS[i:i + 4] for i in range(0, len(FUNASR_DEPS), 4)]:
        steps.append(
            (["install", *pkg_group, "-i", INDEX],
             f"安装依赖：{', '.join(pkg_group)}"))

    failed = []
    for args, label in steps:
        print(f"\n---- {label} ----")
        if not _run(args):
            failed.append(label)

    print("\n" + "=" * 62)
    if failed:
        print("以下步骤失败，请查看上方错误：")
        for f in failed:
            print(f"  · {f}")
        print("可用 --no-cache-dir 或更换镜像重试。")
        return 1
    print("FunASR 依赖安装完成。")
    print("下一步：python scripts/download_audio_models.py 下载 ASR 模型。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
