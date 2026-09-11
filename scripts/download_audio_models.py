"""
下载 L2.5 音频复核所需的阿里 FunASR 模型（ModelScope 国内源，无需代理）。

模型清单（合计约 1.2GB）：
  fsmn-vad          人声分段 —— 回答"这段声音里有没有人说话、从哪秒到哪秒"
  paraformer-zh     ASR 转写 —— 中文监控画外音（远场/噪声鲁棒性优于 whisper）
  ct-punc           标点（可选）—— 让人读转写更顺

用法：
    python scripts/download_audio_models.py          # 全量
    python scripts/download_audio_models.py --skip-punc   # 跳过标点模型

依赖：modelscope（pip install modelscope）
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config  # noqa: E402

# funasr 简名 → ModelScope 完整仓库 id（与 funasr AutoModel 的 alias 表一致）
MODELS = {
    "fsmn-vad": "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    "paraformer-zh": "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
    "ct-punc": "iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
}
SIZES = {"fsmn-vad": "约 2MB", "paraformer-zh": "约 840MB", "ct-punc": "约 400MB"}


def _snapshot_download():
    """兼容 modelscope 不同版本的 snapshot_download 导入路径。"""
    try:
        from modelscope import snapshot_download
        return snapshot_download
    except Exception:  # noqa: BLE001
        from modelscope.hub.snapshot_download import snapshot_download  # type: ignore
        return snapshot_download


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-punc", action="store_true", help="跳过标点模型 ct-punc")
    args = ap.parse_args()

    cfg = load_config()
    base = Path(cfg.get("audio", {}).get("model_dir") or "data/models/audio")
    base = base if base.is_absolute() else Path.cwd() / base
    base.mkdir(parents=True, exist_ok=True)

    try:
        snapshot = _snapshot_download()
    except ImportError:
        print("未安装 modelscope。请先运行：pip install modelscope")
        return 1

    todo = dict(MODELS)
    if args.skip_punc:
        todo.pop("ct-punc", None)

    ok = True
    for key, model_id in todo.items():
        local = base / key
        # 已有模型目录且含 config.yaml → 跳过
        if (local / "config.yaml").exists():
            print(f"[跳过] {key} 已存在：{local}")
            continue
        print(f"[下载] {key}（{SIZES.get(key, '')}） <- {model_id}")
        try:
            snapshot(model_id, local_dir=str(local))
            if (local / "config.yaml").exists():
                print(f"[完成] {key} -> {local}")
            else:
                print(f"[警告] {key} 下载后未发现 config.yaml，目录：{local}")
                ok = False
        except Exception as e:  # noqa: BLE001
            print(f"[失败] {key}: {type(e).__name__}: {e}")
            ok = False

    if ok:
        print("\n全部模型就绪。在「设置与扫描 → 🔊 音频复核」点「引擎自检」确认，")
        print("或直接启动音频复核。")
    else:
        print("\n部分模型下载失败，请检查网络后重试。")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
