"""
10TB 监控视频 AI 筛选系统 —— 配置层

所有可调参数集中在此。程序启动时从项目根目录的 config.json 读取，
缺失项自动用 DEFAULT_CONFIG 补齐，因此增删配置项不会导致旧配置失效。

职责边界：本程序只做「分析 + 分级移动」，永不删除任何文件。
"""
from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.json"
DATA_DIR = ROOT / "data"
THUMB_DIR = DATA_DIR / "thumbs"
MODEL_DIR = DATA_DIR / "models"

# COCO 80 类名称（YOLOv8 官方顺序），用于把类别 id 翻译成人话
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]

DEFAULT_CONFIG: Dict[str, Any] = {
    # ---------------------------------------------------------------- 路径
    # 视频源根目录（移动硬盘，按年份分目录）。程序对其只读。
    "source_root": "",
    # 分级回收目录。必须与 source_root 在同一物理盘，否则 move 会退化成复制。
    "trash_root": "",
    # 本地 SSD 工作目录：数据库 / 缩略图 / 日志。绝不能落在移动硬盘上。
    "work_dir": str(DATA_DIR),

    # ---------------------------------------------------------------- 分析
    "analysis": {
        "sample_frames": 6,          # 每个片段抽帧数
        "grid_cols": 3,              # 缩略图网格列数（行数自动算）
        "thumb_width": 320,          # 单帧缩略图宽度
        "decode_backend": "auto",    # auto / cuda / cpu
        "yolo_enabled": True,        # 没有模型文件时自动降级为纯动静检测
        "yolo_model_path": str(MODEL_DIR / "yolov8n.onnx"),
        "yolo_conf_threshold": 0.50,  # 判定阈值
        "yolo_keep_raw_above": 0.15,  # 低于此值不记录，但仍会记录高于它的"未达阈值"检出
        "yolo_input_size": 640,
        "nms_threshold": 0.45,
    },

    # ---------------------------------------------------------------- 打分
    "scoring": {
        "person_weight": 50,
        "vehicle_weight": 40,
        "animal_weight": 15,
        "motion_area_weight": 10,
        "motion_area_threshold": 0.05,   # 前景面积占比超过此值才加分
        "audio_weight": 10,
        "audio_energy_threshold": 0.02,  # 归一化音频 RMS 超过此值认为有声音
    },

    # ---------------------------------------------------------------- 结论
    "verdict": {
        # score <= trash_threshold        → 建议清理（A/B 档）
        # score <  review_threshold       → 人工审核
        # score >= review_threshold       → 保留
        "trash_threshold": 0,
        "review_threshold": 30,
    },

    # ---------------------------------------------------------------- 类别
    "classes": {
        "person": ["person"],
        "vehicle": ["car", "truck", "bus", "motorcycle", "bicycle", "train"],
        "animal": ["cat", "dog", "bird", "horse", "sheep", "cow", "elephant", "bear"],
    },

    # ---------------------------------------------------------------- L2 VLM
    "vlm": {
        "enabled": False,
        "endpoint": "http://127.0.0.1:11434",   # Ollama 兼容的 OpenAI 接口
        "model": "qwen3-vl:4b",
        "temperature": 0.1,
        "timeout": 90,
        "max_image_side": 1280,
    },

    # ---------------------------------------------------------------- 音频复核 L2.5
    # 视觉判定"无目标"的删除候选，可能画面无人但画外有人声（电话/对讲/询价/警报）。
    # 这层在删除前抽音轨 → VAD 人声检测 → ASR 转写 → LLM 判有用性，供人二次确认。
    "audio": {
        "enabled": False,              # 总开关（引擎安装就绪后再开）
        "engine": "funasr",            # funasr（阿里）/ faster_whisper（降级兜底）
        "device": "auto",              # auto=有 CUDA 用 GPU 否则 CPU；cuda/cpu 强制。
                                       # GPU 被其它任务（如 DVR 转码）占满/驱动重置时改 cpu 保复核不饿死
        "model_dir": str(MODEL_DIR / "audio"),   # fsmn-vad / paraformer 模型目录
        "whisper_model": "small",      # faster_whisper 降级时的模型规格
        "language": "zh",              # ASR 语言
        "sample_seconds": 0,           # 0=抽全量音轨；>0 只抽前 N 秒（快速初筛）
        "vad_speech_threshold": 0.3,   # 语音占总时长比例低于此值 → "基本无人声"
        "min_speech_sec": 1.0,         # 单段人声短于该秒数视为噪音段，不转写
        "punc_model": True,            # 是否启用标点模型 ct-punc（让人好读）
        "max_segments": 40,            # 单片段最多转写段数，超长截断（控制成本）
        "cleanup_wav": False,          # 复核落库后是否清理抽出的 wav（要播放则保留）
    },

    # ---------------------------------------------------------------- LLM 有用性判断
    # 把 ASR 转写文本 + 元数据送本地 LLM，判断"这段声音内容是否可能有用"。
    # 结论仅是参考意见，最终由人在审核台拍板。
    "llm": {
        "enabled": False,
        "endpoint": "http://127.0.0.1:11434",   # ollama OpenAI 兼容接口
        "model": "qwen3:4b",
        "temperature": 0.1,
        "timeout": 120,
    },

    # ---------------------------------------------------------------- 运行
    "runtime": {
        "batch_size_limit": 500,     # 单次批量移动上限
        "default_batch_limit": 200,  # 执行上限（扫描/分析/音频复核共用），启动时记住上次值
        "sqlite_path": str(DATA_DIR / "videos.db"),
        "ffprobe_path": "",          # 留空则自动从 PATH 找
        "ffmpeg_path": "",
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    """递归合并，override 优先；只合并 dict，其余直接覆盖。"""
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------- 同步风险检测
# 网盘同步目录里的 SQLite 数据库会被"双向同步"来回覆盖：
# 本地写入一半时文件被同步、NAS 端的旧版本覆盖回来 —— 几天的分析结果可能凭空消失。
SYNC_MARKERS = {
    ".SynologyWorkingDirectory": "群晖 Synology Drive",
    ".synology": "群晖 Synology Drive",
    ".sync": "Resilio Sync / Syncthing",
    ".dropbox": "Dropbox",
    ".dropbox.cache": "Dropbox",
    ".seafile-data": "Seafile",
    ".onedrive": "OneDrive",
    "desktop.ini": None,   # 太泛，只作参考，不据此判定
}


def detect_sync_risk(path: Path | str | None = None) -> str | None:
    """检测路径是否位于网盘同步目录内。

    沿路径逐级向上查找同步软件留下的标记文件。
    返回同步软件名，无风险返回 None。
    """
    p = Path(path) if path else ROOT
    try:
        p = p.resolve()
    except OSError:
        return None
    for parent in [p, *p.parents]:
        try:
            if parent.name.lower() in ("onedrive", "one drive", "dropbox", "坚果云"):
                return f"{parent.name}（同步目录：{parent}）"
            for marker, name in SYNC_MARKERS.items():
                if name and (parent / marker).exists():
                    return f"{name}（同步目录：{parent}）"
        except OSError:
            continue
    return None


def _default() -> Dict[str, Any]:
    """返回一份全新的默认配置（深拷贝，避免调用方改到 DEFAULT_CONFIG）。"""
    return json.loads(json.dumps(DEFAULT_CONFIG))


def _notice(msg: str) -> None:
    """记录最近一条配置相关提示，供 UI 展示给用户。"""
    global LAST_NOTICE
    LAST_NOTICE = msg


LAST_NOTICE: str = ""


def _is_valid(cfg: Any) -> bool:
    return isinstance(cfg, dict) and bool(cfg)


def _read_json(path: Path) -> Dict[str, Any] | None:
    """读取并校验一个 json 配置。无效（空/损坏/非对象）时返回 None。"""
    try:
        if not path.exists() or path.stat().st_size == 0:
            return None
        cfg = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, OSError, ValueError):
        return None
    return cfg if _is_valid(cfg) else None


def _candidates(p: Path) -> list[Path]:
    """配置损坏时的恢复候选，按优先级排序。

    1. config.json.bak      —— 上次成功保存的完好副本（信息最贴近）
    2. *Conflict*.json      —— 网盘同步冲突产生的副本（用户真实配置常在这里）
    3. 默认配置             —— 兜底
    """
    out = [p.with_suffix(p.suffix + ".bak")]
    out += sorted(
        p.parent.glob("*Conflict*.json"),
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )
    return [c for c in out if c.exists() and c != p]


def load_config(path: Path | str | None = None) -> Dict[str, Any]:
    """加载配置。

    容错策略（关键）：配置文件可能被网盘同步/断电/写入中断搞成 0 字节或半截 JSON。
    这种情况下**绝不让程序崩**，而是按 bak → 同步冲突副本 → 默认 的顺序恢复，
    把损坏文件隔离留档，并通过 LAST_NOTICE 告知用户。
    """
    global LAST_NOTICE
    LAST_NOTICE = ""
    p = Path(path) if path else CONFIG_PATH

    if not p.exists():
        return _default()

    user_cfg = _read_json(p)
    if user_cfg is not None:
        return _deep_merge(DEFAULT_CONFIG, user_cfg)

    # ---- 配置损坏：隔离 + 恢复 ----
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    corrupt = p.with_name(f"{p.name}.corrupt-{stamp}")
    try:
        shutil.copy2(p, corrupt)  # 留档，不删除任何东西
    except OSError:
        corrupt = p

    for cand in _candidates(p):
        recovered = _read_json(cand)
        if recovered is not None:
            merged = _deep_merge(DEFAULT_CONFIG, recovered)
            try:
                save_config(merged, p)
            except OSError:
                pass
            _notice(f"配置文件损坏（0 字节或格式错误），已自动从 {cand.name} 恢复。"
                    f"损坏副本留存为 {corrupt.name}。")
            return merged

    merged = _default()
    try:
        save_config(merged, p)
    except OSError:
        pass
    _notice(f"配置文件损坏且未找到可用备份，已重置为默认配置。"
            f"损坏副本留存为 {corrupt.name}。**请重新填写视频源目录与回收目录**。")
    return merged


def save_config(cfg: Dict[str, Any], path: Path | str | None = None) -> Path:
    """原子写入配置。

    为什么必须原子：直接用 open("w") 会先截断文件再写入，
    一旦中途崩溃/被同步软件抢走文件句柄，就会留下 0 字节的 config.json，
    下次启动直接 JSONDecodeError 崩溃。先写临时文件再 os.replace 可彻底避免。
    """
    p = Path(path) if path else CONFIG_PATH
    p.parent.mkdir(parents=True, exist_ok=True)

    tmp = p.with_name(f"{p.name}.tmp-{os.getpid()}")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)  # 原子替换，不存在"半截文件"的中间态
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass

    # 写入成功后才更新 .bak —— 让 .bak 始终等于"最后一份确认有效的配置"。
    # （若在写之前备份，恢复出来的会是上一版，用户刚做的修改就白丢了。）
    try:
        shutil.copy2(p, p.with_suffix(p.suffix + ".bak"))
    except OSError:
        pass
    return p


def ensure_dirs(cfg: Dict[str, Any] | None = None) -> None:
    """确保工作目录存在。注意：只创建本地工作目录，绝不创建/改动源盘目录。"""
    cfg = cfg or load_config()
    Path(cfg["work_dir"]).mkdir(parents=True, exist_ok=True)
    THUMB_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    db_path = Path(cfg["runtime"]["sqlite_path"])
    db_path.parent.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_dirs()
    if not CONFIG_PATH.exists():
        save_config(load_config())
        print(f"已生成默认配置：{CONFIG_PATH}")
        print("请先编辑 source_root / trash_root 后再运行。")
    else:
        print(f"配置文件已存在：{CONFIG_PATH}")
