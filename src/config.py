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
from typing import Any, Dict

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
        # 有声音的判据：ffmpeg volumedetect 的 mean_volume，单位 dB（负值，越接近 0
        # 越响）。-50 dB 是实测下来"环境底噪"与"确实有动静"的分界。
        # （旧键 audio_energy_threshold 写的是 RMS 口径，与这里的 dB 对不上，已废）
        "audio_db_threshold": -50.0,
    },

    # ---------------------------------------------------------------- 结论
    "verdict": {
        # score <= trash_threshold        → 建议清理（A/B 档）
        # score <  review_threshold       → 人工审核
        # score >= review_threshold       → 保留
        "trash_threshold": 0,
        "review_threshold": 30,

        # ---- 放行判定（ADR 0003）----
        # 放行 = 判删候选里"机器没看到任何人物信号"的片段：不经审核直接移入回收，
        # 由画面精华剪辑兜底重扫。两个条件是「与」关系：
        #   release_require_no_person  必须是"无人物信号"——L1 未检出人物，且
        #                              decision_trace 里也没有"疑似人物未达阈值"
        #                              （near_miss，漏检高发区）。动物/车辆不排除。
        #   release_threshold          分数上界，只作保险丝。因为 review 档本就 <30，
        #                              而"无人物信号"时分数上限也够不到保留线，
        #                              所以 30 等于不生效；把它设 0 并关闭语义条件，
        #                              即回到 ADR 0002 之前的"仅 0 分桶"行为。
        "release_threshold": 30,
        "release_require_no_person": True,
    },

    # ---------------------------------------------------------------- 类别
    "classes": {
        "person": ["person"],
        "vehicle": ["car", "truck", "bus", "motorcycle", "bicycle", "train"],
        "animal": ["cat", "dog", "bird", "horse", "sheep", "cow", "elephant", "bear"],
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
        "vad_speech_noise_thres": 0.6, # fsmn-vad 语音/噪声判别阈值（越高越宽容，越大越容易把噪声判成语音）
        "min_transcript_chars": 4,     # VAD 漏检时 ASR 兜底：转写有效字数≥该值判定有语音
        "dsp_filter": "highpass=f=120,lowpass=f=3400,afftdn=nr=12:nf=-25",
                                       # 三级流水线第 2 级：VAD 漏检时生成降噪副本再 VAD 的
                                       # ffmpeg 滤镜链（实测 2015 DVR 8kHz 音频 0%→30%+）。
                                       # 置空 "" 关闭第 2 级（漏检直接走整段 ASR 兜底）
        "punc_model": True,            # 是否启用标点模型 ct-punc（让人好读）
        "max_segments": 40,            # 单片段最多转写段数，超长截断（控制成本）
        "cleanup_wav": False,          # 复核落库后是否清理抽出的 wav（要播放则保留）
        "review_scope": "all",         # 复核范围: all=待清理+需人工确认 / trash=仅待清理
                                       #          / review=仅需人工确认（先给人看的内容先复核）
    },

    # ---------------------------------------------------------------- 语音档案
    # 删除视频前把"有效语音"的音频 + 转写永久收藏（用于 ASR 训练 / 资料检索）
    "archive": {
        "dir": "E:\\audio-archive",  # 语音档案存放目录（红线：绝不能设在源盘/回收目录
                                     # 所在的 G 盘——那里会被手动清空）
    },

    # ---------------------------------------------------------------- 画面精华 L2.8
    # 删除视频前的最后一道保险：把"画面里还有点东西"的几秒剪出来留存。
    # 音频已由 archive 全量保存，本段负责画面 —— 二者合起来，原视频才敢删。
    #
    # 思路来自 docs/自动剪辑方案说明.md：抽帧 → CLAHE 增强 → 帧差预筛 →
    # YOLO 检测 → 事件聚合 → 重编码切割。与 L1 的区别是**采样密度高一个量级**
    # （L1 每段只抽 16 帧，这里按秒级抽），专门捞 L1 稀疏采样漏掉的短暂瞬间。
    "highlight": {
        "enabled": True,
        "dir": "E:\\video-highlights",  # 精华存放目录（红线同上：绝不能设在源盘/回收目录所在盘）

        # ---- 为什么是两级：L1 每段只抽 12-16 帧（43 秒视频里仅 2.5%），小孩身形小、
        # 进出快、或被门框家具挡住只剩半身，这种密度必然漏。但全片拉到 1280 精扫要
        # 17 小时，不现实。所以：粗扫负责不漏（低阈值高密度），精扫负责看清（高分辨率）。
        # 第一级 · 粗扫全片
        "coarse_fps": 3.0,          # 采样密度 3fps（L1 的 8 倍）
        "coarse_size": 640,         # 检测输入尺寸
        "coarse_conf": 0.15,        # 阈值刻意压低 —— 误检交给精扫过滤，粗扫只管不漏
        # 第二级 · 精扫候选窗
        "fine_enabled": True,
        "fine_fps": 4.0,            # 窗内更密，补回粗扫帧之间的间隙
        "fine_size": 1280,          # 高分辨率 —— 小目标与半身画面主要靠这一级捞回
        "fine_conf": 0.25,          # 精扫阈值 = A/B 级的"确认"门槛
        "fine_window_pad": 2.0,     # 候选窗向两侧扩展的秒数（补回入场/出场）
        "max_fine_ratio": 1.5,      # 精扫帧数上限（相对粗扫帧数），超出只对高分窗精扫

        # ---- 分级产出：全自动必然有垃圾，用"分级目录"表达确定性，而不是让人来筛
        # A 确认 = 精扫确认 ≥8s 或多人   → 切 mp4（含原音）
        # B 短暂 = 精扫确认 2-8s        → 切 mp4（含原音）
        # C 疑似 = 精扫没确认但粗扫看到过，或纯动静 → 切 mp4 + 多张代表帧
        "suspect_conf": 0.15,       # 低于 fine_conf 但高于此值 → C 级疑似
        "grade_a_seconds": 8.0,     # A / B 级的时长分界
        "grade_a_persons": 2,       # 多人同框直接算 A 级

        # ---- 删前保全（Track C · ADR 0006）----
        # 定位：精华剪辑是"原片被删之前的最后一道保全"，所以偏置是**宁可多留**。
        # 这三个开关是本次保全量的主控，出问题一键退回旧行为。
        # ⚠️ 改这个值必须同步 highlighter.HL_VERSION 与 db.HIGHLIGHT_TARGET_VERSION
        #    （selftest 有断言钉住三者相等；漂移会让回收目录永远归不了零）
        "version": "hlv3",             # 产物口径版本；已剪过的行版本落后即进重剪队列
        "keep_coarse_suspect": True,   # 粗扫看到、精扫没确认 ⇒ 降级保留为 C（不再丢弃）
                                       # 关掉即回到旧行为：这类信号直接消失（D2 的成因）
        "c_cut_mp4": True,             # C 级也切可播放 mp4（关掉则只留代表帧）
        "c_thumbs": 3,                 # C 级每段留几张代表帧（A/B 用 thumb_per_clip）

        "clahe": True,              # 红外/夜视对比度增强（素材 94% 是夜间，必开）
        "motion_prefilter": True,   # 帧差预筛：静态画面不送检测，省 40-60% 算力
        "motion_threshold": 0.004,  # 帧差面积占比阈值（0.4%，刻意调低宁可多检）
        "prefilter_dilate": 1,      # 预筛窗口向两侧扩展的采样帧数（补回入场/出场帧）

        "keep_person": True,        # 人物片段算精华
        "keep_vehicle": False,      # 车辆片段算精华（门口过车一般无保留价值）
        "keep_animal": False,       # 宠物片段算精华
        "motion_fallback": True,    # 无人但有大幅动静（开关门/物品倒地）→ 归入 C 级
        "motion_fallback_threshold": 0.05,   # 触发兜底的帧差面积占比

        # 聚合：把离散的"有检帧"拼成有意义的事件
        "gap_merge": 3.0,           # 间隔小于此值的有检帧合并为同一事件（抵消遮挡抖动）
        "min_duration": 1.0,        # 短于此的事件丢弃（1s —— 小孩一闪而过也要能抓住）
        "pre_pad": 2.0,             # 事件起点前移，避免人物"半路冒出来"
        "post_pad": 3.0,            # 事件终点后移，动作收尾完整
        "max_clips_per_video": 8,   # 单视频最多切几段（按价值排序，防止输出失控）

        # 编码：必须重编码，不能 -c copy（监控 GOP 长，copy 会把切点吸附到关键帧）
        "crf": 20,                  # 视觉近无损
        "preset": "veryfast",
        "audio_bitrate": "128k",
        "gop_seconds": 2,           # 输出关键帧间隔（秒）—— 便于拖动预览
        "thumb_per_clip": 1,        # 每段抽几张代表帧（0=不抽）
        "thumb_width": 640,         # 代表帧宽度（C 级时这张图就是唯一产物，别太小）

        # ---- 审核减负（ADR 0001 先剪后删）：剪辑结果反哺处置决策 ----
        "keep_full_ratio": 0.6,     # 精华占比超过此值 → 整片都有内容，切片反而丢
                                    # 上下文 → 自动改判保留原片（全自动，无需人判断）
        "auto_dispose": True,       # 剪辑完成后是否自动处置未决候选：
                                    #   keep_full → 保留；其余（含无精华）→ 移入回收目录。
                                    # 关闭则只剪不移，处置仍由审核台人工完成。
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

    # ---------------------------------------------------------------- 一键执行 ADR 0004
    # 把「扫描 → 分析 → 音频复核 → 语音归档 → 批量放行 → 画面精华剪辑」串成一条链，
    # 参数一次配好、点一下跑完，终点是"回收目录就绪，只剩手动删除"。
    #
    # 顺序由依赖决定，不能随意调换：
    #   分析先于音频复核/剪辑 —— 两者的队列都只认已判定的 trash/review
    #   音频复核先于语音归档 —— 归档的数据源是复核产出的转写与缓存 wav
    #   放行排在剪辑之前 —— 剪辑的自动处置会把未处置候选搬走，而放行队列只认
    #                        未处置，排在后面就是空操作（ADR 0001 的原话也是
    #                        "放行后由画面精华剪辑用 8 倍采样密度兜底重扫"）
    "pipeline": {
        "stages": {                 # 六个阶段可单独勾掉，默认全开
            "scan": True,
            "analyze": True,
            "audio": True,
            "archive": True,
            "release": True,
            "highlight": True,
        },
        # 下游各阶段的单次处理上限。按 ADR 0004 的口径，只有扫描按「本次上限」
        # 分批导入，其余阶段一律"跑全库队列"（带幂等标记，不会重复劳动）——
        # 这个值只是防跑飞的天花板，正常远大于积压量。
        "stage_limit": 100000,
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


def bin_path(cfg: Dict[str, Any], key: str, name: str) -> str | None:
    """ffmpeg / ffprobe 路径：配置指定优先，其次 PATH，都没有返回 None。

    五个模块（扫描 / 分析 / 复核 / 剪辑 / 预览）都要找这两个可执行文件，
    口径必须一致 —— 各写一份 `which` 迟早会有一份漏掉配置里的自定义路径。
    找不到时返回 None，由调用方决定是抛自己的错误还是降级。
    """
    p = (cfg.get("runtime", {}).get(key) or "").strip()
    if p and Path(p).exists():
        return p
    return shutil.which(name)


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
