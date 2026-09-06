"""
L2.5 音频复核引擎 —— 判删候选的"有声事件"捞回

视觉判定"无目标对象"只说明画面里看不到人/车，不代表没有内容：
画外通话、对讲、询价、警报、广播等声音事件会录在音轨里，删了就丢。
本模块在删除前把音频过一遍：

  1. ffmpeg 抽音轨  → work_dir/audio/{video_id}.wav（16kHz 单声道 pcm）
  2. FSMN-VAD 人声分段（阿里）→ 有无人声 / 语音占比 / 起止时间戳
  3. Paraformer-zh 转写（阿里 ASR）→ 带时间戳文本（给人看）
  4. LLM（ollama，OpenAI 兼容）判断内容是否可能有用 → 供人二次确认

红线（与全系统一致）
--------------------
- 本模块不删除、不移动任何**用户文件**。
- 抽出的 wav 是**自产中间产物**，只放 work_dir/audio/，
  落库后 cleanup_wav=True 时清理的是自己刚抽出的文件（AUDIT-OK）。
- LLM 结论只是参考意见；有说话内容一律先展示给人，最终由人拍板。

引擎可插拔：audio.engine = "funasr"（阿里，默认）| "faster_whisper"（兜底）。
引擎与模型均为懒加载：缺依赖/缺模型时给出明确修复指引，不拖垮主流程。
"""
from __future__ import annotations

import importlib
import json
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# --------------------------------------------------------------------------- 常量

SAMPLE_RATE = 16000          # 抽轨采样率（ASR 标准输入）
CHANNELS = 1                 # 单声道
MAX_EXTRACT_SEC = 600        # 单段音频抽取上限（防异常超长文件拖死任务）
FFMPEG_TIMEOUT = 180

JUDGE_SYSTEM = (
    "你是监控录像复核助手。下面是一段【已被 AI 判定画面中无人无车】的监控视频里"
    "录到的声音转写（可能含噪声音量、识别错字）。\n"
    "判断这段声音内容是否**可能对机主有用**（如：画外有人说话/争吵/电话、询价、"
    "警报、门铃、广播、异常声响等事件，机主事后可能想回看）。\n"
    "注意：纯环境噪声、风声雨声、车流白噪、无人说话→无用；"
    "只要疑似有人声交谈或异常事件→有用（宁可保守判有用）。\n"
    "只输出 JSON：{\"useful\": 0|1|2, \"reason\": \"一句话理由\", "
    "\"event_type\": \"speech|noise|alarm|music|other\"}\n"
    "useful: 1=可能有用 0=几乎无用 2=不确定。不要输出 JSON 以外的内容。"
)


class AudioEngineError(RuntimeError):
    """引擎/依赖/模型缺失或运行失败。"""


class AudioReviewer:
    """单实例复核器：懒加载引擎，对一批判删候选做音频复核。"""

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.a = cfg.get("audio", {})
        self.runtime = cfg.get("runtime", {})
        work_dir = Path(cfg.get("work_dir") or ".")
        self.audio_dir = work_dir / "audio"
        self._engine = None            # funasr 引擎缓存
        self._asr_engine = None        # faster_whisper 缓存
        self._warned: set = set()

    # ------------------------------------------------------------------ 工具
    def _ffmpeg(self) -> str:
        p = (self.runtime.get("ffmpeg_path") or "").strip()
        if p:
            return p
        found = shutil.which("ffmpeg")
        if not found:
            raise AudioEngineError("未找到 ffmpeg，请在 config 的 runtime.ffmpeg_path 指定")
        return found

    def extract_wav(self, video_path: str | Path, vid: int) -> Path:
        """抽 16kHz 单声道 pcm wav。sample_seconds>0 时只抽前 N 秒（快速初筛）。"""
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        out = self.audio_dir / f"{vid}.wav"
        if out.exists():          # 已抽过则复用（断点续跑友好）
            return out
        cmd = [self._ffmpeg(), "-y", "-i", str(video_path),
               "-vn", "-ac", str(CHANNELS), "-ar", str(SAMPLE_RATE),
               "-acodec", "pcm_s16le"]
        secs = int(self.a.get("sample_seconds") or 0)
        if secs > 0:
            cmd += ["-t", str(secs)]
        cmd += [str(out)]
        try:
            r = subprocess.run(cmd, capture_output=True, timeout=FFMPEG_TIMEOUT)
        except subprocess.TimeoutExpired as e:
            raise AudioEngineError(f"抽音轨超时（>{FFMPEG_TIMEOUT}s）") from e
        if r.returncode != 0 or not out.exists() or out.stat().st_size < 44:
            tail = r.stderr.decode("utf-8", "ignore")[-300:]
            raise AudioEngineError(f"抽音轨失败: {tail}")
        return out

    # ------------------------------------------------------------------ 引擎
    def check_ready(self) -> List[str]:
        """返回引擎就绪检查的问题列表（空=就绪）。UI 自检按钮用。"""
        problems: List[str] = []
        engine = self.a.get("engine", "funasr")
        if engine == "funasr":
            # 用 find_spec 轻量探测（import funasr 首次加载 >2 分钟，会卡 UI）
            if importlib.util.find_spec("funasr") is None:
                problems.append("未安装 funasr（阿里 ASR）。请运行: "
                                "pip install funasr kaldi-native-fbank "
                                "librosa soundfile sentencepiece jamo scipy")
            else:
                model_dir = self._model_dir()
                if not (model_dir / "fsmn-vad").exists():
                    problems.append(f"缺少 VAD 模型: {model_dir / 'fsmn-vad'}（运行下载脚本）")
                if not (model_dir / "paraformer-zh").exists():
                    problems.append(f"缺少转写模型: {model_dir / 'paraformer-zh'}（运行下载脚本）")
        elif engine == "faster_whisper":
            if importlib.util.find_spec("faster_whisper") is None:
                problems.append("未安装 faster-whisper。请运行: pip install faster-whisper")
        else:
            problems.append(f"未知引擎: {engine}（audio.engine 应为 funasr 或 faster_whisper）")

        # GPU 探活：驱动被重置 / 显存被其它任务（如 DVR GPU 转码）占满时，CUDA 推理会
        # 无限等待（表现为"卡着不动"）。启动前探一次，异常时给明确处置提示而非干等。
        dev = self._pick_device(self.a)
        if dev.startswith("cuda"):
            try:
                import torch
                t = torch.zeros(4, 4, device="cuda")
                (t @ t).cpu()   # 真实前向探活：驱动重置时这里会抛
                del t
                torch.cuda.empty_cache()
            except Exception as e:  # noqa: BLE001
                problems.append(
                    f"CUDA 探活失败（{type(e).__name__}）—— GPU 可能被其它任务占满或驱动已重置。"
                    f"建议先停止 DVR 等 GPU 转码任务再复核；或在设置里把「推理设备」改为 cpu"
                    f"（慢但稳定，不受抢卡影响）。")
        if self.cfg.get("llm", {}).get("enabled"):
            try:
                self._llm_ping()
            except Exception as e:  # noqa: BLE001
                problems.append(f"LLM 不可用（{type(e).__name__}: {e}）")
        return problems

    def _model_dir(self) -> Path:
        p = Path(self.a.get("model_dir") or "data/models/audio")
        return p if p.is_absolute() else Path.cwd() / p

    def _load_engine(self):
        """懒加载 funasr：VAD 与 ASR 两个 AutoModel。"""
        if self._engine is not None:
            return self._engine
        import funasr  # 延迟导入：缺依赖在 review_one 抛 AudioEngineError
        md = self._model_dir()
        vad_dir, asr_dir = md / "fsmn-vad", md / "paraformer-zh"
        punc = md / "ct-punc"
        if not vad_dir.exists() or not asr_dir.exists():
            raise AudioEngineError(
                f"缺少 ASR 模型（{vad_dir} 或 {asr_dir} 不存在）。"
                "请先运行下载脚本拉取 fsmn-vad / paraformer-zh 模型。")
        device = self._pick_device(self.a)   # 尊重配置：cuda/cpu/auto（GPU 被占时可切 cpu）
        # noqa: 大量 kwargs 由 funasr AutoModel 接收
        vad = funasr.AutoModel(model=str(vad_dir), device=device, disable_update=True)
        asr_kw = dict(model=str(asr_dir), device=device, disable_update=True)
        if punc.exists():
            asr_kw["punc_model"] = str(punc)
        asr = funasr.AutoModel(**asr_kw)
        self._engine = (vad, asr, funasr)
        return self._engine

    def _cuda_ok(self) -> bool:
        return "cuda" in self._pick_device(self.a)

    @staticmethod
    def _pick_device(a: Dict[str, Any]) -> str:
        """按配置选推理设备。auto：有 CUDA 用 GPU，否则 CPU。
        cpu：强制 CPU（GPU 被 DVR 转码等占满 / 驱动异常时保复核不饿死）。
        cuda：强制 GPU（探测失败也让引擎报错，不做静默回落）。"""
        want = str(a.get("device", "auto")).lower()
        try:
            import torch
            have = bool(torch.cuda.is_available())
        except Exception:  # noqa: BLE001 —— torch 未装/导入失败按无 GPU 处理
            have = False
        if want == "cpu":
            return "cpu"
        if want == "cuda":
            return "cuda:0"
        return "cuda:0" if have else "cpu"

    def _engine_fw(self):
        """faster-whisper 兜底引擎。"""
        if self._asr_engine is not None:
            return self._asr_engine
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise AudioEngineError("未安装 faster-whisper: pip install faster-whisper") from e
        size = self.a.get("whisper_model", "small")
        model = WhisperModel(size, device="cuda" if self._cuda_ok() else "cpu",
                             compute_type="float16" if self._cuda_ok() else "int8")
        self._asr_engine = model
        return model

    # ------------------------------------------------------------------ 复核
    def review_wav(self, wav: Path, duration_sec: float | None) -> Dict[str, Any]:
        """对一个 wav 做 VAD + ASR，返回结构化结果。引擎不可用抛 AudioEngineError。"""
        engine = self.a.get("engine", "funasr")
        if engine == "funasr":
            return self._review_funasr(wav, duration_sec)
        if engine == "faster_whisper":
            return self._review_fw(wav, duration_sec)
        raise AudioEngineError(f"未知 audio.engine: {engine}")

    def _review_funasr(self, wav: Path, duration_sec: float | None) -> Dict[str, Any]:
        vad, asr, _ = self._load_engine()
        dur = duration_sec or self._wav_duration(wav)

        # 1) VAD 人声分段。funasr 输出形如 [{'key':…, 'value': [[s_ms,e_ms],…]}] 或
        #    [{'value': …}]；低版本也可能直接是 [[s,e],…]。宽容解析。
        raw = vad.generate(input=str(wav))
        segs_ms: List[List[int]] = []
        for item in raw if isinstance(raw, (list, tuple)) else [raw]:
            v = item.get("value") if isinstance(item, dict) else None
            if isinstance(v, list):
                for pair in v:
                    if (isinstance(pair, (list, tuple)) and len(pair) >= 2
                            and isinstance(pair[0], (int, float))):
                        segs_ms.append([int(pair[0]), int(pair[1])])
            elif isinstance(item, (list, tuple)) and item and isinstance(item[0], (list, tuple)):
                segs_ms = [[int(x[0]), int(x[1])] for x in item]
        segs_ms = [s for s in segs_ms if s[1] > s[0]]

        min_sec = float(self.a.get("min_speech_sec", 1.0))
        segs_ms = [s for s in segs_ms if (s[1] - s[0]) / 1000.0 >= min_sec]
        speech_sec = sum((b - a) / 1000.0 for a, b in segs_ms)
        ratio = min(1.0, speech_sec / dur) if dur > 0 else 0.0

        # 2) 有语音段才跑 ASR（省算力，避免对纯噪声做无谓转写）
        text = ""
        if segs_ms:
            raw_asr = asr.generate(input=str(wav))
            texts = []
            for item in raw_asr if isinstance(raw_asr, (list, tuple)) else [raw_asr]:
                if isinstance(item, dict) and item.get("text"):
                    texts.append(str(item["text"]).strip())
                elif isinstance(item, str) and item.strip():
                    texts.append(item.strip())
            text = "\n".join(t for t in texts if t)

        speech_present = 1 if segs_ms else 0
        # 纯噪声被 VAD 放过但 ASR 空转写 → 视为无人声
        if speech_present and not text.strip():
            if ratio < float(self.a.get("vad_speech_threshold", 0.3)):
                speech_present = 0
        return {
            "speech_present": speech_present,
            "speech_ratio": round(ratio, 4),
            "speech_segments": json.dumps(
                [[a / 1000.0, b / 1000.0] for a, b in segs_ms], ensure_ascii=False),
            "transcript_text": text,
            "transcript_segments": json.dumps(
                [{"start": a / 1000.0, "end": b / 1000.0, "text": ""}
                 for a, b in segs_ms], ensure_ascii=False),
        }

    def _review_fw(self, wav: Path, duration_sec: float | None) -> Dict[str, Any]:
        model = self._engine_fw()
        dur = duration_sec or self._wav_duration(wav)
        lang = self.a.get("language", "zh")
        segments, info = model.transcribe(
            str(wav), language=lang, vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500})
        segs = []
        text_parts = []
        for s in segments:
            segs.append([round(float(s.start), 2), round(float(s.end), 2)])
            text_parts.append(s.text.strip())
        text = "\n".join(t for t in text_parts if t)
        speech_sec = sum(b - a for a, b in segs)
        ratio = min(1.0, speech_sec / dur) if dur > 0 else 0.0
        return {
            "speech_present": 1 if text.strip() else 0,
            "speech_ratio": round(ratio, 4),
            "speech_segments": json.dumps(segs, ensure_ascii=False),
            "transcript_text": text,
            "transcript_segments": json.dumps(
                [{"start": a, "end": b, "text": t} for (a, b), t in zip(segs, text_parts)],
                ensure_ascii=False),
        }

    @staticmethod
    def _wav_duration(wav: Path) -> float:
        import wave
        try:
            with wave.open(str(wav), "rb") as w:
                return w.getnframes() / float(w.getframerate() or 1)
        except Exception:  # noqa: BLE001
            return 0.0

    # ------------------------------------------------------------------ LLM 判断
    def judge(self, transcript: str, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """把转写文本送 LLM 判"是否可能有用"。失败返回 None（不阻断复核流程）。"""
        llm = self.cfg.get("llm", {})
        if not llm.get("enabled") or not transcript.strip():
            return None
        row = dict(row)  # 兼容 sqlite3.Row（无 .get）
        dur = row.get("duration_sec") or 0
        user = (
            f"视频信息：{row.get('filename') or ''}｜时长 {dur:.0f}s｜"
            f"画面判定无目标（无人无车）\n\n声音转写：\n{transcript[:3000]}"
        )
        payload = {
            "model": llm.get("model", "qwen3:4b"),
            "messages": [{"role": "system", "content": JUDGE_SYSTEM},
                         {"role": "user", "content": user}],
            "temperature": float(llm.get("temperature", 0.1)),
            "stream": False,
        }
        try:
            req = urllib.request.Request(
                f"{llm.get('endpoint','http://127.0.0.1:11434')}/v1/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST")
            with urllib.request.urlopen(req, timeout=int(llm.get("timeout", 120))) as r:
                data = json.loads(r.read().decode("utf-8"))
            content = data["choices"][0]["message"]["content"]
            return self._parse_judge(content)
        except Exception:  # noqa: BLE001 —— LLM 失败不阻断复核
            return None

    @staticmethod
    def _parse_judge(content: str) -> Dict[str, Any]:
        """宽容解析 LLM 输出：剥离 ```json 围栏，尽量取出 useful/reason/event_type。"""
        c = content.strip()
        if c.startswith("```"):
            c = c.strip("`")
            if c.lower().startswith("json"):
                c = c[4:]
        c = c.strip()
        try:
            obj = json.loads(c)
        except json.JSONDecodeError:
            m = {}
            for key in ("useful", "reason", "event_type"):
                import re
                mm = re.search(rf'"{key}"\s*:\s*"?([^",}}]+)"?', c)
                if mm:
                    m[key] = mm.group(1).strip().strip('"')
            obj = m
        useful = obj.get("useful", 2)
        if isinstance(useful, str):
            useful = {"有用": 1, "无用": 0, "不确定": 2}.get(useful,
                    1 if "有用" in useful else (0 if "无用" in useful else 2))
        try:
            useful = int(useful)
        except (TypeError, ValueError):
            useful = 2
        return {
            "useful": useful if useful in (0, 1, 2) else 2,
            "reason": str(obj.get("reason", ""))[:500],
            "event_type": str(obj.get("event_type", "other"))[:20],
        }

    def _llm_ping(self) -> None:
        llm = self.cfg.get("llm", {})
        req = urllib.request.Request(
            f"{llm.get('endpoint','http://127.0.0.1:11434')}/v1/models",
            method="GET")
        with urllib.request.urlopen(req, timeout=5) as r:
            r.read()

    # ------------------------------------------------------------------ 批处理
    def review_one(self, db, row) -> Dict[str, Any]:
        """复核单个片段并写库。返回结果字典（含错误信息供日志）。"""
        # sqlite3.Row 支持 row["列名"] 但不支持 .get() —— 统一转 dict 兼容两种访问
        row = dict(row)
        path = Path(row["path"])
        vid = row["id"]
        if not path.exists():
            db.mark_audio_skipped(vid, "missing")
            return {"id": vid, "skipped": "源文件不存在"}
        if not row["has_audio"]:
            db.mark_audio_skipped(vid, "no_audio")
            return {"id": vid, "skipped": "无音轨"}
        try:
            wav = self.extract_wav(path, vid)
            result = self.review_wav(wav, row.get("duration_sec"))
        except AudioEngineError as e:
            db.mark_audio_skipped(vid, f"error: {e}")
            return {"id": vid, "error": str(e)}
        except Exception as e:  # noqa: BLE001 —— 单段失败不拖垮批次
            db.mark_audio_skipped(vid, f"error: {type(e).__name__}: {e}")
            return {"id": vid, "error": f"{type(e).__name__}: {e}"}

        # LLM 有用性判断（可选，失败不影响转写落库）
        if result.get("speech_present") and self.cfg.get("llm", {}).get("enabled"):
            j = self.judge(result.get("transcript_text", ""), row)
            if j:
                result["llm_useful"] = j["useful"]
                result["llm_reason"] = f"[{j.get('event_type','other')}] {j.get('reason','')}".strip()
        else:
            # 无人声：LLM 无文本可判，视为无用（reason 留空让人抽查）
            result["llm_useful"] = 0

        result["audio_checked"] = 1
        db.update_audio_review(vid, result)
        if self.a.get("cleanup_wav"):
            try:
                wav.unlink()  # AUDIT-OK：清理自产中间产物
            except OSError:
                pass
        return {"id": vid, "speech": result.get("speech_present", 0),
                "text_len": len(result.get("transcript_text", "") or "")}

    def run_batch(self, db, limit: int | None = None,
                  progress: Optional[Callable[[int, int, str], None]] = None,
                  log: Optional[Callable[[str], None]] = None,
                  stop_event=None,
                  batch_id: int | None = None) -> Dict[str, Any]:
        """对判删候选批量复核。可中断，中断/失败不影响已写入结果。"""
        if not self.a.get("enabled"):
            return {"error": "音频复核未启用（config.audio.enabled=false）"}
        problems = self.check_ready()
        if problems:
            raise AudioEngineError("；".join(problems[:3]))
        rows = db.audio_review_queue(batch_id=batch_id,
                                     limit=limit or 1000)
        total = len(rows)
        if not total:
            if log:
                log("没有待复核的判删候选（有音轨、未复核、未处置）")
            return {"checked": 0, "total": 0}
        if log:
            log(f"开始音频复核：{total} 段（引擎={self.a.get('engine')}）")
        checked = speech = errors = 0
        t0 = time.time()
        for i, row in enumerate(rows):
            if stop_event is not None and stop_event.is_set():
                if log:
                    log(f"收到停止请求，已复核 {checked} 段，收尾中")
                break
            r = self.review_one(db, row)
            if "error" in r:
                errors += 1
                if log:
                    log(f"[{row['id']}] {r['error'][:120]}")
            else:
                checked += 1
                if r.get("speech"):
                    speech += 1
            if progress:
                progress(i + 1, total, f"{row['filename']}")
        if log:
            el = time.time() - t0
            log(f"复核完成：{checked} 段（检出人声 {speech} 段，失败 {errors} 段），"
                f"用时 {el:.0f}s，均速 {checked / el:.2f} 段/s")
        return {"checked": checked, "speech": speech, "errors": errors, "total": total}
