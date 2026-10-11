"""
自测脚本 —— 验证核心逻辑，可作为回归测试重复运行

覆盖：
  1. 误判归因的四种典型场景
  2. 回收档位分档规则
  3. 批次进度回写（UI 进度条的数据来源）
  4. 审核队列分页
  5. 空间统计与清单生成
  6. 后台任务：状态机 / 停止 / 防重复启动 / 异常兜底
  7. 快捷键键位 × UI 按钮文案一致性（防改文案后快捷键静默失效）
  8. 代码审计：确认全库无任何删除调用（程序永不删除的硬保证）
  9. 配置容错：损坏 / 0 字节 / 网盘冲突副本的恢复路径

用法：
    python scripts/selftest.py
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Windows GBK 终端编不了码 emoji（ui.py 按钮文案里有 🗑 等），
# 在模块装载阶段就把 stdout/stderr 切到 UTF-8，让所有 [OK]/[FAIL] 输出能落地。
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

from src import feedback as fb  # noqa: E402
from src import mover  # noqa: E402
from src import config as cfgmod  # noqa: E402
from src.config import load_config, save_config, _read_json, detect_sync_risk  # noqa: E402
from src.db import Database  # noqa: E402

PASS, FAIL = "  [OK]  ", "  [FAIL]"
_failures = 0


def check(cond: bool, msg: str):
    global _failures
    print(f"{PASS if cond else FAIL}{msg}")
    if not cond:
        _failures += 1


def section(title: str):
    print()
    print("=" * 66)
    print(title)
    print("=" * 66)


class FakeRow(dict):
    """模拟 sqlite3.Row：支持 dict 取值即可。"""

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k)


def blank_row(**kw):
    base = {
        "id": 1, "path": "x.mp4", "filename": "x.mp4",
        "raw_detections": "[]", "frame_stats": "{}", "decision_trace": "{}",
        "retention_score": 0, "verdict": "trash", "motion_area_ratio": 0.0,
        "year": "2024", "rel_dir": "", "batch_id": 1,
    }
    base.update(kw)
    return FakeRow(base)


def _tmp_db(n: int = 5):
    """建一个临时库，塞 n 条已扫描待分析的记录。返回 (临时目录, db, 批次id, 视频id列表)。"""
    d = Path(tempfile.mkdtemp(prefix="vs_test_"))
    db = Database(d / "t.db")
    bid = db.create_batch("test", str(d))
    ids = []
    for i in range(n):
        vid = db.upsert_video_meta(str(d / f"v{i}.mp4"), {
            "filename": f"v{i}.mp4", "rel_dir": "", "year": "2024",
            "size_bytes": 1024 * 1024 * 500, "mtime": 0, "duration_sec": 60,
            "width": 1920, "height": 1080, "video_codec": "hevc",
            "has_audio": 0, "audio_energy": None, "bitrate_kbps": 1000,
            "meta_ok": 1, "meta_error": None})
        db.set_video_batch(vid, bid)
        ids.append(vid)
    return d, db, bid, ids


def _mark_analyzed(db, vid: int, verdict: str = "trash", score: float = 0):
    db.update_analysis(vid, {"status": "analyzed", "retention_score": score,
                             "verdict": verdict, "decision_trace": {}})


# --------------------------------------------------------------------------- 1
def test_misjudgment(cfg):
    section("误判归因")
    # 测试输入必须确定：不依赖用户 config 的实际阈值（用户可能调过），
    # 强制用默认阈值 0.5 / 记录下限 0.15，保证"conf=0.38 未达阈值"成立。
    import copy
    cfg = copy.deepcopy(cfg)
    cfg.setdefault("analysis", {})["yolo_conf_threshold"] = 0.5
    cfg["analysis"]["yolo_keep_raw_above"] = 0.15
    # 1) 未启用 YOLO
    r = fb.analyze_misjudgment(blank_row(), cfg, yolo_on=False)
    check(r["category"] == "no_yolo", f"未启用模型 → {r['category']}")

    # 2) 有未达阈值的 person 检出 —— 最典型、最可操作
    row = blank_row(
        raw_detections=json.dumps(
            [{"class": "person", "conf": 0.38, "area_ratio": 0.02, "box": [1, 2, 30, 60]}],
            ensure_ascii=False),
    )
    r = fb.analyze_misjudgment(row, cfg, yolo_on=True)
    check(r["category"] == "low_confidence",
          f"置信度未达阈值 → {r['category']}（{r['detail'][:34]}…）")
    check("0.38" in r["detail"], "归因详情包含具体置信度数值")
    check("阈值" in r["suggestion"], "给出阈值调整建议")

    # 3) 夜视场景
    row = blank_row(frame_stats=json.dumps(
        {"brightness": 22.0, "saturation": 12.0, "is_low_light": True, "blur_var": 300}))
    r = fb.analyze_misjudgment(row, cfg, yolo_on=True)
    check(r["category"] == "night_vision", f"夜视/低照度 → {r['category']}")

    # 4) 有动静但无目标
    row = blank_row(motion_area_ratio=0.22)
    r = fb.analyze_misjudgment(row, cfg, yolo_on=True)
    check(r["category"] == "motion_only", f"有动静无目标 → {r['category']}")

    # 5) 模糊
    row = blank_row(frame_stats=json.dumps({"brightness": 120, "saturation": 60,
                                            "is_blurry": True, "blur_var": 40}))
    r = fb.analyze_misjudgment(row, cfg, yolo_on=True)
    check(r["category"] == "occlusion_blur", f"画面模糊 → {r['category']}")


# --------------------------------------------------------------------------- 2
def test_tier(cfg):
    section("回收档位分档")
    row_a = blank_row(retention_score=0, decision_trace=json.dumps({"near_miss": {}}))
    check(mover.decide_tier(row_a, cfg) == "A", "无目标且分数为 0 → A 档（可批量删除）")

    row_b = blank_row(retention_score=0,
                      decision_trace=json.dumps({"near_miss": {"person": 0.38}}))
    check(mover.decide_tier(row_b, cfg) == "B", "存在未达阈值疑似目标 → B 档（建议抽查）")


# --------------------------------------------------------------------------- 3
def test_batch_progress():
    """进度条的数据来源：batches.analyzed 必须随分析真实增长。"""
    section("批次进度回写")
    d, db, bid, ids = _tmp_db(5)
    try:
        db.refresh_batch_progress(bid)
        p = db.batch_progress(bid)
        check(p["total"] == 5 and p["scanned"] == 5 and p["analyzed"] == 0,
              f"未分析：总数={p['total']} 已扫描={p['scanned']} 已分析={p['analyzed']}")

        _mark_analyzed(db, ids[0])
        _mark_analyzed(db, ids[1])
        db.refresh_batch_progress(bid)
        p = db.batch_progress(bid)
        check(p["analyzed"] == 2, f"分析 2 条后 analyzed={p['analyzed']}（期望 2）")
        check(p["pending"] == 3, f"待分析 pending={p['pending']}（期望 3）")

        db.refresh_batch_progress(bid)
        db.refresh_batch_progress(bid)
        check(db.batch_progress(bid)["analyzed"] == 2, "重复刷新不累加（幂等）")

        db.set_human_decision(ids[2], "keep")
        db.refresh_batch_progress(bid)
        check(db.batch_progress(bid)["analyzed"] == 3, "人工处置后的片段计入已分析")

        # 失败项不能算进已分析，否则进度会虚高
        db.update_analysis(ids[3], {"status": "error", "analyze_error": "无法解码",
                                    "retention_score": 100, "verdict": "keep",
                                    "decision_trace": {}})
        db.refresh_batch_progress(bid)
        p = db.batch_progress(bid)
        check(p["analyzed"] == 3 and p["errors"] == 1,
              f"失败项不计入 analyzed（analyzed={p['analyzed']} errors={p['errors']}）")
    finally:
        shutil.rmtree(d, ignore_errors=True)


# --------------------------------------------------------------------------- 4
def test_paging():
    section("审核队列分页")
    d, db, bid, ids = _tmp_db(7)
    try:
        for vid in ids:
            _mark_analyzed(db, vid)
        total = db.count_review_queue()
        check(total == 7, f"待审核总数 = {total}（期望 7）")
        p0 = [r["id"] for r in db.review_queue(limit=3, offset=0)]
        p1 = [r["id"] for r in db.review_queue(limit=3, offset=3)]
        p2 = [r["id"] for r in db.review_queue(limit=3, offset=6)]
        check(len(p0) == 3 and len(p1) == 3 and len(p2) == 1,
              f"分页大小正确：{len(p0)}/{len(p1)}/{len(p2)}")
        check(not (set(p0) & set(p1)), "相邻页无重复条目")
        check(len(set(p0 + p1 + p2)) == 7, "三页合计覆盖全部 7 条")
    finally:
        shutil.rmtree(d, ignore_errors=True)


# --------------------------------------------------------------------------- 5
def test_reports(cfg):
    section("空间统计与清单生成")
    d, db, bid, ids = _tmp_db(4)
    try:
        for i, vid in enumerate(ids):
            _mark_analyzed(db, vid)
            db.set_human_decision(vid, "trash", trash_path=f"{d}/t/{i}.mp4",
                                  trash_tier="A" if i < 3 else "B")
        rep = db.space_report(bid)
        t = rep["total"]
        check(t["n"] == 4 and t.get("A") == 3 and t.get("B") == 1,
              f"统计：共 {t['n']} 个（A {t.get('A')} / B {t.get('B')}）")
        check(abs(t["bytes"] - 4 * 1024 * 1024 * 500) < 1, "容量统计正确")
        check(abs(t["secs"] - 240) < 0.01, "时长统计正确")

        cfg2 = json.loads(json.dumps(cfg))
        cfg2["trash_root"] = str(d)
        r = mover.write_reports(db, cfg2, bid)
        check(Path(r["csv"]).exists(), f"清单已生成：{Path(r['csv']).name}")
        check(Path(r["txt"]).exists(), f"空间统计已生成：{Path(r['txt']).name}")
        txt = Path(r["txt"]).read_text(encoding="utf-8")
        check("合计" in txt and "A_高置信" in txt, "统计报告含合计与档位说明")
    finally:
        shutil.rmtree(d, ignore_errors=True)


# --------------------------------------------------------------------------- 6
def test_jobs():
    section("后台任务（状态机 / 停止 / 防重复启动 / 异常兜底）")
    from src import jobs as jobsmod

    def slow(j, n=200, step=0.02):
        for i in range(n):
            if j.stop_requested():
                break
            time.sleep(step)
            j.processed = i + 1
            j.total = n

    job = jobsmod.MANAGER.get("__test__")
    ok, msg = jobsmod.MANAGER.start("__test__", lambda j: slow(j, 5, 0.02))
    check(ok, f"首次启动成功：{msg}")
    ok2, msg2 = jobsmod.MANAGER.start("__test__", lambda j: slow(j, 5, 0.02))
    check(not ok2 and "正在运行" in msg2, f"重复启动被拒绝：{msg2}")
    for _ in range(80):
        if not job.running:
            break
        time.sleep(0.05)
    check(job.status == "done", f"任务正常收尾：{job.status}")
    check(job.processed == 5, f"处理计数正确：{job.processed}")
    check(job.speed > 0, f"吞吐可计算：{job.speed:.1f}/秒")

    # 停止：置位后应在当前步结束后收尾，而不是被强杀（已完成的写入不能丢）
    jobsmod.MANAGER.start("__test__", lambda j: slow(j, 100, 0.03))
    time.sleep(0.12)
    jobsmod.MANAGER.stop("__test__")
    for _ in range(80):
        if not job.running:
            break
        time.sleep(0.05)
    check(job.status == "stopped", f"停止后状态：{job.status}")
    check(0 < job.processed < 100,
          f"停止时已处理 {job.processed} 条（未被强杀，写入不丢）")

    # 异常不应把整个进程打死，且状态可查
    def boom(_j):
        raise RuntimeError("模拟任务异常")

    jobsmod.MANAGER.start("__test__", boom)
    for _ in range(80):
        if not job.running:
            break
        time.sleep(0.05)
    check(job.status == "error" and "RuntimeError" in job.error,
          f"任务异常被捕获：{job.error}")


# --------------------------------------------------------------------------- 7
def test_hotkey_binding():
    """快捷键模块已**废弃**（v3 阶段 4）—— 这里改成反向断言钉住这件事。

    为什么废弃：快捷键靠 `clickByText(indexOf)` 按**按钮文案**定位控件，
    而 Streamlit 把**所有 tab 的内容都渲染进同一个 DOM**。于是
    "在任意页按空格"都会去点审核台的「确认清理」—— 一个静默的跨页误触。
    它的三层降级（components.html → iframe → pass）也让失效**毫无提示**。
    原本它只查"文案还在不在"，查不出这个真 bug；程序全驱动之后人工只抽 N 条看，
    快捷键的收益也趋近于 0。整模块删除（ADR 0007）。

    保留这条测试是为了防止将来有人觉得"快捷键挺方便"又加回来 ——
    那个坑是静默的，靠 review 发现不了。
    """
    section("快捷键模块已废弃（反向断言）")
    ui_src = Path("src/ui.py").read_text(encoding="utf-8")
    check(not Path("src/hotkeys.py").exists(),
          "src/hotkeys.py 不存在（模块已删）")
    check("hotkeys" not in ui_src,
          "ui.py 不再 import / 调用 hotkeys")
    check("DEFAULT_KEYS" not in ui_src,
          "没有残留的键位表引用")
    # 反向：也不该再有"跳到第 N 条"这类全量审核导航（ADR 0005 的产物已退役）
    check("rv_jump_idx" not in ui_src and "rv_jump_score" not in ui_src,
          "全量审核的跳转控件已退役（改为抽查页的随机抽样）")


# --------------------------------------------------------------------------- 8
def test_no_delete_calls():
    section("代码审计：确认没有删除调用")
    # 带 AUDIT-OK 标记的行是已确认安全的（如清理自己创建的临时目录），跳过
    banned = ["os.remove", "shutil.rmtree", "Path.unlink", "os.unlink", "os.rmdir"]
    hits = []
    for p in sorted(Path("src").rglob("*.py")):
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if "AUDIT-OK" in line:
                continue
            for b in banned:
                if b in line:
                    hits.append(f"{p}:{i}  {b}")
    check(not hits, f"src/ 下无删除用户数据的调用（检查项：{', '.join(banned)}）")
    for h in hits:
        print(f"      发现：{h}")


def test_model_verify_sizes():
    """换模型的两道门必须被 `download_model.verify()` 真的测出来。

    2026-09-12 实测踩到：官方 `yolov8s.onnx` 把 reshape 写死在 640，
    喂 `fine_size=1280` 会崩在 reshape2_layer；`yolo11s.onnx` 连装载都失败。
    而当时的 `verify()` **只在 640 上试一次**，这两种模型都会被判成"合格"。
    """
    section("模型校验：逐输入尺寸 + 装载（换模型的两道门）")
    import importlib.util

    import cv2  # verify() 内部 import cv2，拿到的是同一个模块对象，直接改属性即可

    p = Path("scripts/download_model.py")
    check(p.exists(), "download_model.py 在位")
    spec = importlib.util.spec_from_file_location("_dm", p)
    dm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dm)

    real_load = cv2.dnn.readNetFromONNX

    class _Net:
        def __init__(self, fail_at=None):
            self.fail_at = fail_at or []
            self.tried = []

        def setInput(self, blob):
            self.size = blob.shape[-1]
            self.tried.append(self.size)

        def forward(self):
            if self.size in self.fail_at:
                raise RuntimeError("reshape2: outTotal == inpTotal")
            import numpy as np
            return np.zeros((1, 84, 8400), dtype=np.float32)

    try:
        # A. 装载失败 → 判不合格（yolo11s 那种）
        def _boom(*a, **k):
            raise RuntimeError("onnx importer 不支持该图")

        cv2.dnn.readNetFromONNX = _boom
        check(dm.verify(Path("x.onnx"), sizes=[640]) is False,
              "装载失败 → 直接判不合格（引不到推理层就不该放行）")

        # B. 只在 1280 崩 → 判不合格，且两个尺寸都试过（yolov8s 那种）
        net = _Net(fail_at=[1280])
        cv2.dnn.readNetFromONNX = lambda *a, **k: net
        check(dm.verify(Path("x.onnx"), sizes=[640, 1280]) is False,
              "static-640 模型喂 1280 → 判不合格（精扫档会整段挂）")
        check(net.tried == [640, 1280],
              f"逐个尺寸都试过，不是只试第一个（试过 {net.tried}）")

        # C. 两档都过 → 合格
        net2 = _Net()
        cv2.dnn.readNetFromONNX = lambda *a, **k: net2
        check(dm.verify(Path("x.onnx"), sizes=[640, 1280]) is True,
              "两档都能跑 → 合格")
        check(net2.tried == [640, 1280], "合格判定也走过两个尺寸，不是短路")
    finally:
        cv2.dnn.readNetFromONNX = real_load

    # 下载清单：v8.4.0 才有 .onnx（v8.3.0 没有），且 s/m 都在
    check(set(("yolov8n", "yolov8s", "yolov8m", "yolo11n", "yolo11s", "yolo11m"))
          <= set(dm.SPECS), "SPECS 覆盖 n/s/m 两族（能直接下候选模型）")
    u = dm.urls_for("yolov8s")[0]
    check("v8.4.0" in u and u.endswith("yolov8s.onnx"),
          f"候选模型地址走 v8.4.0（{u.split('/')[-1]}）")

    # 默认尺寸必须取自**配置里真正会用到的两档**，不能写死
    src = p.read_text(encoding="utf-8")
    check("fine_size" in src and "yolo_input_size" in src,
          "默认校验尺寸取自配置（yolo_input_size + highlight.fine_size），不写死")


# --------------------------------------------------------------------------- 9
def test_config_resilience():
    """配置容错 —— 这是 2026-09-05 上线后真实踩过的坑，必须钉死回归。"""
    section("配置容错（损坏 / 0 字节 / 网盘冲突副本 / 原子写入）")

    # 准备一个独立目录，避免污染真实 config.json
    d = Path(tempfile.mkdtemp(prefix="cfg_test_"))
    cfg_path = d / "config.json"
    bak_path = d / "config.json.bak"
    user_cfg = {"source_root": "G:\\camera\\2015", "trash_root": "G:\\camera\\need-del",
                "work_dir": "E:\\code\\video-search\\data", "note": "用户改过的"}

    try:
        # —— A. 正常读取：损坏分支不应被触发
        save_config(user_cfg, cfg_path)
        check(bak_path.exists(), "保存后 .bak 同步生成")
        check(_read_json(cfg_path) is not None, "正常文件可读")
        cfg = load_config(cfg_path)
        check(cfg.get("note") == "用户改过的", "用户字段被保留")
        check(cfg["analysis"]["sample_frames"] == 6, "默认配置在缺失项上被补齐")
        check(not cfgmod.LAST_NOTICE, "正常加载不产生任何提示")

        # —— B. 0 字节：模拟同步软件把 config.json 截成空文件
        cfg_path.write_bytes(b"")
        cfg = load_config(cfg_path)
        check(cfg.get("note") == "用户改过的", "0 字节 → 从 .bak 恢复，保留用户修改")
        check("损坏" in cfgmod.LAST_NOTICE or "恢复" in cfgmod.LAST_NOTICE,
              f"产生恢复提示：{cfgmod.LAST_NOTICE[:30]}…")
        # 损坏副本应被隔离留档，**不能被删除**
        isolates = list(d.glob("config.json.corrupt-*"))
        check(len(isolates) == 1, f"损坏文件被隔离为 {isolates[0].name}")

        # —— C. JSON 半截：写一半被 kill 的常见状态
        cfg_path.write_text('{"source_root": "X"', encoding="utf-8")
        # .bak 还在，应该恢复到上一步写回的版本
        cfg = load_config(cfg_path)
        check(cfg.get("note") == "用户改过的", "半截 JSON → 仍能从 .bak 恢复")

        # —— D. .bak 也坏了，只有 Conflict 副本可用 —— 模拟群晖冲突的真实场景
        #      主文件存在但损坏 + .bak 损坏 + Conflict 完好 → 必须从 Conflict 恢复
        bak_path.unlink()
        cfg_path.write_text("{not valid json", encoding="utf-8")
        conflict = d / "config_USER-20251123YT_9月-05-105348-2026_Conflict.json"
        conflict.write_text(json.dumps(user_cfg, ensure_ascii=False), encoding="utf-8")
        cfg = load_config(cfg_path)
        check(cfg.get("note") == "用户改过的", "主+bak 都坏 → 从 Conflict 副本恢复")
        check(cfg_path.exists() and _read_json(cfg_path) is not None,
              "恢复过程中把可用配置写回主文件")
        check(any(d.glob("*Conflict*.json")), "Conflict 副本未被删除（供用户查阅）")

        # —— D2. 主文件干脆被同步软件删了 → 当作全新安装，用默认值
        cfg_path.unlink()
        cfg = load_config(cfg_path)
        check(cfg.get("scoring", {}).get("person_weight") == 50,
              "主文件不存在 → 按全新安装处理，返回默认配置")

        # —— E. 全部坏掉：主文件损坏 + .bak 损坏 + 无 Conflict → 给默认配置 + 提示重填
        bak_path.write_text("also broken{", encoding="utf-8")  # .bak 故意损坏
        cfg_path.write_text("{not valid", encoding="utf-8")     # 主文件损坏
        for f in d.glob("*Conflict*.json"):
            f.unlink()  # 删掉所有 Conflict 副本
        cfg = load_config(cfg_path)
        check(isinstance(cfg, dict) and cfg.get("scoring", {}).get("person_weight") == 50,
              "主+bak+Conflict 全坏 → 返回默认配置")
        check("重置为默认" in cfgmod.LAST_NOTICE,
              f"明确提示用户需要重填：{cfgmod.LAST_NOTICE[:50]}")

        # —— F. 原子写入：保存中途崩溃不应留下半截文件
        for ext in (".json", ".bak"):
            for f in d.glob(f"*{ext}"):
                f.unlink()
        cfg = {"source_root": "G:\\camera\\2024", "scoring": {"person_weight": 99}}
        save_config(cfg, cfg_path)
        # 不应残留任何 .tmp-* 临时文件
        tmps = list(d.glob("*.tmp-*"))
        check(not tmps, f"无残留临时文件（{len(tmps)} 个）")
        # .bak 必须是"最后一次成功保存"的副本，不是上一次
        bak_content = json.loads(bak_path.read_text(encoding="utf-8"))
        check(bak_content["scoring"]["person_weight"] == 99,
              ".bak 保存的是最新版本，不是上一版（修复过此 bug）")

        # —— G. 同步风险识别
        # 临时建一个假的同步目录树
        fake_sync = d / "SynologyDrive" / "project"
        fake_sync.mkdir(parents=True)
        (fake_sync / ".SynologyWorkingDirectory").touch()
        risk = detect_sync_risk(fake_sync)
        check(risk is not None and "群晖" in risk, f"识别群晖同步目录：{risk}")
        risk2 = detect_sync_risk(d)  # 父目录不在同步路径里
        check(risk2 is None, f"非同步路径返回 None（{risk2}）")
    finally:
        shutil.rmtree(d, ignore_errors=True)


# --------------------------------------------------------------------------- 10
def test_audio_review():
    """音频复核层（L2.5）回归：DB 迁移/队列/复核写入/judge 解析/UI 接线。

    不依赖 funasr 引擎与模型 —— 只测不碰引擎的分支与数据层逻辑，
    引擎行为在装好依赖后用真实音频做冒烟（见 docs/设计档案-10TB筛选系统.md §3.6）。
    """
    from src.audioreview import AudioReviewer
    from src.config import DEFAULT_CONFIG
    from src.db import AUDIO_COLUMNS

    section("音频复核 L2.5：DB 迁移 / 队列 / 复核写入 / judge / UI 接线")

    # 1) 默认配置含 audio/llm 段
    check("audio" in DEFAULT_CONFIG and DEFAULT_CONFIG["audio"].get("engine") == "funasr",
          "config 默认含 audio 段（engine=funasr）")
    check(DEFAULT_CONFIG["audio"].get("device", "auto") in ("auto", "cuda", "cpu"),
          "config 默认含推理设备（device=auto）")
    check("llm" in DEFAULT_CONFIG and DEFAULT_CONFIG["llm"].get("model", "").startswith("qwen3"),
          "config 默认含 llm 段（qwen3 模型）")

    # 2) 新库字段齐全
    d, db, bid, ids = _tmp_db(3)
    try:
        with db.conn() as c:
            cols = {r[1] for r in c.execute("PRAGMA table_info(videos)")}
        check(all(x in cols for x in AUDIO_COLUMNS),
              "新库 videos 表含全部音频复核列")

        # 3) 队列过滤：has_audio=1 判删候选才计入待复核
        check(db.count_audio_pending() == 0, "无有音轨候选时待复核为 0")
        # 把 v0 改为 has_audio=1 且 verdict=trash
        db.upsert_video_meta(str(d / "v0.mp4"), {
            "filename": "v0.mp4", "rel_dir": "", "year": "2024",
            "size_bytes": 1, "mtime": 0, "duration_sec": 60,
            "width": 1920, "height": 1080, "video_codec": "hevc",
            "has_audio": 1, "audio_energy": None, "bitrate_kbps": 1000,
            "meta_ok": 1, "meta_error": None})
        _mark_analyzed(db, ids[0], "trash", 0)
        check(db.count_audio_pending() == 1, "有音轨判删候选计入待复核（1 条）")
        q = db.audio_review_queue()
        check(len(q) == 1 and q[0]["id"] == ids[0], "audio_review_queue 命中该候选")

        # 4) 复核写入后不再进队列；跳过（无音轨/缺失）同理
        db.update_audio_review(ids[0], {
            "audio_checked": 1, "speech_present": 1, "speech_ratio": 0.5,
            "speech_segments": "[]", "transcript_text": "你好，有人吗",
            "transcript_segments": "[]", "llm_useful": 1, "llm_reason": "有人声"})
        check(db.count_audio_pending() == 0, "复核写入后不计入待复核")
        row = db.get_video(ids[0])
        check(row["speech_present"] == 1 and row["llm_useful"] == 1
              and "有人" in (row["transcript_text"] or ""),
              "复核结果正确落库")

        # 5) review_one 的引擎无关分支：无音轨 → 跳过；文件缺失 → 跳过
        cfg = {"work_dir": str(d), "audio": {"engine": "funasr"},
               "llm": {"enabled": False}, "runtime": {}}
        ar = AudioReviewer(cfg)
        # v1 has_audio=0 且真实文件存在 → no_audio 分支（不触发引擎）
        Path(d / "v1.mp4").write_bytes(b"x")
        db.upsert_video_meta(str(d / "v1.mp4"), {
            "filename": "v1.mp4", "rel_dir": "", "year": "2024",
            "size_bytes": 1, "mtime": 0, "duration_sec": 60,
            "width": 1920, "height": 1080, "video_codec": "hevc",
            "has_audio": 0, "audio_energy": None, "bitrate_kbps": 1000,
            "meta_ok": 1, "meta_error": None})
        r = ar.review_one(db, db.get_video(ids[1]))
        row = db.get_video(ids[1])
        check(r.get("skipped") == "无音轨" and row["audio_checked"] == -1
              and row["audio_error"] == "no_audio",
              "review_one：无音轨片段跳过（audio_checked=-1）")
        # v2 文件不存在 → missing 分支
        r = ar.review_one(db, db.get_video(ids[2]))
        check(r.get("skipped") == "源文件不存在",
              "review_one：源文件缺失跳过，不进引擎")
    finally:
        shutil.rmtree(d, ignore_errors=True)

    # 6) judge 解析（纯函数，5 例）
    from src.audioreview import AudioReviewer as AR
    p = AR._parse_judge
    check(p('{"useful": 1, "reason": "有人说话", "event_type": "speech"}')["useful"] == 1,
          "judge 解析：标准 JSON")
    check(p('```json\n{"useful": 0, "reason": "风声", "event_type": "noise"}\n```')["useful"] == 0,
          "judge 解析：```json 围栏")
    check(p('{"useful": "有用", "reason": "x"}')["useful"] == 1, "judge 解析：中文 useful")
    check(p('{"useful": 7, "reason": "y"}')["useful"] == 2, "judge 解析：越界值归一为不确定")
    check(p('没啥结构')["useful"] == 2 and p('没啥结构')["event_type"] == "other",
          "judge 解析：无 JSON 兜底为不确定")

    # 6.5) 容错：judge 直接收 sqlite3.Row（队列行）不崩
    #      sqlite3.Row 支持 row["列名"] 但无 .get() —— 入口须 dict(row)，
    #      曾因 review_one/judge 对 Row 调 .get() 抛 AttributeError
    import sqlite3
    _rc = sqlite3.connect(":memory:")
    _rc.row_factory = sqlite3.Row
    _rc.execute("CREATE TABLE t (id INT, path TEXT, filename TEXT, has_audio INT,"
                " duration_sec REAL)")
    _rc.execute("INSERT INTO t VALUES (1,'C:/no/such.mp4','t.mp4',1,12.5)")
    _rrow = _rc.execute("SELECT * FROM t").fetchone()
    ar_row = AudioReviewer({
        "audio": {"enabled": True, "engine": "funasr", "model_dir": "x"},
        "llm": {"enabled": True, "endpoint": "http://127.0.0.1:1",  # 立即拒连
                "model": "qwen3:4b", "timeout": 2},
        "runtime": {}, "work_dir": "."})
    try:
        _rj = ar_row.judge("有人在说话", _rrow)   # 端点拒连 → None，但不抛异常
        _row_ok = True
    except AttributeError:
        _row_ok = False
    check(_row_ok and (_rj is None or isinstance(_rj, dict)),
          "容错：judge 收 sqlite3.Row 不抛 AttributeError")

    # 7) UI / jobs 接线（源码级检查，防改 UI 忘接线）
    ui_src = (Path(__file__).resolve().parent.parent / "src" / "ui.py").read_text(encoding="utf-8")
    jobs_src = (Path(__file__).resolve().parent.parent / "src" / "jobs.py").read_text(encoding="utf-8")
    check("def render_audio_card" in ui_src and "render_audio_card(row, cfg)" in ui_src,
          "UI：抽查页已挂载音频复核卡")
    check("render_audio_card" in ui_src and "_grid_audio_html(row)" in ui_src,
          "UI：音频摘要行也挂在抽查页（转写一眼可见）")
    # ⚠️ 音频复核关口：旧审核台会 `disabled=not confirm_skip_audio`（硬拦），
    # 抽查页**故意不拦、只提示**。理由：音频队列已放宽为「处置候选」，
    # 移入回收后仍能补复核（review_one 有 trash_path 回落）⇒ 程序路径
    # （_dispose_one / move_all）早已撤销同一道守卫；UI 再拦就是"半拦半不拦"，
    # 会让一键执行的终点不可预期。这里钉死"有提示、不拦"。
    check("有音轨但尚未音频复核" in ui_src,
          "UI：抽查页对未音频复核的片段给提示")
    # 只查**代码行**（注释里提到旧名字不算数 —— 注释里要解释清楚
    # "为什么撤掉这道关口"，那些说明必须留着）
    code_only = "\n".join(ln for ln in ui_src.splitlines()
                          if not ln.strip().startswith("#"))
    check("confirm_skip_audio" not in code_only and "b_gate_ok" not in code_only,
          "★UI 不再有硬拦式音频关口（与程序路径同口径）")
    check('_AUDIO = "audio"' in jobs_src and "音频复核" in jobs_src
          and "start_audio_review" in jobs_src,
          "jobs：音频复核任务类型注册")
    check("jobsmod.get_audio_job()" in ui_src,
          "UI：进度页接入音频任务面板（get_audio_job 被引用）")
    check("review_scope" in ui_src and 'scope="trash"' in ui_src and 'scope="review"' in ui_src,
          "UI：音频复核范围选择（待清理/需人工确认分开展示与启动）")
    check("scope: str = \"all\"" in jobs_src and "scope=scope" in jobs_src,
          "jobs：音频复核任务支持范围参数（all/trash/review）")

    # 8) 复核范围过滤（trash 与 review 两类都覆盖，可分别/合并复核）
    from src.db import _scope_sql
    check(_scope_sql("all") == "IN ('trash','review')"
          and _scope_sql("trash") == "= 'trash'"
          and _scope_sql("review") == "= 'review'"
          and _scope_sql(None) == "IN ('trash','review')",
          "范围 → verdict SQL：all 含两类，trash/review 单档，非法回落 all")
    d2, db2, _, ids2 = _tmp_db(6)   # v0,v1 待复核各1；v2 已放行未复核；v3 已放行已复核
    try:
        for v, verdict in ((ids2[0], "trash"), (ids2[1], "review"),
                           (ids2[2], "trash"), (ids2[3], "trash")):
            db2.upsert_video_meta(db2.get_video(v)["path"], {
                "filename": db2.get_video(v)["filename"], "rel_dir": "", "year": "2024",
                "size_bytes": 1, "mtime": 0, "duration_sec": 60,
                "width": 1920, "height": 1080, "video_codec": "hevc",
                "has_audio": 1, "audio_energy": None, "bitrate_kbps": 1000,
                "meta_ok": 1, "meta_error": None})
            _mark_analyzed(db2, v, verdict, 5.0)
        check(db2.count_audio_pending() == 4 and db2.count_audio_pending(scope="all") == 4,
              "范围 all：trash+review 均计入待复核（默认全量覆盖）")
        check(db2.count_audio_pending(scope="trash") == 3
              and db2.count_audio_pending(scope="review") == 1,
              "范围拆分：trash/review 各自计数正确")
        q_all = {x["id"] for x in db2.audio_review_queue()}
        q_r = {x["id"] for x in db2.audio_review_queue(scope="review")}
        check(q_all == set(ids2[:4]) and q_r == {ids2[1]},
              f"队列过滤：all 含两类（{sorted(q_all)}）；review 只取需人工确认档")

        # 9) 队列放宽（ADR 0003）：已放行/已自动处置的片段仍能补复核 + 归档
        moved = d2 / "need-del"
        moved.mkdir(exist_ok=True)
        (moved / "v2.mp4").write_bytes(b"x")
        db2.set_human_decision(ids2[2], "trash",
                               trash_path=str(moved / "v2.mp4"), trash_tier="A")
        db2.update_audio_review(ids2[3], {"audio_checked": 1, "speech_present": 0})
        check(db2.count_audio_pending() == 3,
              "已放行但未复核的片段仍计入待复核（队列放宽后不再漏画外音）")
        check(ids2[2] in {x["id"] for x in db2.audio_review_queue()},
              "audio_review_queue 收录已移入回收目录的未复核片段")
        check(ids2[3] not in {x["id"] for x in db2.audio_review_queue()},
              "复核过的片段出队（无论是否已移入回收）")
        check(db2.count_audio_pending(scope="trash") == 2,
              "scope=trash：已放行的 review 档不被 verdict 挡掉（scope 只约束未处置半边）")

        # 10) 复核可按回收路径取源（源失效 + trash_path 在 → 不再误判源文件不存在）
        ar2 = AudioReviewer({"work_dir": str(d2), "audio": {"engine": "funasr"},
                             "llm": {"enabled": False}, "runtime": {}})
        with db2.conn() as c:      # has_audio=0 → 取到源后走"无音轨"分支，不触发引擎
            c.execute("UPDATE videos SET has_audio=0 WHERE id=?", (ids2[2],))
        r2 = ar2.review_one(db2, db2.get_video(ids2[2]))
        check(r2.get("skipped") == "无音轨",
              "review_one：源路径失效时回落到回收路径（不再误判源文件不存在）")
    finally:
        shutil.rmtree(d2, ignore_errors=True)

    # 9) fmt_eta 完整时长格式（用户要求不省略，metric 窄列截断时进度条文本保底）
    from src.ui import fmt_eta
    _eta_cases = [(0, "0秒"), (45, "45秒"), (90, "1分30秒"), (754, "12分34秒"),
                  (3600, "1小时00分"), (9000, "2小时30分"), (9375, "2小时36分"),
                  (86400, "1天0小时00分"), (93750, "1天2小时02分"),
                  (93750 + 86400 * 2, "3天2小时02分"), (-5, "—"), (None, "—")]
    for _s, _want in _eta_cases:
        assert fmt_eta(_s) == _want, f"fmt_eta({_s})={fmt_eta(_s)!r} 期望 {_want!r}"
    check(all(fmt_eta(s) == w for s, w in _eta_cases),
          f"fmt_eta 完整时长格式（{len(_eta_cases)} 档位全对，无省略）")

    # 10) VAD 漏检兜底（真实案例：2015 DVR 8kHz 音频有人说话，fsmn-vad 返回空段 → 0%）
    #     规则：VAD 空段时强制整段 ASR，有效字数 ≥ min_transcript_chars → 判有语音
    from src.audioreview import AudioReviewer as AR2, _effective_chars
    check(_effective_chars("你好啊啊，那里面还有一个证据。") == 13
          and _effective_chars("嗯。啊？") == 2 and _effective_chars("") == 0
          and _effective_chars("，。！？ ") == 0,
          "_effective_chars：去标点空白计有效字数")

    def _mk_ar(vad_segs, asr_text, **audio_extra):
        audio = {"engine": "funasr", "min_speech_sec": 1.0,
                 "vad_speech_threshold": 0.3, "min_transcript_chars": 4}
        audio.update(audio_extra)
        ar = AR2({"audio": audio, "llm": {"enabled": False}, "runtime": {}, "work_dir": "."})
        class _V:   # 模拟 fsmn-vad
            @staticmethod
            def generate(input, **kw):
                return [{"key": "k", "value": [list(s) for s in vad_segs]}]
        class _A:   # 模拟 paraformer
            @staticmethod
            def generate(input, **kw):
                return [{"key": "k", "text": asr_text}]
        ar._load_engine = lambda: (_V, _A, None)   # 覆盖真实引擎加载
        return ar

    wav = Path("unused.wav")   # mock 引擎不读文件
    r = _mk_ar([], "你好啊啊，那里面还有一个证据。").review_wav(wav, 43.5)
    check(r["speech_present"] == 1 and r["speech_ratio"] > 0
          and r["speech_segments"] == "[[0.0, 43.5]]",
          f"VAD 漏检兜底：ASR 有字 → 判有语音（present=1, ratio={r['speech_ratio']}）")
    r = _mk_ar([], "").review_wav(wav, 43.5)
    check(r["speech_present"] == 0 and r["speech_ratio"] == 0.0,
          "VAD 漏检兜底：ASR 也无字 → 维持无人声")
    r = _mk_ar([], "嗯，啊。").review_wav(wav, 43.5)   # 有效字数 2 < 4
    check(r["speech_present"] == 0 and r["speech_ratio"] == 0.0,
          "VAD 漏检兜底：ASR 仅语气词（低于 min_transcript_chars）→ 无人声")
    r = _mk_ar([[1000, 9000]], "测试转写内容").review_wav(wav, 43.5)
    check(r["speech_present"] == 1 and abs(r["speech_ratio"] - 8.0 / 43.5) < 0.01
          and r["speech_segments"] == "[[1.0, 9.0]]",
          "VAD 正常检出路径不受兜底影响（按段计时长）")

    # 10.1) 三级流水线第 2 级：原始 VAD 空 → 降噪副本再 VAD → 按段 ASR
    def _mk_ar2(vad_map, asr_text, **audio_extra):
        audio = {"engine": "funasr", "min_speech_sec": 1.0,
                 "vad_speech_threshold": 0.3, "min_transcript_chars": 4}
        audio.update(audio_extra)
        ar = AR2({"audio": audio, "llm": {"enabled": False}, "runtime": {}, "work_dir": "."})
        class _V:
            @staticmethod
            def generate(input, **kw):
                return [{"key": "k", "value": [list(s) for s in vad_map.get(input, [])]}]
        class _A:
            @staticmethod
            def generate(input, **kw):
                return [{"key": "k", "text": asr_text}]
        ar._load_engine = lambda: (_V, _A, None)
        ar._denoise_wav = lambda w: w.with_name(w.stem + ".dsp.wav")
        return ar

    orig, dspf = Path("unused.wav"), Path("unused.dsp.wav")
    ar2 = _mk_ar2({str(orig): [], str(dspf): [[2000, 12000]]}, "降噪后检出的语音")
    r = ar2.review_wav(orig, 20.0)
    check(r["speech_present"] == 1 and abs(r["speech_ratio"] - 0.5) < 0.01
          and r["speech_segments"] == "[[2.0, 12.0]]"
          and ar2.last_dsp_wav == dspf,
          "三级流水线 L2：原始 VAD 空 → 降噪副本 VAD 检出 → 对副本按段 ASR")
    # dsp_filter 置空 → 跳过第 2 级直接整段兜底（用原始 wav）
    ar3 = _mk_ar({str(orig): [], str(dspf): []}, "整段兜底转写", dsp_filter="")
    ar3._denoise_wav = lambda w: None
    r = ar3.review_wav(orig, 20.0)
    check(r["speech_present"] == 1 and r["speech_segments"] == "[[0.0, 20.0]]"
          and ar3.last_dsp_wav is None,
          "dsp_filter 置空：跳过降噪级，直接整段 ASR 兜底")


# --------------------------------------------------------------------------- 12
def test_archive():
    """L2.7 语音档案归档：同盘提示 / 谓词 / 配对 / 幂等 / 撤销不回收。"""
    section("语音档案归档")
    from src import archiver as A

    # 1) 同盘只提示不禁止：删除是手动操作，用户自己可控，决定权留给人
    same = {"trash_root": r"G:\need-del", "source_root": r"G:\change\2015",
            "archive": {"dir": r"G:\audio-archive"}}
    p, w = A.check_ready(same), A.check_warnings(same)
    check(p == [], f"同盘不再是硬错误，允许执行（硬错误 {len(p)} 条）")
    check(len(w) == 2, f"同盘给出 2 条软提示（{len(w)} 条）")
    check(A.check_ready({"trash_root": r"G:\need-del", "source_root": r"G:\change\2015",
                         "archive": {"dir": r"E:\audio-archive"}}) == [],
          "异盘归档目录 → 检查通过")
    check(A.check_warnings({"trash_root": r"G:\need-del", "source_root": r"G:\change\2015",
                            "archive": {"dir": r"E:\audio-archive"}}) == [],
          "异盘归档目录 → 无提示")
    check(A.check_ready({"trash_root": "", "source_root": "",
                         "archive": {"dir": r"E:\audio-archive"}}) == [],
          "未配置源/回收目录时不误报")
    # 真正的硬错误仍然拦截（含 G:\ 这种盘符根，不能只比 "/"）
    check(len(A.check_ready({"trash_root": "", "source_root": "",
                             "archive": {"dir": "G:\\"}})) == 1,
          "目录为盘符根目录 G:\\ → 硬错误拦截")
    check(len(A.check_ready({"trash_root": "", "source_root": "",
                             "archive": {"dir": "/"}})) == 1,
          "目录为 / → 硬错误拦截")
    check(A.is_drive_root("G:\\") and A.is_drive_root("/") and A.is_drive_root(""),
          "is_drive_root 识别 G:\\ / 空")
    check(not A.is_drive_root(r"G:\audio-archive") and not A.is_drive_root("audio"),
          "is_drive_root 不误伤普通目录")

    d, db, bid, ids = _tmp_db(4)
    try:
        # 造 4 种情况：
        # v0 已入回收目录 + 有效语音 + 缓存 wav 在   → 归档成功
        # v1 有效语音但未移动（无 trash_path）       → 新谓词下仍是处置候选，
        #                                             可归档（画外音盲区兜底，ADR 0001）
        # v2 已移动但无人声                          → 不入队
        # v3 已移动 + 有效语音但缓存 wav 缺失        → 入队后跳过
        def _prep(vid, moved: bool, speech: bool):
            db.upsert_video_meta(db.get_video(vid)["path"], {
                "filename": db.get_video(vid)["filename"], "rel_dir": "", "year": "2024",
                "size_bytes": 1, "mtime": 0, "duration_sec": 60,
                "width": 1920, "height": 1080, "video_codec": "hevc",
                "has_audio": 1, "audio_energy": None, "bitrate_kbps": 1000,
                "meta_ok": 1, "meta_error": None})
            _mark_analyzed(db, vid, "trash", 5.0)
            db.update_audio_review(vid, {
                "speech_present": 1 if speech else 0, "speech_ratio": 0.3 if speech else 0.0,
                "speech_segments": "[[1.0,3.0]]", "transcript_text": "测试转写内容" if speech else "",
                "transcript_segments": "[]", "audio_checked": 1})
            if moved:
                db.set_human_decision(vid, "trash",
                                      trash_path=str(d / "need-del" / f"v{vid}.mp4"),
                                      trash_tier="A")

        _prep(ids[0], True, True)
        _prep(ids[1], False, True)
        _prep(ids[2], True, False)
        _prep(ids[3], True, True)

        # 缓存 wav 只给 v0 / v1 造
        aud = d / "audio"
        aud.mkdir(exist_ok=True)
        (aud / f"{ids[0]}.wav").write_bytes(b"RIFF" + b"\x00" * 100)
        (aud / f"{ids[1]}.wav").write_bytes(b"RIFF" + b"\x00" * 100)

        q = {r["id"] for r in db.archive_queue()}
        check(q == {ids[0], ids[1], ids[3]},
              f"队列谓词：已入回收∨未处置判删候选，∧有效语音∧未归档（{sorted(q)}）")
        check(db.count_archive_pending() == 3 and db.count_archived() == 0,
              "计数：待归档 3 / 已归档 0（含未移入回收的 review 类候选）")

        cfg = {"work_dir": str(d), "trash_root": "", "source_root": "",
               "archive": {"dir": str(d / "arch")}}
        res = A.archive_pending(db, cfg, limit=100)
        check(res["done"] == 2 and res["skipped"] == 1 and res["failed"] == 0,
              f"归档结果：成功 {res['done']}（v0+v1）/ 缓存缺失跳过 {res['skipped']} / 失败 0")

        # 文件名 wav+txt 配对，txt 内容 = 数据库转写
        wav = Path(d) / "arch" / "2024" / f"{db.get_video(ids[0])['filename'][:-4]}.wav"
        txt = wav.with_suffix(".txt")
        check(wav.exists() and txt.exists(), "归档产物：wav + 同名 txt 配对存在")
        check(txt.read_text(encoding="utf-8") == "测试转写内容",
              "txt 内容与数据库 transcript_text 一致")

        mf = Path(d) / "arch" / "manifest.jsonl"
        lines = mf.read_text(encoding="utf-8").strip().splitlines()
        rec = json.loads(lines[0])
        check(len(lines) == 2 and rec["id"] == ids[0]
              and rec["transcript"] == "测试转写内容"
              and rec["trash_path"].endswith(".mp4"),
              "manifest.jsonl：v0 一行含 id/转写/回收路径（共 2 行）")
        check(json.loads(lines[1])["id"] == ids[1]
              and json.loads(lines[1])["trash_path"] is None,
              "manifest.jsonl：v1（未移动）也已归档，trash_path 为空（数据源是缓存 wav）")
        check((Path(d) / "arch" / "2024" / "v1.wav").exists(),
              "归档产物：未移入回收的候选同样产出 wav")

        # 幂等：再跑一次不新增
        res2 = A.archive_pending(db, cfg, limit=100)
        lines2 = mf.read_text(encoding="utf-8").strip().splitlines()
        check(res2["done"] == 0 and len(lines2) == 2,
              "幂等：重复执行不重复归档、manifest 不追加")
        check(db.count_archived() == 2, "已归档计数 = 2（v0+v1，archived_at 标记）")

        # 撤销移动 → 退出处置候选，但已归档文件保留（档案独立，宁存勿删）
        db.set_human_decision(ids[0], "keep")
        check(db.count_archive_pending() == 1
              and ids[0] not in {r["id"] for r in db.archive_queue()},
              "撤销移动后退出归档队列")
        check(wav.exists() and txt.exists(), "撤销移动不回收已生成的档案文件")
    finally:
        shutil.rmtree(d, ignore_errors=True)

    # 3) 迁移幂等：archived_at 列存在
    _d2, db2, _, _ = _tmp_db(1)
    try:
        cols = {r["name"] for r in _pcol(db2)}
        check("archived_at" in cols, "DB 迁移：videos 已补齐 archived_at 列")
    finally:
        shutil.rmtree(_d2, ignore_errors=True)

    # 4) UI / jobs 接线（源码级检查，防改后端忘接线）
    ui_src = (Path(__file__).resolve().parent.parent / "src" / "ui.py").read_text(encoding="utf-8")
    jobs_src = (Path(__file__).resolve().parent.parent / "src" / "jobs.py").read_text(encoding="utf-8")
    check("立即归档有效语音" in ui_src and "count_archive_pending" in ui_src,
          "UI：设置页已挂载语音档案卡与归档计数")
    check('jobsmod.get_archive_job()' in ui_src, "UI：进度页接入归档任务面板")
    check('_ARCHIVE = "archive"' in jobs_src and "start_archive" in jobs_src,
          "jobs：归档任务类型已注册")


def _pcol(db):
    """取 videos 列名集合（兼容 Database / sqlite3 连接两种形态）。"""
    conn = getattr(db, "db_path", None)
    import sqlite3
    c = sqlite3.connect(conn) if conn else db
    try:
        cur = c.execute("PRAGMA table_info(videos)")
        return [{"name": r[1]} for r in cur]
    finally:
        if conn:
            c.close()


def test_grid_audio_text():
    """音频转写摘要 HTML 的四种状态渲染（2026-09-07 引入，2026-10-04 换宿主）。

    场景：画面被判删的片段，画外可能有人声事件。转写摘要必须一眼可见 ——
    让人逐条点进详情才看到「说了什么」，等于把最关键的信息藏起来。

    ⚠️ 宿主已换：原实现挂在「批量浏览」九宫格上，该页已随全量审核导航一起删除
    （ADR 0007）。`_grid_audio_html` 本身保留 —— 它的 CSS class 与四种状态分支
    是可复用的资产，抽查页与未来的列表视图还会用到。下面的源码级断言改成
    **反向**断言：钉住"旧的九宫格宿主不再存在"。
    """
    section("音频转写摘要：四种状态渲染")

    ui_src = Path("src/ui.py").read_text(encoding="utf-8")
    check("def tab_grid" not in ui_src,
          "「批量浏览」页已删除（程序全驱动后只保留抽查）")
    check("_grid_audio_html" in ui_src,
          "_grid_audio_html 保留为可复用资产（不是随宿主一起删）")
    check("_GRID_AUDIO_CSS" in ui_src, "格子音频摘要样式仍在")

    from src.ui import _grid_audio_html

    # 2) 四种状态的渲染分支
    d, db, bid, ids = _tmp_db(5)
    try:
        for i in ids:
            _mark_analyzed(db, i)
        # _tmp_db 造的是无音轨记录，这里统一标成有音轨（否则音频行不渲染）
        with db.conn() as c:
            c.execute("UPDATE videos SET has_audio=1")

        def _mk(vid, **kw):
            db.update_audio_review(vid, kw)
            return db.get_video(vid)

        # 无人声
        r0 = _mk(ids[0], speech_present=0, speech_ratio=0.0, audio_checked=1)
        h0 = _grid_audio_html(r0)
        check("无人声" in h0 or h0 == "", "无人声：显示「无人声」或不占版面")
        check(bool(r0["has_audio"]) is False or "g-aud-none" in h0,
              "无人声用弱化样式（g-aud-none）")

        # 有人声 + 转写
        r1 = _mk(ids[1], speech_present=1, speech_ratio=0.257, audio_checked=1,
                 transcript_text="你是干嘛来啊？")
        h1 = _grid_audio_html(r1)
        check("g-aud-sp" in h1 and "26%" in h1 and "你是干嘛来啊？" in h1,
              "有人声：高亮样式 + 占比 + 转写文本都在格子里")
        check("你是干嘛来啊？" in h1.split("title=")[-1][:80],
              "转写全文挂 title（hover 可看完整内容）")

        # 未复核
        r2 = db.get_video(ids[2])
        h2 = _grid_audio_html(r2)
        check("未音频复核" in h2, f"未复核：提示「未音频复核」（got {h2[:40]!r}）")

        # HTML 转义（转写里含尖括号/引号不能破坏结构）
        r3 = _mk(ids[3], speech_present=1, speech_ratio=0.5, audio_checked=1,
                 transcript_text='<script>"x"</script>')
        h3 = _grid_audio_html(r3)
        check("<script>" not in h3 and "&lt;script&gt;" in h3,
              "转写内容做 HTML 转义（防注入/破版）")

        # 3) 仅看检出人声：SQL 层过滤，分页计数与结果一致
        db.update_audio_review(ids[4], {"speech_present": 1, "speech_ratio": 0.4,
                                        "audio_checked": 1, "transcript_text": "喂"})
        n_all = db.count_review_queue()
        n_sp = db.count_review_queue(speech_only=True)
        q_sp = db.review_queue(limit=100, speech_only=True)
        check(n_sp < n_all and n_sp >= 1,
              f"speech_only 计数生效（全部 {n_all} → 有人声 {n_sp}）")
        check(len(q_sp) == n_sp, "speech_only 分页计数与实际取数一致")
        check(all(r["speech_present"] == 1 for r in q_sp),
              "speech_only 结果集只含检出人声的行")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_grid_video_play():
    """批量浏览调系统播放器（2026-09-07）：静态缩略图不足以判断画面，必须能看动态。

    红线复核：这些入口只能"打开"，绝不能删除/移动原文件。
    """
    section("系统播放器入口（路径探测）")

    # ⚠️ 源码接线段改成反向断言：宿主「批量浏览」已删（ADR 0007），
    # 下面的**行为**段（路径探测）才是真正要守的东西 —— 源/回收/两边都没有/
    # 两边都在，四种情况的行为不能退化。
    ui_src = Path("src/ui.py").read_text(encoding="utf-8")
    check("def tab_grid" not in ui_src,
          "「批量浏览」页已删除（播放器入口现存于抽查页与精华复核页）")
    check("open_in_player" in ui_src and "reveal_in_explorer" in ui_src,
          "系统播放器 / 资源管理器定位入口仍在（换宿主不丢功能）")
    # 红线：这两个入口只能"打开"，绝不能移动/删除原文件
    check("shutil.move" not in ui_src and "shutil.rmtree" not in ui_src,
          "★ui.py 里没有移动/删除调用（红线：入口只能打开）")

    from src.ui import current_video_path, open_in_player, reveal_in_explorer

    # 路径探测：源 / 回收 / 都没有 / 两边都在
    con = sqlite3.connect(":memory:")
    con.row_factory = sqlite3.Row
    d = Path(tempfile.mkdtemp(prefix="vs_play_"))
    try:
        src = d / "a.mp4"; src.write_bytes(b"x")
        trash = d / "b.mp4"; trash.write_bytes(b"x")

        def _row(p, t):
            return con.execute("SELECT ? AS path, ? AS trash_path",
                               (p, t)).fetchone()

        check(current_video_path(_row(str(src), None)) == src,
              "未移动：用源路径")
        check(current_video_path(_row(str(d / "gone.mp4"), str(trash))) == trash,
              "已移入回收目录：源路径失效后自动回退回收路径")
        check(current_video_path(_row(str(d / "gone.mp4"), None)) is None,
              "两边都不存在：返回 None（由调用方报错）")
        check(current_video_path(_row(str(src), str(trash))) == src,
              "两边都在：优先源路径")

        # 红线：这些入口只读打开，不得删除
        import inspect
        for fn in (open_in_player, reveal_in_explorer):
            body = inspect.getsource(fn)
            check(not any(w in body for w in ("unlink", "remove(", "rmtree", "shutil")),
                  f"{fn.__name__} 不含任何删除调用")
    finally:
        shutil.rmtree(d, ignore_errors=True)
        con.close()


def test_rerun_analysis():
    """改了判定参数后重新分析（2026-09-07）：把已分析记录退回待分析。

    根因：分析队列只取 status IN ('metadata','pending')，analyzed/keep/trash 都不在
    其中。改完 YOLO 阈值直接点「启动 AI 分析」→ 0 条待分析 → 看起来像坏了。
    """
    section("改参数后重新分析")

    d, db, bid, ids = _tmp_db(6)
    try:
        # 0,1 已分析未处置 / 2,3 分析失败 / 4,5 已人工处置(keep/trash)
        for i in (0, 1):
            db.update_analysis(ids[i], {"status": "analyzed", "retention_score": 10,
                                        "verdict": "trash", "decision_trace": {"x": 1},
                                        "final_score": 10, "has_person": 0})
        for i in (2, 3):
            db.update_analysis(ids[i], {"status": "error", "analyze_error": "解码超时"})
        db.set_human_decision(ids[4], "keep")
        db.set_human_decision(ids[5], "trash", trash_path=str(d / "x.mp4"),
                              trash_tier="A")

        check(db.count_resettable("undecided") == 4,
              f"未人工处置可退回 4 条（2 已分析 + 2 失败），实际 "
              f"{db.count_resettable('undecided')}")
        check(db.count_resettable("failed") == 2, "仅失败项 2 条")
        check(db.count_resettable("all") == 6, "全部 6 条")

        n = db.reset_analysis("undecided")
        check(n == 4, f"退回 4 条，实际 {n}")
        r0 = db.get_video(ids[0])
        check(r0["status"] == "metadata" and r0["retention_score"] is None
              and r0["verdict"] is None and r0["decision_trace"] is None,
              "分析结果已清空（分数/判定/依据链），状态回到 metadata")
        check(len(db.iter_pending_analysis(limit=100)) == 4,
              "退回后重新进入待分析队列 —— 这才是重跑生效的关键")
        check(db.count_resettable("undecided") == 0, "幂等：再无可退回项")

        # 人工处置的不受影响
        r4, r5 = db.get_video(ids[4]), db.get_video(ids[5])
        check(r4["human_decision"] == "keep" and r5["human_decision"] == "trash",
              "默认范围不动人工处置决定（keep / trash 均保留）")
        check(r5["trash_path"] is not None, "已移入回收目录的 trash_path 不被清空")

        # failed 档：只重试错误项
        check(db.reset_analysis("failed") == 0,
              "undecided 已把 error 一并退回，failed 档此时为 0")

        try:
            db.reset_analysis("bogus")
            check(False, "非法 scope 应抛 ValueError")
        except ValueError:
            check(True, "非法 scope 抛 ValueError（防止误传字符串静默全表重置）")
    finally:
        shutil.rmtree(d, ignore_errors=True)

    # 红线：重置只能 UPDATE 清字段，绝不能删行删表
    import inspect
    from src import db as dbmod
    body = inspect.getsource(dbmod.Database.reset_analysis)
    check(not any(w in body for w in ("DELETE", "DROP", "unlink", "rmtree")),
          "reset_analysis 不含 DELETE/DROP/文件删除（只 UPDATE 字段）")

    ui_src = Path("src/ui.py").read_text(encoding="utf-8")
    seg = ui_src[ui_src.index("🔄 改了判定参数？重新分析"):]
    seg = seg[:seg.index("🔊 音频复核")]
    check('st.button("↩ 退回待分析"' in seg, "设置页提供「退回待分析」按钮")
    check('st.button("🔄 退回并立即重跑分析"' in seg, "提供「退回并立即重跑」一键入口")
    check('key="reset_scope"' in seg and '"undecided"' in seg,
          "范围单选默认 undecided（不动人工结论）")
    check('key="reset_all_ok"' in seg, "全量退回需二次确认勾选")
    check("db.count_resettable(" in seg, "按钮前显示各范围条数")


def test_review_queue_selfheal():
    """审核队列对"已重置记录"的自愈（2026-09-07 真实崩溃修复）。

    崩溃链路：审核队列 id 缓存在 session_state → 用户「退回并立即重跑分析」把
    那批记录的 verdict/score 清空 → 页面 rerun 用旧 id 取行 → f"{None:g}" TypeError。
    修复：渲染前校验 verdict 仍在审核范围，失效则整页重载（自愈收敛，不会循环）；
    render_reasons 对 None 分数容错。
    """
    section("审核队列 过期自愈")

    ui_src = Path("src/ui.py").read_text(encoding="utf-8")
    seg = ui_src[ui_src.index("def render_reasons") + 4:]
    seg = seg[:seg.index("def ")]
    check('if score is None and verdict is None' in seg,
          "render_reasons：分数与判定均为空时给出重置提示而非崩溃")
    check('f"{score:g}" if score is not None else "—"' in seg,
          "render_reasons：None 分数显示 —（指标不格式化 None）")

    seg2 = ui_src[ui_src.index("def tab_review"):]
    seg2 = seg2[:seg2.index("def tab_", 5)] if "def tab_" in seg2[5:] else seg2
    # 队列 id 缓存 + "_q_reload" 自愈机制随全量审核队列一起退役（v3 阶段 4）：
    # 抽查页的批次是**当场抽出来的一次性列表**，不缓存在 session_state 里
    # 跨 rerun 复用，所以"条目过期"这个失效模式本身不存在了。
    check("q_reload" not in ui_src and "_q_reload" not in ui_src,
          "队列自愈机制已退役（抽查批次不跨 rerun 缓存，无过期可愈）")
    check('row["verdict"] not in ("trash", "review")' not in seg2,
          "旧的 verdict 范围校验也随之移除")
    # 抽查页的等价保护：抽中的行取不到时必须优雅退出而不是 KeyError
    check("if not row:" in seg2 and 'st.error("记录不存在")' in seg2,
          "抽查页：行取不到时优雅退出（等价保护）")

    # 行为级：把一条已被重置（verdict IS NULL）的 id 塞进抽查批次，页面必须不崩。
    #
    # ⚠️ 这里**不用项目真实库**，改用一次性临时库（靠 ui.main() 的
    # VIDEO_SEARCH_DB_OVERRIDE 钩子注入路径）。自检脚本必须能在
    # "data/ 尚未生成"时跑完 —— 新机首次装 / 库被清 / CI 都会遇到。
    # 2026-10-04 真实库丢失时，这段直接 sqlite3.OperationalError 崩掉整个自检，
    # 后面一百多项根本没执行。自检依赖真实库是设计错误，不是环境问题。
    from streamlit.testing.v1 import AppTest
    tmp_dir = Path(tempfile.mkdtemp(prefix="selftest-selfheal-"))
    old_override = os.environ.get("VIDEO_SEARCH_DB_OVERRIDE")
    try:
        tmp_db = str(tmp_dir / "probe.db")
        with Database(tmp_db).conn() as c:
            c.execute("INSERT INTO videos (path, filename, verdict, meta_ok) "
                      "VALUES ('dead.mp4','dead.mp4',NULL,1)")
        os.environ["VIDEO_SEARCH_DB_OVERRIDE"] = tmp_db
        at = AppTest.from_file(str(Path("src/ui.py").resolve()), default_timeout=240)
        at.run()
        check(not at.exception,
              "空库（抽不到样本）时页面无异常 —— 抽查池为空是**正常状态**"
              "（还没跑分析 / 口径不匹配），必须优雅提示而不是崩")
        check(any("抽样池" in getattr(w, "value", "") for w in at.warning),
              "抽样池为空时明确告知（机主能分辨是'没数据'还是'配置错了'）")
    finally:
        if old_override is None:
            os.environ.pop("VIDEO_SEARCH_DB_OVERRIDE", None)
        else:
            os.environ["VIDEO_SEARCH_DB_OVERRIDE"] = old_override
        shutil.rmtree(tmp_dir, ignore_errors=True)  # AUDIT-OK: 自创临时目录


def test_review_jump_nav():
    """审核台 / 批量浏览的「跳转 + 筛选」（ADR 0002 配套改动）回归。

    要守住三件事：
      1. 队列、计数、跳转定位共用同一组谓词 —— 任何一处单独拼 SQL 都会出现
         「计数说 300 条、翻到第 2 页却空」（count_audio_pending 就这样错过一次）；
      2. 「跳到分数 ≥X」等价于「数出低于 X 的条数」，不是 UI 自己扫队列；
      3. 换筛选条件后必须从第一条重载，越界跳转要被夹住而不是崩。
    """
    section("审核台 / 批量浏览：跳转与筛选")

    # ---------- A. db 层：计数与队列必须逐字同口径 ----------
    d, db, bid, ids = _tmp_db(8)
    try:
        for i, vid in enumerate(ids):
            db.update_analysis(vid, {"status": "analyzed", "retention_score": i * 5,
                                     "final_score": i * 5, "verdict": "trash",
                                     "has_person": 1 if i >= 6 else 0,
                                     "decision_trace": {}})
        for f in ({}, {"person_only": True}, {"min_score": 15},
                  {"min_score": 15, "person_only": True}):
            n = db.count_review_queue(**f)
            rows = db.review_queue(limit=999, **f)
            check(n == len(rows), f"计数与队列同口径 {f or '（无筛选）'}：{n} 条")
        check(db.count_review_queue(person_only=True) == 2
              and all(r["has_person"] for r in db.review_queue(limit=99, person_only=True)),
              "仅看检出人物：只回 has_person=1 的片段")
        for x in (0, 12, 15, 35, 100):
            exp = len([r for r in db.review_queue(limit=999)
                       if (r["final_score"] or 0) < x])
            check(db.count_review_below(x) == exp,
                  f"按分数定位的口径 x={x} → 位置 {db.count_review_below(x)}")
        check(db.count_review_below(100) == 8,
              "跳到分数 ≥100（无人达标）指向队列之后，由跳转逻辑夹回末尾")

        # ---------- A2. 谓词参数收口（v3 阶段 2）----------
        # max_score 是死参数、only_unreviewed 语义已反转，两者的形参都应从
        # 签名里消失。留着它们会诱导调用点以为"能连已处置的一起看"——
        # 而项目已程序驱动，抽查要看的正是已处置的结果。
        import inspect
        from src import db as DBMOD
        for fn in (DBMOD.Database._review_sql, DBMOD.Database.review_queue,
                   DBMOD.Database.count_review_queue,
                   DBMOD.Database.count_review_below):
            params = set(inspect.signature(fn).parameters)
            check("max_score" not in params,
                  f"{fn.__name__} 已删死参数 max_score")
            check("only_unreviewed" not in params,
                  f"{fn.__name__} 已删 only_unreviewed（语义已反转）")
    finally:
        shutil.rmtree(d, ignore_errors=True)

    # ---------- A3. 已处置的行退出审核队列，但放行队列不受影响 ----------
    # 独立 try 块：**这组断言会改库状态**（置 human_decision），放在上面那个
    # 块里会让后续断言的基数凭空变化 —— 真实踩过（count_review_below 从 8 变 7，
    # 报出三个"失败"）。
    d3, db3, _b3, ids3 = _tmp_db(4)
    try:
        for vid in ids3:
            db3.update_analysis(vid, {"status": "analyzed", "retention_score": 5,
                                      "final_score": 5, "verdict": "trash",
                                      "has_person": 0, "decision_trace": {}})
        with db3.conn() as c:
            c.execute("UPDATE videos SET human_decision='trash' WHERE id=?",
                      (ids3[0],))
        check(ids3[0] not in [r["id"] for r in db3.review_queue(limit=99)],
              "已处置的行不进审核队列（human_decision IS NULL 无条件生效）")
        check(ids3[0] not in {r["id"] for r in
                              db3.release_queue(limit=99, threshold=30,
                                                require_no_person=False)},
              "★放行队列仍只认未处置（release_queue 的独立谓词没被收口误伤）")
    finally:
        shutil.rmtree(d3, ignore_errors=True)

    # ---------- B. UI 接线：全量审核导航已退役（反向断言）----------    # ADR 0005 的"跳到第 N 条 / 跳到分数 ≥X"是**全量人工审核时代**的导航。
    # 程序全驱动之后人工只抽 N 条看，"跳到第 1500 条"没有使用场景 ——
    # 连同批量浏览整页一起删除（ADR 0007）。保留反向断言防止有人又加回来：
    # 那套导航一旦回来，就意味着"人工要逐条过"又回来了。
    ui_src = Path("src/ui.py").read_text(encoding="utf-8")
    for gone, why in (("def _queue_jump", "条号/分数跳转"),
                      ("def _queue_ensure", "队列游标管理"),
                      ("def _queue_sig", "筛选指纹"),
                      ("def tab_grid", "批量浏览页"),
                      ("rv_jump_idx", "条号跳转控件"),
                      ("rv_jump_score", "分数跳转控件"),
                      ("grid_jump_page", "九宫格页码跳转")):
        check(gone not in ui_src, f"{why}已退役（{gone} 不存在）")

    # ---- C. 抽查页替代了它 ----
    check("def _spot_l1" in ui_src and "def _spot_clips" in ui_src,
          "抽查页两个入口就位（L1 判定 / 已剪 A-B 精华）")
    check("def _spot_sample_bar" in ui_src and "_spot_reroll" in ui_src,
          "抽样工具条 + 换一批（种子+1，可回退）")
    check("sample_ids" in ui_src and "sample_ab_clips" in ui_src,
          "抽样走 db 层的可复现方法（不在 UI 里自己抽）")

    # ---- D. 口径分叉的钉死（v3 阶段 2 留下）----
    # ADR 0005 承诺"队列 / 计数 / 跳转三处逐字一致"。那套 UI 删了，但**原则留下**：
    # 凡是"显示条数"的地方，都必须与"实际显示的集合"同口径。
    # 现在抽查页承担这个角色 —— 抽样池的规模与实际抽出的样本必须一致。
    empty_parens = [ln.strip() for ln in ui_src.splitlines()
                    if "count_review_queue()" in ln]
    check(not empty_parens,
          f"无任何 count_review_queue() 空参调用（分母必须带筛选）：{empty_parens}")
    check("count_sample_pool" in ui_src,
          "抽查页用 count_sample_pool 报池规模（与 sample_ids 同谓词）")
def test_coarse_suspect_kept():
    """Track C · C1（ADR 0006）+ hlv3：粗扫看到的检出要**一律**降级保留，不能丢。

    这是本次改动里最值钱的一条 —— 旧口径把这类检出直接扔掉，导致 474 条视频
    "粗扫明明看到过目标、最终零产物"，原片删掉后一个字都不留。

    hlv3 又补了一处：hlv2 里"该窗没被精扫到"还额外要求**有动静**
    （`c_diffs > fb_thr`）才保留，于是"人坐着、只有小动作"（动静低于阈值）
    第二次被丢掉 —— 实测 621 条确有真人的画面因此变成零产物。现在**检出必留**，
    动静只影响"这个窗要不要送去精扫"，不再是"要不要留证"的门槛。
    """
    section("Track C · C1 粗扫疑似降级保留（hlv3：检出必留）")
    from src.highlighter import LV_CONFIRMED, LV_SUSPECT, merge_two_levels

    # 粗扫在 t=10 检出一个人（落在已精扫的窗内），精扫没确认 → f_hits 为空
    c_hits = [{"t": 10.0, "conf": 0.20, "n_person": 1, "max_area": 0.05,
               "level": LV_SUSPECT, "motion_only": False}]
    scanned = [(8.0, 12.0)]

    kept, n_down = merge_two_levels(c_hits, [], scanned, fine_on=True,
                                    keep_coarse_suspect=True)
    check(n_down == 1 and len(kept) == 1 and kept[0]["level"] == LV_SUSPECT,
          f"粗扫有目标、精扫没确认 ⇒ 降级保留为 C 疑似（{len(kept)} 条）")

    dropped, n_down2 = merge_two_levels(c_hits, [], scanned, fine_on=True,
                                        keep_coarse_suspect=False)
    check(not dropped and n_down2 == 0,
          "keep_coarse_suspect=False 退回旧行为（丢弃）—— 开关能一键回到改动前")

    # 精扫确认的照常进（升级权还在）
    f_hits = [{"t": 10.0, "conf": 0.31, "n_person": 1, "max_area": 0.08,
               "level": LV_CONFIRMED, "motion_only": False}]
    conf, _ = merge_two_levels(c_hits, f_hits, scanned, fine_on=True,
                              keep_coarse_suspect=True)
    check(any(h["level"] == LV_CONFIRMED for h in conf),
          "精扫确认的仍按确认级保留（精扫是升级权，不是否决权）")

    # 超预算没精扫到的窗（t=100 不在 scanned 里）：检出直接留证，不再要求有动静
    far = [{"t": 100.0, "conf": 0.2, "n_person": 1, "max_area": 0.05,
            "level": LV_SUSPECT, "motion_only": False}]
    got, _ = merge_two_levels(far, [], scanned, fine_on=True,
                              keep_coarse_suspect=True)
    check(len(got) == 1 and got[0]["level"] == LV_SUSPECT,
          "未精扫的窗：检出照样留证（不再要求「有动静」—— hlv2 的第二个漏点）")

    # 精扫关掉时（fine_on=False）粗扫检出原样返回 —— 合并函数不在这里二次过滤
    quiet = [{"t": 100.0, "conf": 0.2, "n_person": 1, "max_area": 0.05,
              "level": LV_SUSPECT, "motion_only": False}]
    got2, _ = merge_two_levels(quiet, [], scanned, fine_on=False,
                               keep_coarse_suspect=True)
    check(len(got2) == 1, "fine_on=False 时粗扫检出原样返回（不丢）")


def test_isolated_hit_not_dropped():
    """hlv3：孤立检出不能被 min_duration 当"一闪而过的误检"吃掉。

    min_duration 的本意是剪掉**纯动静**的碎屑。但 hits 里混着"检测器确实报出了
    目标"的帧（motion_only=False），一起按时长丢弃会让"人坐着不动、只在一帧被
    检出"变成**零产物** —— 而零产物在回收目录里等于"可直接删"。实测这类让 621 条
    确有真人的画面被判成可删，其中 489 条已经在回收目录里。
    """
    section("hlv3 · 孤立检出不被 min_duration 吞掉")
    from src.highlighter import LV_SUSPECT, merge_events

    one = [{"t": 10.0, "conf": 0.2, "n_person": 1, "max_area": 0.05,
            "level": LV_SUSPECT, "motion_only": False}]
    ev = merge_events(one, 40.0, min_duration=1.0)
    check(len(ev) == 1 and ev[0]["grade"] == "C",
          f"单帧**检出** ⇒ 仍出一个 C 事件（出了 {len(ev)} 个）")
    check(ev and ev[0]["dur"] >= 1.0,
          f"前后缓冲把单帧撑成可回放的段（{ev[0]['dur'] if ev else 0}s）")

    mo = [dict(one[0], motion_only=True, conf=0.0)]
    check(merge_events(mo, 40.0, min_duration=1.0) == [],
          "单帧**纯动静**仍按旧规则丢弃（不是把时长判断一律取消）")

    m2 = [dict(one[0], motion_only=True), dict(one[0], t=11.0, motion_only=True)]
    check(merge_events(m2, 40.0, min_duration=5.0) == [],
          "对纯动静组，min_duration 照样生效（开关没被废掉）")


def test_hl_version_single_source():
    """hlv 版本号必须只有一个值 —— 三处真源漂移会让回收目录永远归不了零。

    剪辑器**写库**用的是 `config.highlight.version`（或 DEFAULT_CONFIG 兜底），
    重剪队列与「待保全」判据读的是 `db.HIGHLIGHT_TARGET_VERSION`，
    而 `highlighter.HL_VERSION` 是代码里的默认值。三者不一致的后果是单向且
    不可见的：config 比 db 新 ⇒ 每条新剪的行都被判"版本落后 ⇒ 待保全"，
    回收目录永不归零；db 比 config 新 ⇒ 该重剪的行不重剪，原片删后精华永失。
    """
    section("hlv 版本号单一真源（四处必须相等）")
    from src.config import DEFAULT_CONFIG, load_config
    from src.highlighter import HL_VERSION
    import src.db as DBmod

    got = {
        "highlighter.HL_VERSION": HL_VERSION,
        "db.HIGHLIGHT_TARGET_VERSION": DBmod.HIGHLIGHT_TARGET_VERSION,
        "config.json": str((load_config().get("highlight") or {}).get("version") or ""),
        "DEFAULT_CONFIG": str(DEFAULT_CONFIG["highlight"].get("version") or ""),
    }
    vals = set(got.values())
    check(len(vals) == 1, f"四处版本号一致（实测 {got}）")


def test_c_cut_mp4():
    """Track C · C2/C4：C 级也要产出**可播放**的 mp4 与多张代表帧。"""
    section("Track C · C2/C4 C 级切视频 + 多张静帧")
    from src.highlighter import DEFAULT_C_THUMBS, HL_VERSION, HighlightCutter

    cfg = load_config()
    h = cfg["highlight"]
    check(h.get("c_cut_mp4") is True and h.get("keep_coarse_suspect") is True,
          "配置默认：C 级切 mp4 + 粗扫疑似保留（删前保全的两个主开关）")
    check(str(h.get("version") or "") == HL_VERSION,
          f"产物口径版本 = {h.get('version')}（驱动重剪队列）")

    src = Path("src/highlighter.py").read_text(encoding="utf-8")
    seg = src[src.index("def cut_one"):src.index("def run_batch")]
    check('grade in ("A", "B") or (grade == "C" and cut_c)' in seg,
          'C 级走切 mp4 分支（旧代码是 if grade in ("A","B")）')
    check('"thumbs": thumbs' in seg,
          "clips 记录多张代表帧（一张图证明不了这 30 秒里发生过什么）")

    ev = {"start": 10.0, "end": 40.0, "best_t": 22.0, "dur": 30.0}
    ts = HighlightCutter._thumb_times(ev, DEFAULT_C_THUMBS)
    check(len(ts) == DEFAULT_C_THUMBS and ts[0] == 22.0,
          f"C 级取 {DEFAULT_C_THUMBS} 张，首张是最高置信帧（{[round(t,1) for t in ts]}）")
    check(len(set(round(t, 2) for t in ts)) == len(ts)
          and all(10.0 <= t <= 40.0 for t in ts),
          "代表帧时刻落在段内且互不重复")
    check(HighlightCutter._thumb_times(ev, 1) == [22.0],
          "thumb_per_clip=1 时行为不变（A/B 级不受影响）")


def test_coarse_zero_hits_no_crash():
    """粗扫零命中必须被记成「无精华」，绝不抛异常 —— 否则是静默漏检。

    真实事故（2026-09-12 实测）：cut_one 把 `scanned` 定义在 `if fine_on and c_hits`
    的**里面**，而合并两级结果时无条件引用它。而「粗扫零命中」恰恰是最常见的一类
    （帧差预筛跳过 + 确实没有目标）。于是这些片段不是被记成"无精华"，而是抛
    UnboundLocalError → mark_highlight_error → 被踢出队列。

    为什么这条比普通 bug 贵：`highlight_queue` 与 `rehighlight_queue` **都**要求
    `highlight_error IS NULL`，所以一旦踩中，这条视频既不进正常队列、也不进重剪
    队列 —— 它从两边的视野里同时消失，而且库里看起来只是"失败了几条"。
    实测 1,382 条就是这样卡死的。
    """
    section("粗扫零命中：记成无精华，不抛 UnboundLocalError")
    import src.highlighter as HL

    d = Path(tempfile.mkdtemp(prefix="vs_zero_"))
    cutter = object.__new__(HL.HighlightCutter)     # 跳过 __init__（不碰 ffmpeg/模型）
    cutter.cfg = {"highlight": {"auto_dispose": False}}
    cutter.h = {"fine_enabled": True, "c_cut_mp4": True, "keep_person": True,
                "version": HL.HL_VERSION}
    cutter.ffmpeg = cutter.ffprobe = "ffmpeg"
    cutter.work_dir = d
    cutter.coarse = cutter.fine = None              # 没有检测器 → fine_on=False

    real_probe = HL.probe_av
    real_extract = HL.HighlightCutter._extract
    real_scan = HL.HighlightCutter._scan
    HL.probe_av = lambda p, f, timeout=30: {"duration": 42.0, "fps": 15.0,
                                            "has_audio": True}
    HL.HighlightCutter._extract = (
        lambda self, src, out, fps, width, dur=None, start=0.0: [Path("f_00001.jpg")])
    HL.HighlightCutter._scan = (
        lambda self, frames, fps, t0, dur, det, conf, pre, keep, fb: ([], [0.0], 0))
    try:
        srcf = d / "20150619095035_010.mp4"
        srcf.write_bytes(b"v")
        row = {"id": 1, "path": str(srcf), "filename": srcf.name, "year": "2015",
               "rel_dir": "", "duration_sec": 42.0, "batch_id": None,
               "human_decision": None, "duration": 42.0}
        r = cutter.cut_one(row, d / "hl", "t", log=None)
        check(r.get("ok") is True and r.get("clips") == [],
              f"零命中 → ok=True / clips=[]（不抛异常；{r.get('reason') or r.get('note')}）")
        check((r.get("stats") or {}).get("coarse_sent") == 0,
              "统计留下 coarse_sent=0（区分「预筛跳过」与「确实没有」的唯一依据）")
        # 源码级双保险：scanned 的初始化必须在 if 之外。
        # 注意要在 cut_one 的**函数体切片**里找 `if fine_on and c_hits:` ——
        # 全文找会先命中 merge_two_levels 里同名的那个 if（我第一次就这么写错了）。
        src = Path("src/highlighter.py").read_text(encoding="utf-8")
        body = src[src.index("def cut_one"):src.index("def run_batch")]
        head = body[:body.index("if fine_on and c_hits:")]
        check("scanned: List[Tuple[float, float]] = []" in head[-400:],
              "scanned 在 `if fine_on and c_hits` **之外**初始化（回归钉死）")
    finally:
        HL.probe_av = real_probe
        HL.HighlightCutter._extract = real_extract
        HL.HighlightCutter._scan = real_scan
        shutil.rmtree(d, ignore_errors=True)


def _q2_det(cls, conf, area, frame):
    return {"class": cls, "class_id": 0, "conf": conf, "box": [0, 0, 10, 10],
            "area_ratio": area, "frame": frame}


_Q2_GOOD = {"width": 1280, "height": 720, "brightness": 80, "contrast": 50,
            "blur_var": 300, "saturation": 60,
            "is_low_light": False, "is_low_saturation": False, "is_blurry": False}


def _q2_run(dets, motion=0.0, audio=None, has_audio=False, fstats=None,
            mode="quality_v2", n_frames=15):
    """跑一遍 build_decision。阈值按 mode 配套（口径不同，分数不可跨 mode 比）。"""
    from src.analyzer import build_decision
    from src.config import _default

    cfg = _default()
    cfg["scoring"]["mode"] = mode
    cfg["analysis"]["yolo_conf_threshold"] = 0.4
    if mode == "quality_v2":
        # 质量分 0-100，单类内容分上限 0.55；动静/音频权重 0.06 ⇒ 各 6 分。
        # trash 必须 > 6 才可能把"只有动静"判 trash；review 35 让清晰近人
        # （实测 49.5）能进 keep，同时把远小的人（24.8）留在 review。
        cfg["verdict"].update(trash_threshold=20, review_threshold=35,
                              release_threshold=20)
    return build_decision(dets, motion, audio, has_audio,
                          fstats if fstats is not None else dict(_Q2_GOOD),
                          cfg, True, n_frames=n_frames)


def test_l1_quality_scoring():
    """quality_v2 公式：多帧一致性 + noisy-OR 内容分（算例钉死）。"""
    section("L1 · quality_v2 公式（质量分）")
    P = _q2_det

    # ---- 1. 修掉的核心漏洞：conf 0.6~1.0 一律 keep ----
    d = [P("person", 0.62, 0.03, i) for i in (0, 3, 6, 9, 12)]
    new, old = _q2_run(d), _q2_run(d, mode="legacy")
    check(old["verdict"] == "keep",
          f"旧公式：conf 0.62 直接 keep（{old['score']} 分）—— 这就是漏洞")
    check(new["verdict"] == "review",
          f"★新公式：同一段进 review（{new['score']} 分）")

    # ---- 2. 清晰但远的人也不该一步 keep ----
    d2 = [P("person", 0.75, 0.01, i) for i in range(8)]
    check(_q2_run(d2, mode="legacy")["verdict"] == "keep"
          and _q2_run(d2)["verdict"] == "review",
          "清晰但远小的人：旧 keep → 新 review")

    # ---- 3. 「只有动静」不再白送 10 分进 review ----
    m_new, m_old = _q2_run([], motion=0.12), _q2_run([], motion=0.12,
                                                      mode="legacy")
    check(m_old["verdict"] == "review",
          f"旧公式：只有动静 → review（{m_old['score']} 分）")
    check(m_new["verdict"] == "trash",
          f"★新公式：只有动静 → trash（{m_new['score']} 分）")

    # ---- 4. 多帧一致性真的有区分度（旧公式完全没有）----
    one = _q2_run([P("person", 0.75, 0.03, 7)])
    many = _q2_run([P("person", 0.75, 0.03, i) for i in range(1, 9)])
    check(many["score"] > one["score"],
          f"8 帧检出比单帧高（{many['score']} > {one['score']}）")
    check(_q2_run([P("person", 0.75, 0.03, 7)],
                  mode="legacy")["score"]
          == _q2_run([P("person", 0.75, 0.03, i) for i in range(1, 9)],
                     mode="legacy")["score"],
          "对照：旧公式下两者完全同分（分不出多帧）")

    # ---- 5. 典型画面的判定 ----
    cases = [
        ("清晰近人 conf0.9 占比25% 10帧",
         [P("person", 0.9, 0.25, i) for i in range(10)], {}, "keep"),
        ("无人空景", [], {"motion": 0.01}, "trash"),
        ("有声音但无目标", [], {"audio": -30, "has_audio": True}, "trash"),
    ]
    for tag, dets, opts, expect in cases:
        r = _q2_run(dets, motion=opts.get("motion", 0.0), audio=opts.get("audio"),
                    has_audio=opts.get("has_audio", False))
        check(r["verdict"] == expect, f"「{tag}」⇒ {r['verdict']}（期望 {expect}）")

    # ---- 6. mode 可回滚 ----
    check(new.get("l1_version") == "qv2", "v2 写 l1_version='qv2'（抽查页据此过滤）")
    check(old.get("l1_version") is None, "legacy 不写 l1_version")
    check(old["verdict"] != new["verdict"],
          f"同一段 legacy={old['verdict']} / v2={new['verdict']} ⇒ 可对比可回滚")


def test_l1_floor_safety():
    """保底规则（Floor）与 Veto：任何情况下不能把"有人"判成 trash。"""
    section("L1 · quality_v2 保底规则与安全边界")
    P = _q2_det
    bad = dict(_Q2_GOOD, height=240, is_low_light=True, is_blurry=True,
               is_low_saturation=True, brightness=20, blur_var=30, saturation=10)

    # F0：任何一帧检出 person ⇒ 绝不 trash（哪怕糊 + 远小 + 单帧）
    w = _q2_run([P("person", 0.42, 0.003, 2)], fstats=bad)
    check(w["verdict"] != "trash",
          f"糊+远小+单帧弱信号仍不判 trash（{w['score']} → {w['verdict']}）")
    check(w["floor"]["F0_any_person"] is True, "F0（见到人）已触发")

    # Veto 只取消弱保底
    v = _q2_run([P("person", 0.8, 0.09, i) for i in (1, 2)], fstats=bad)
    check(v["floor"]["veto"] is True, "糊+暗 ⇒ Veto 触发")
    check(v["floor"]["F4_weak_person"] is False, "Veto 取消了弱保底 F4")
    check(v["floor"]["F0_any_person"] is True,
          "★Veto 绝不取消 F0 —— 见到人就不能判 trash")

    # 质量乘数下界：质量再差也不清零
    worst = dict(_Q2_GOOD, height=100, is_low_light=True, is_blurry=True,
                 is_low_saturation=True)
    w2 = _q2_run([P("person", 0.95, 0.5, i) for i in range(15)], fstats=worst)
    check(w2["quality_mult"] >= 0.5,
          f"质量乘数不低于 0.5（实得 {w2['quality_mult']}）")
    check(w2["content_score"] > 0, f"内容分未清零（{w2['content_score']}）")
    check(w2["verdict"] == "keep",
          f"极端画质 + 强目标 ⇒ 仍 keep（{w2['verdict']}）")

    # 质量维度真的进了判定（现状里 blur/低光/分辨率只是文案）
    sharp = _q2_run([P("person", 0.7, 0.04, i) for i in range(4)])
    blurry = _q2_run([P("person", 0.7, 0.04, i) for i in range(4)],
                     fstats=dict(_Q2_GOOD, is_blurry=True, blur_var=30))
    check(blurry["score"] < sharp["score"],
          f"★同样检出，画面糊则分数更低（{blurry['score']} < {sharp['score']}）"
          f"—— 旧公式下两者完全相同")


def test_l1_mode_switch():
    """`scoring.mode` 切换：配置层即可回滚，不需 git revert、不需重跑分析。"""
    section("L1 · quality_v2 / legacy 模式切换")
    import inspect

    from src.analyzer import build_decision
    from src.config import DEFAULT_CONFIG, _deep_merge, _default, _warn_sample_frames

    check("mode" in DEFAULT_CONFIG["scoring"],
          "DEFAULT_CONFIG.scoring 有 mode 键（缺省即 legacy）")
    check(DEFAULT_CONFIG["scoring"]["mode"] == "legacy",
          "默认仍是 legacy —— 切到 v2 是机主的显式决定")
    for seg in ("content", "consistency", "size", "quality"):
        check(seg in DEFAULT_CONFIG["scoring"],
              f"scoring.{seg} 段存在（v2 专用参数）")
    check("floor" in DEFAULT_CONFIG["scoring"]["quality"],
          "scoring.quality.floor 保底规则可配")
    check("mult_floor" in DEFAULT_CONFIG["scoring"]["quality"],
          "scoring.quality.mult_floor —— 乘数下界（永不清零）可配")
    # F0 写死在代码里，不该出现在配置里（它是底线语义，不该被配置关掉）
    check("f0" not in str(DEFAULT_CONFIG["scoring"]["quality"]["floor"]).lower(),
          "F0 不在配置里 —— 见到人就保底是底线，不给关")

    # 缺配置时 load 能补齐（增删配置项不破坏旧配置，项目既有约定）
    merged = _deep_merge(DEFAULT_CONFIG, {"scoring": {"person_weight": 99}})
    check(merged["scoring"]["person_weight"] == 99
          and "content" in merged["scoring"],
          "旧配置 + 新默认值能合并（老 config.json 不会因新键失效）")

    # sample_frames < 12 且 mode=qv2 ⇒ 给出告警（静默失效比崩溃更难发现）
    import src.config as CFG
    low = _default()
    low["scoring"]["mode"] = "quality_v2"
    low["analysis"]["sample_frames"] = 6
    low["verdict"]["trash_threshold"] = 20
    CFG.LAST_NOTICE = ""
    _warn_sample_frames(low)
    check("6 帧" in CFG.LAST_NOTICE,
          f"抽帧 < 12 且用 v2 ⇒ 提示一致性会失效（{CFG.LAST_NOTICE[:40]}…）")

    # 阈值没跟着换 ⇒ 也要提示（0 是旧 125 分制的口径，v2 下 trash 不可达）
    th = _default()
    th["scoring"]["mode"] = "quality_v2"
    th["analysis"]["sample_frames"] = 15
    th["verdict"]["trash_threshold"] = 0
    CFG.LAST_NOTICE = ""
    _warn_sample_frames(th)
    check("trash_threshold" in CFG.LAST_NOTICE,
          "v2 下 trash_threshold 仍为 0 ⇒ 提示重设阈值")

    # legacy 模式不打扰
    leg = _default()
    leg["analysis"]["sample_frames"] = 6
    CFG.LAST_NOTICE = ""
    _warn_sample_frames(leg)
    check(CFG.LAST_NOTICE == "", "legacy 模式不触发 v2 专属告警")

    # build_decision 接受 n_frames（旧公式忽略它）
    import inspect as _i
    check("n_frames" in _i.signature(build_decision).parameters,
          "build_decision 接受 n_frames（persist 的分母）")


def _spot_seed_db(n: int = 60):
    """造一个带 L1 行 + A/B 产物的临时库，供抽查相关测试用。"""
    d = Path(tempfile.mkdtemp(prefix="spot-"))
    db = Database(str(d / "t.db"))
    with db.conn() as c:
        for i in range(n):
            v = ("keep", "review", "trash")[i % 3]
            clips = ([{"grade": "A", "file": f"G:/h/a{i}.mp4", "dur": 9.0},
                      {"grade": "B", "file": f"G:/h/b{i}.mp4", "dur": 4.0},
                      {"grade": "C", "file": f"G:/h/c{i}.mp4", "dur": 5.0}]
                     if i % 2 else [])
            c.execute(
                """INSERT INTO videos (path, filename, meta_ok, status, verdict,
                   retention_score, l1_version, has_person,
                   highlight_at, highlight_clips, ab_clips_count,
                   highlight_best_grade, highlight_count)
                   VALUES (?,?,1,'analyzed',?,?,?,?,?,?,?,?,?)""",
                (f"G:/src/{i}.mp4", f"{i}.mp4", v, 25.0 + (i % 10),
                 "qv2", 1 if v == "keep" else 0,
                 "2026-10-05 10:00:00" if clips else None,
                 json.dumps(clips, ensure_ascii=False) if clips else "[]",
                 2 if clips else 0, "A" if clips else None,
                 len(clips)))
    return d, db


def test_spotcheck_sampling():
    """抽查抽样：可复现、跨层、不混口径、只取必要列。"""
    section("抽查页：抽样可复现")
    d, db = _spot_seed_db(60)
    try:
        # 1) 同 seed ⇒ 完全相同（这是抽查页存在的**前提**：
        #    多人对齐口径、复查某条，都要求"同一批"能再抽出来）
        a = [r["id"] for r in db.sample_ids(seed=42, n=10)]
        b = [r["id"] for r in db.sample_ids(seed=42, n=10)]
        check(a == b, f"同 seed 完全可复现（{a[:5]}…）")
        c = [r["id"] for r in db.sample_ids(seed=43, n=10)]
        check(set(a) != set(c) or len(a) < 3,
              "换 seed 得到不同的样本（不是固定前 N 条）")

        # 2) 跨层：三种 verdict 都能抽到（trash 层是 L1 漏洞最大处）
        with db.conn() as cc:
            allv = {r["id"]: r["verdict"] for r in cc.execute(
                "SELECT id, verdict FROM videos").fetchall()}
        got = {allv[i] for i in a}
        check(len(got) >= 2, f"跨 verdict 分层（抽到 {sorted(got)}）")

        # 3) ★只抽 l1_version='qv2'（旧公式的行混进来会把"公式不准"
        #    与"样本不是这个公式判的"混为一谈）
        with db.conn() as cc:
            cc.execute("UPDATE videos SET l1_version=NULL WHERE id=?", (a[0],))
        check(a[0] not in [r["id"] for r in db.sample_ids(seed=42, n=10)],
              "★l1_version 为 NULL（legacy）的行不会被抽到")

        # 4) 池规模与样本同口径（报"keep 层 20 条"却抽不出 keep 就是口径漂移）
        pool = db.count_sample_pool()
        check(pool["total"] == 59,
              f"池规模与谓词一致（{pool}，l1_version=NULL 那条已排除）")
        ids = [r["id"] for r in db.sample_ids(seed=7, n=200)]
        check(len(ids) == 59, f"能抽满全池（{len(ids)} / 59）")

        # 5) A/B 片段抽样：可复现 + 只取有 mp4 的 A/B
        x = db.sample_ab_clips(seed=7, n=5)
        y = db.sample_ab_clips(seed=7, n=5)
        check([(i["video_id"], i["index"], i["grade"]) for i in x]
              == [(i["video_id"], i["index"], i["grade"]) for i in y],
              "A/B 片段抽样可复现")
        check(all(i["grade"] in ("A", "B") and i.get("file") for i in x),
              "A/B 抽样只含 A/B 级且有 mp4 的段")
        check(all(i["grade"] != "C" for i in x), "C 级档案不进 A/B 抽样")
        check(db.count_sample_pool_ab()["total"] == 60,
              f"A/B 池规模（{db.count_sample_pool_ab()['total']} 段 = 30 条 × 2）")

        # 6) 源码级：抽样不在 UI 里自己抽（口径会分叉）
        ui_src = Path("src/ui.py").read_text(encoding="utf-8")
        check("random.sample" not in ui_src,
              "UI 层不自己做抽样（统一走 db.sample_ids / sample_ab_clips）")
        db_src = Path("src/db.py").read_text(encoding="utf-8")
        db_code = "\n".join(ln for ln in db_src.splitlines()
                            if not ln.strip().startswith("#"))
        check("ORDER BY random()" not in db_code,
              "★SQL 里没有 ORDER BY random()（无种子、rerun 就变、无法复现）"
              "—— 注释里提到它是为了说明为什么不这么写")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_spotcheck_rejudge():
    """抽查改判：保底不变量 + meta_ok=0 不许移走 + clips 派生列同步。"""
    section("抽查页：即改判")
    d, db = _spot_seed_db(4)
    try:
        with db.conn() as c:
            ids = [r[0] for r in c.execute("SELECT id FROM videos LIMIT 2").fetchall()]
        row = db.get_video(ids[0])
        db.set_machine_decision(ids[0], "trash", trash_path="G:/need-del/x.mp4",
                                trash_tier="A")

        # 1) 改判「该留」⇒ trash_path/trash_tier 必须清空（硬不变量）
        db.set_human_decision(ids[0], "keep", decided_by="human")
        r2 = db.get_video(ids[0])
        check(r2["trash_path"] is None and r2["trash_tier"] is None,
              "★改判该留 ⇒ trash_path/trash_tier 恒为 NULL")
        check(r2["decided_by"] == "human", "改判来源记为 'human'（与机器区分）")

        # 2) 抽样页 UI 不该给 meta_ok=0 的行开「该删」
        ui_src = Path("src/ui.py").read_text(encoding="utf-8")
        seg = ui_src[ui_src.index("def _spot_actions"):]
        seg = seg[:seg.index("\ndef ")] if "\ndef " in seg[3:] else seg
        check("disabled=not is_meta_ok" in seg,
              "★meta_ok=0（分析失败/解不开）的行「该删」被禁用 —— "
              "那类行的 keep 是技术兜底，不是判定结论")
        check('st.error("原文件已不存在")' in ui_src or
              "原文件已不存在" in seg,
              "文件缺失时明确报错")

        # 3) ★改片段级别 ⇒ 派生列同步（否则「已剪辑」标记失真 ⇒
        #    机主以为可删、实际 A 级 mp4 还在硬盘上）
        with db.conn() as c:
            rowv = c.execute(
                "SELECT id FROM videos WHERE ab_clips_count>0 LIMIT 1").fetchone()
        vid = rowv["id"] if rowv else None
        if vid:
            ok = db.set_clip_grade(vid, 0, "C")     # 把 A 级降成 C
            check(ok, "set_clip_grade 可改级别")
            r3 = db.get_video(vid)
            check(r3["ab_clips_count"] == 1,
                  f"★ab_clips_count 同步（A 降 C 后从 2 变 1，实得 {r3['ab_clips_count']}）")
            check(r3["highlight_best_grade"] == "B",
                  f"★highlight_best_grade 同步（A 降 C 后最高级变 B，"
                  f"实得 {r3['highlight_best_grade']!r}）")
            check("mover.align_one" in ui_src,
                  "改级别后 UI 会调 align_one 把原片挪到正确子档")
        # 4) 越界索引不得改坏数据
        check(db.set_clip_grade(vid if vid else 0, 999, "A") is False,
              "越界的片段索引被拒绝（返回 False 而不是写坏 JSON）")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_decided_by():
    """decided_by 溯源 + 决策留痕（v3 阶段 1 的固化）。"""
    section("决策溯源：decided_by 与 decide:* 日志")
    d, db, _b, ids = _tmp_db(3)
    try:
        with db.conn() as c:
            v = c.execute("SELECT decided_by FROM videos WHERE id=?",
                          (ids[0],)).fetchone()
        check(v["decided_by"] is None,
              "新行 decided_by 为 NULL（不是 'human' —— 否则机器决策被伪装成人工）")

        db.set_machine_decision(ids[0], "trash", trash_path="G:/x.mp4",
                                trash_tier="A")
        with db.conn() as c:
            v = c.execute("SELECT decided_by, human_decision FROM videos "
                          "WHERE id=?", (ids[0],)).fetchone()
            logs = [r["action"] for r in c.execute(
                "SELECT action FROM action_log WHERE video_id=?", (ids[0],))]
        check(v["decided_by"] == "auto", "set_machine_decision ⇒ 'auto'")
        check(v["human_decision"] == "trash", "处置结论正确写入")
        check("decide:trash" in logs,
              f"★决策留痕在 set_human_decision 内部完成（{logs}）")

        db.set_human_decision(ids[0], "keep", decided_by="human")
        with db.conn() as c:
            v = c.execute("SELECT decided_by, trash_path, trash_tier FROM videos "
                          "WHERE id=?", (ids[0],)).fetchone()
        check(v["decided_by"] == "human", "人工路径写 'human'")
        check(v["trash_path"] is None and v["trash_tier"] is None,
              "★不变量：keep ⇒ trash_path/trash_tier 恒为 NULL")

        # 不传 decided_by 时不动该列（宁可不写也不猜）
        db.set_human_decision(ids[1], "keep", decided_by="human")
        db.set_human_decision(ids[1], "trash", trash_path="G:/y.mp4")
        with db.conn() as c:
            v = c.execute("SELECT decided_by FROM videos WHERE id=?",
                          (ids[1],)).fetchone()
        check(v["decided_by"] == "human", "不传 decided_by 时保留旧值（不猜默认值）")
    finally:
        shutil.rmtree(d, ignore_errors=True)

    # ab_clips_count：只数 A/B 且有 mp4（与 mover.ab_clip_files 同判据）
    d2, db2, _b2, ids2 = _tmp_db(1)
    try:
        clips = [
            {"grade": "A", "file": "a.mp4"}, {"grade": "B", "file": "b.mp4"},
            {"grade": "C", "file": "c.mp4"},          # C 不计
            {"grade": "A", "file": None, "thumbs": ["x"]},   # 无 mp4 不计
            {"grade": "C", "thumbs": ["y"]},                 # 只有静帧不计
        ]
        db2.mark_highlight(ids2[0], "t", 5, 19.0, clips, version="hlv3")
        with db2.conn() as c:
            v = c.execute("SELECT ab_clips_count, highlight_count, "
                          "highlight_best_grade FROM videos WHERE id=?",
                          (ids2[0],)).fetchone()
        check(v["ab_clips_count"] == 2, f"ab_clips_count=2（实得 {v['ab_clips_count']}）")
        check(v["highlight_count"] == 5, "highlight_count = len(clips)（含 C 段）")
        check(v["ab_clips_count"] <= v["highlight_count"], "ab ≤ count（差额是 C 段）")
    finally:
        shutil.rmtree(d2, ignore_errors=True)

    # reset_analysis 必须清掉 v2 六列 —— 否则改了公式重跑，抽查页会把
    # 旧公式的行当新样本（l1_version 对得上、内容却是旧的）
    d3, db3, _b3, ids3 = _tmp_db(1)
    try:
        db3.update_analysis(ids3[0], {
            "status": "analyzed", "retention_score": 30, "verdict": "review",
            "l1_version": "qv2", "content_score": 40.0, "quality_score": 100.0,
            "person_frames": 5, "person_persist": 0.33, "person_area_max": 0.03,
        })
        with db3.conn() as c:
            v = c.execute("SELECT l1_version, person_frames FROM videos "
                          "WHERE id=?", (ids3[0],)).fetchone()
        check(v["l1_version"] == "qv2" and v["person_frames"] == 5,
              "update_analysis 写入 v2 六列")
        db3.reset_analysis(scope="undecided")
        with db3.conn() as c:
            v = c.execute("SELECT l1_version, content_score, person_frames, "
                          "person_persist, person_area_max FROM videos "
                          "WHERE id=?", (ids3[0],)).fetchone()
        check(all(v[k] is None for k in v.keys()),
              f"★reset_analysis 清掉 v2 六列（否则改了公式重跑会抽到旧行）")
    finally:
        shutil.rmtree(d3, ignore_errors=True)


def test_c_not_in_review():
    """Track C · C3：产物**只有 C 级**的视频是档案不是待办，默认不进人工队列。

    C1 会让 C 段数量级上升；若它们同时涌进审核队列，"多留"就变成了"多看"，
    与减负目标正面冲突。这条开关只影响列表展示，不参与任何处置判定。
    """
    section("Track C · C3 C 级档案不入审核队列")
    d, db, bid, ids = _tmp_db(3)
    try:
        for vid in ids:
            db.update_analysis(vid, {"status": "analyzed", "retention_score": 5,
                                     "final_score": 5, "verdict": "trash",
                                     "has_person": 0, "decision_trace": {}})
        db.mark_highlight(ids[0], "2026-09-10 12:00:00", 0, 0.0, [],
                          sent=100, coarse_hits=0, version="hlv2")
        db.mark_highlight(ids[1], "2026-09-10 12:00:00", 2, 20.0,
                          [{"grade": "C", "file": "c.mp4", "thumb": "c.jpg"}],
                          sent=100, coarse_hits=3, version="hlv2")
        db.mark_highlight(ids[2], "2026-09-10 12:00:00", 1, 30.0,
                          [{"grade": "A", "file": "a.mp4", "thumb": "a.jpg"}],
                          sent=100, coarse_hits=3, version="hlv2")

        all_rows = db.review_queue(limit=99)
        hid = db.review_queue(limit=99, hide_c_archive=True)
        check(len(all_rows) == 3 and len(hid) == 2,
              f"仅 C 级档案被滤掉（不滤 {len(all_rows)} → 滤后 {len(hid)}）")
        check(ids[1] not in [r["id"] for r in hid],
              "只有 C 级产物的那条不再出现在人工队列")
        check(ids[0] in [r["id"] for r in hid] and ids[2] in [r["id"] for r in hid],
              "无产物 / 有 A 级产物的照常留在队列（筛选不过度）")
        check(db.count_review_queue(hide_c_archive=True) == len(hid),
              "计数与队列同口径（筛选不能只改一边）")
        check(db.no_clip_breakdown()["total"] == 1,
              "三义拆分只统计无产物行，不受该筛选影响")

        # 放行判据（真实处置路径）绝不能被这个展示开关带偏
        check(len(db.release_queue(limit=99, threshold=30,
                                   require_no_person=True)) == 3,
              "放行候选不受 hide_c_archive 影响（展示开关不参与处置判定）")

        # ---- 修掉一个"曾经空转"的真实缺陷（2026-10-04，v3 阶段 2）----
        #
        # 旧判据只有 `COALESCE(highlight_best_grade,'') <> 'C'`。而**没剪过的行**
        # best_grade 是 NULL，`NULL <> 'C'` 在 SQL 里恒为真 ⇒ 那些行照样留在队列。
        # 这与"仅滤掉已剪出纯 C 档案"的本意不冲突，但真正的漏洞是：
        # 历史行（剪辑在 best_grade 这一列出现之前）的 best_grade **全为 NULL**，
        # 于是 `min(grades)` 算出来的值写不进去 —— 整条判据对存量数据**一条都没滤掉**。
        # 现在判据加了 `highlight_at IS NULL OR ...` 分支，把"没剪"显式放行，
        # 存量行（best_grade=NULL 但 highlight_at 非空）则落在 `'' <> 'C'` 为真的
        # 那一侧 —— 与旧行为一致，不改变存量语义，只修好新写入的行。
        with db.conn() as c:
            c.execute("UPDATE videos SET highlight_best_grade=NULL WHERE id=?",
                      (ids[1],))          # 模拟历史行：剪过但没记 best_grade
        legacy = db.review_queue(limit=99, hide_c_archive=True)
        check(ids[1] in [r["id"] for r in legacy],
              "存量行（best_grade=NULL）不被误滤 —— 判据只对新写入的等级生效")
        # 反向：新写入的纯 C 行必须被滤掉（这条是真断言，不是形式化）
        db.mark_highlight(ids[1], "t", 2, 20.0,
                          [{"grade": "C", "file": "c.mp4", "thumb": "c.jpg"}],
                          sent=100, coarse_hits=3, version="hlv2")
        again = db.review_queue(limit=99, hide_c_archive=True)
        check(ids[1] not in [r["id"] for r in again],
              "新写入的纯 C 档案被真正滤掉（判据已修复）")
        # 未剪的行（highlight_at IS NULL）必须留在队列 —— 那是待办不是档案
        with db.conn() as c:
            c.execute("UPDATE videos SET highlight_at=NULL, "
                      "highlight_best_grade=NULL WHERE id=?", (ids[1],))
        pending_rows = db.review_queue(limit=99, hide_c_archive=True)
        check(ids[1] in [r["id"] for r in pending_rows],
              "未剪的行留在队列（highlight_at IS NULL 分支）")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_no_clip_split():
    """Track 0：把「无精华」拆成三义，且**三数之和 == 无精华总数**。

    没有这把尺子，"无精华 3780" 就永远分不清是"复核过确实没有"、
    "压根没跑检测"、还是"粗扫看到过却被丢弃"—— 第二条曾被当成第一条用过。
    """
    section("Track 0 · 「无精华」拆三义")
    d, db, bid, ids = _tmp_db(4)
    try:
        for vid in ids:
            db.update_analysis(vid, {"status": "analyzed", "retention_score": 0,
                                     "verdict": "trash", "decision_trace": {}})
        # ① 预筛跳过 ② 已检测确实没有 ③ 粗扫看到过但被否决 ④ 有产物（不计入）
        db.mark_highlight(ids[0], "t", 0, 0.0, [], sent=0, coarse_hits=0, version="hlv2")
        db.mark_highlight(ids[1], "t", 0, 0.0, [], sent=120, coarse_hits=0, version="hlv2")
        db.mark_highlight(ids[2], "t", 0, 0.0, [], sent=120, coarse_hits=7, version="hlv2")
        db.mark_highlight(ids[3], "t", 1, 9.0,
                          [{"grade": "B", "file": "b.mp4"}], sent=120,
                          coarse_hits=7, version="hlv2")
        b = db.no_clip_breakdown()
        check((b["skipped"], b["detected_empty"], b["dropped"]) == (1, 1, 1),
              f"三义各归各位：预筛跳过={b['skipped']} 已检测无={b['detected_empty']} "
              f"看到过被否={b['dropped']}")
        check(b["total"] == 3 and b["unknown"] == 0,
              f"三数之和 = 无精华总数（{b['total']}），没有算不进去的行")
        check(db.no_clip_breakdown()["total"] == db.highlight_stats()["done"] - 1,
              "无精华数 + 有产物数 = 已处理数（口径自洽）")
        # 回填前（无送检记录）的行必须显式归到 unknown，不能悄悄并进某一义
        d2, db2, bid2, ids2 = _tmp_db(1)
        try:
            db2.update_analysis(ids2[0], {"status": "analyzed", "retention_score": 0,
                                          "verdict": "trash", "decision_trace": {}})
            db2.mark_highlight(ids2[0], "t", 0, 0.0, [])
            b2 = db2.no_clip_breakdown()
            check(b2["unknown"] == 1 and b2["total"] == 1,
                  "没有送检记录的老行归入 unknown，不伪装成三义之一")
        finally:
            shutil.rmtree(d2, ignore_errors=True)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_rehighlight_idempotent():
    """Track C · C5：按版本重剪必须**幂等**，且只挑收益最集中的那一批。

    重剪跑 1.5~8 小时，中途一定会停；停下再起不能从头再来一遍。
    """
    section("Track C · C5 按版本重剪（幂等）")
    from src.highlighter import HL_VERSION

    d, db, bid, ids = _tmp_db(4)
    try:
        for vid in ids:
            db.update_analysis(vid, {"status": "analyzed", "retention_score": 0,
                                     "verdict": "trash", "decision_trace": {}})
        # ① D2 受害者：粗扫看到过、零产物   ② 只有静帧的 C  ③ 已是新口径  ④ 没剪过
        db.mark_highlight(ids[0], "t", 0, 0.0, [], sent=100, coarse_hits=5,
                          version="hlv1")
        db.mark_highlight(ids[1], "t", 1, 12.0,
                          [{"grade": "C", "file": None, "thumb": "t.jpg"}],
                          sent=100, coarse_hits=2, version="hlv1")
        db.mark_highlight(ids[2], "t", 1, 9.0,
                          [{"grade": "B", "file": "b.mp4"}],
                          sent=100, coarse_hits=1, version=HL_VERSION)
        q = db.rehighlight_queue(limit=99, target_version=HL_VERSION, only_dropped=True)
        got = [r["id"] for r in q]
        check(ids[0] in got and ids[1] in got,
              "抓到「零产物但有粗扫命中」和「只有静帧」的两类（D2/D4）")
        check(ids[2] not in got, "已是新口径的跳过（不重复劳动）")
        check(ids[3] not in got, "没剪过的走正常队列，不出现在重剪队列")
        check(len(q) == 2, f"重剪队列只含目标行（{len(q)} 条）")

        # 全量口径：所有版本落后的都进（不止 D2/D4）
        db.mark_highlight(ids[2], "t", 1, 9.0, [{"grade": "B", "file": "b.mp4"}],
                          sent=100, coarse_hits=1, version="hlv1")
        check(len(db.rehighlight_queue(limit=99, target_version=HL_VERSION,
                                       only_dropped=False)) == 3,
              "only_dropped=False 时全量重剪（含已正常的 A/B 产物）")

        # 幂等：重剪完把版本推上去，队列必须清空
        db.reset_highlight_version(HL_VERSION)
        check(db.count_rehighlight_pending(target_version=HL_VERSION) == 0,
              "剪完版本号跟上 ⇒ 重剪队列清空（中断后重启不会重复跑）")
        check(db.count_rehighlight_pending(target_version=HL_VERSION,
                                           only_dropped=False) == 0,
              "全量口径同样清空")
    finally:
        shutil.rmtree(d, ignore_errors=True)

    # jobs 里有没有真的接上（UI 按钮要靠它）
    jsrc = Path("src/jobs.py").read_text(encoding="utf-8")
    check("_REHIGHLIGHT" in jsrc and "def start_rehighlight" in jsrc
          and "def count_rehighlight_pending" in jsrc,
          "jobs 暴露重剪任务（可启动 / 可计数）")
    check("get_rehighlight_job().running" in jsrc,
          "any_running 纳入重剪任务（它与剪辑抢同一批源文件，不能并发）")


def test_failed_retry_recovers_halfdone():
    """磁盘满留下的半成品行必须能被捞回来 —— 单靠 reset_highlight("failed") 不行。

    实测事故（2026-10-08）：磁盘满发生在 mark_highlight() 之后、写manifest 之前，
    留下 `highlight_at` 非空 + `version` 已是目标值 + `clips=[]` + `error 非空`
    的行。此时两条队列的门同时关上：
      - highlight_queue 要求 `highlight_at IS NULL`
      - rehighlight_queue 要求 `version <> target`
    而 `reset_highlight("failed")` 只清 error ⇒ 清完队列仍为 0（真事故 530 条）。
    """
    section("失败重试：零产物半成品行能退回队列（磁盘满事故）")
    from src.highlighter import HL_VERSION

    d, db, bid, ids = _tmp_db(5)
    try:
        for vid in ids:
            db.update_analysis(vid, {"status": "analyzed", "retention_score": 0,
                                     "verdict": "trash", "decision_trace": {}})
        # ①②③ 模拟磁盘满的半成品：写库已提交（at 与 version 都到位），
        #      但 clips 为空、随后被标了 Errno 28。
        # ④ 一条"有产物"的失败行：不该被重试逻辑拖回队列。
        # ⑤ 一条 keep 行 + 零产物 + 有 error：人工已表态，程序不该动。
        for i in (0, 1, 3):
            db.mark_highlight(ids[i], "t", 0, 0.0, [], sent=80, coarse_hits=4,
                              version=HL_VERSION)
            db.mark_highlight_error(ids[i], "OSError: [Errno 28] No space left on device")
        db.mark_highlight(ids[2], "t", 2, 15.0,
                          [{"grade": "B", "file": "b.mp4"}],
                          sent=50, coarse_hits=3, version=HL_VERSION)
        db.mark_highlight_error(ids[2], "boom")
        db.set_human_decision(ids[4], "keep")

        check(len(db.highlight_queue(limit=99)) == 0,
              "事故态：半成品行进不了正常剪辑队列（highlight_at 已写入）")
        check(len(db.rehighlight_queue(limit=99, target_version=HL_VERSION,
                                       only_dropped=True)) == 0,
              "事故态：也进不了重剪队列（version 已是目标值）")
        check(db.count_failed_retryable() == 3,
              f"零产物失败行被识别为可重试（{db.count_failed_retryable()} 条，应为 3）")

        n = db.reset_highlight("failed_retry")
        check(n == 3, f"failed_retry 退回全部零产物失败行（{n} 条，应为 3）")
        q = [r["id"] for r in db.highlight_queue(limit=99)]
        check(ids[0] in q and ids[1] in q,
              "零产物的失败行已回到正常剪辑队列（可以重剪了）")
        check(ids[2] not in q,
              "有产物的失败行不被拖回队列（产物还在，别浪费机时）")
        check(ids[4] not in q,
              "keep 行不动（人工已表态，程序不碰它的产物状态）")
        check(db.reset_highlight("failed_retry") == 0,
              "幂等：清完再跑一次影响 0 行（不会自我循环）")

        # all 会误伤，必须能说清为什么不能用它
        check(db.reset_highlight("all") == 5,
              "all 是无条件的（5 条全中）—— 这就是它不能用来救这批的原因")
    finally:
        shutil.rmtree(d, ignore_errors=True)

    # ⚠️ 操作陷阱（真实存在，UI 已规避）：先点了失效的 failed 再点 failed_retry，
    #    failed_retry 会**一行都救不回来** —— error 已被清空，它的判据就失效了。
    #    顺序必须是 failed_retry 在前（或干脆只用 failed_retry）。
    d2, db2, _bid2, ids2 = _tmp_db(3)
    try:
        for vid in ids2:
            db2.update_analysis(vid, {"status": "analyzed", "retention_score": 0,
                                      "verdict": "trash", "decision_trace": {}})
            db2.mark_highlight(vid, "t", 0, 0.0, [], sent=80, coarse_hits=4,
                               version=HL_VERSION)
            db2.mark_highlight_error(vid, "OSError: [Errno 28] No space left on device")
        db2.reset_highlight("failed")
        check(db2.count_failed_retryable() == 0,
              "陷阱确认：failed 先跑会把 error 清空，之后 failed_retry 数不出行")
        check(len(db2.highlight_queue(limit=99)) == 0,
              "陷阱后果：这两步按顺序点完，这批行依然剪不到（所以 UI 只给一个入口）")
        uisrc = Path("src/ui.py").read_text(encoding="utf-8")
        check('reset_highlight("failed")' not in uisrc,
              "UI 不再暴露那个会把自己变成死路的 failed 入口")
    finally:
        shutil.rmtree(d2, ignore_errors=True)



def test_highlight_disk_guard():
    """剪辑必须有磁盘硬闸门 —— 空间不足时别开跑。

    理由不是"跑不满会报错"，而是跑一半炸盘会留下捞不回来的半成品行
    （见test_failed_retry_recovers_halfdone）。留在队列里是干净状态，
    留下半成品行则两条队列都收不到。
    """
    section("剪辑前磁盘空间预检")
    from src import highlighter as hl

    check(hasattr(hl, "MIN_FREE_GIB") and hl.MIN_FREE_GIB > 0,
          f"定义了安全阈值（{getattr(hl, 'MIN_FREE_GIB', 0)} GiB）")
    check(callable(getattr(hl, "disk_block_reason", None)),
          "disk_block_reason 存在（不够则返回阻止理由，够则 None）")
    src = Path("src/highlighter.py").read_text(encoding="utf-8")
    # 批次开始前拦一次
    check(src.count("disk_block_reason(self.cfg)") >= 1,
          "run_batch 在批次开始前调用预检（空间不足直接 raise）")
    # 批次中途复查：开跑时有空间 ≠ 跑到第800 条时还有
    check("% 20 == 1" in src or "i % 20" in src,
          "批次中途复查剩余空间（不能只在开跑时看一眼）")
    check("disk_stopped_at" in src,
          "因空间不足中止时把剩余条数带进返回值（剩下的留在队列里）")
    # 取不到空间时不能拦 —— 预检失败不该把整批挡死
    check("return None" in src.split("def free_gib")[1].split("def ")[0],
          "free_gib 取不到时返回 None，不因此拦任务")

    uisrc = Path("src/ui.py").read_text(encoding="utf-8")
    check("disk_block_reason" in uisrc,
          "UI 在启动按钮前就显示空间不足（否则用户只看到'启动失败'）")
    check("reset_highlight(\"failed_retry\")" in uisrc,
          "UI 挂的是 failed_retry 入口（不是那个点了没用的 failed）")
    dbsrc = Path("src/db.py").read_text(encoding="utf-8")
    check("def count_failed_retryable" in dbsrc,
          "db 暴露可重试条数（按钮上要写清会捞回几条）")


def test_recycle_subdirs():
    """回收子档三态：已剪辑 / 待保全 / 无精华产物。

    为什么值得钉死：回收目录的终点是**人工手动删除**，打开文件夹时唯一要回答的
    问题是"这一条我现在能删吗"。判错了会**静默**把"精华还没落袋的"混进
    "可以整批删的"里 —— 代价是产物都没生成，原片已经被删了，不可逆。
    所以四样都要钉：三态判据、落点层级、**旧快照不得污染判据**、
    以及"行内判断"与"剪辑队列口径"必须一致（两套判据漂了就是下一个事故）。
    """
    section("回收子档三态（已剪辑 / 待保全 / 无精华产物）")
    import src.mover as M
    import src.db as DBmod

    # ---------- A. 判据：has_ab_clip ----------
    d0 = Path(tempfile.mkdtemp(prefix="vs_clip_"))
    try:
        real = d0 / "a.mp4"
        real.write_bytes(b"x")
        gone = d0 / "b.mp4"                     # 故意不创建
        check(M.has_ab_clip([{"grade": "A", "file": str(real)}]),
              "A 级 + 产物存在 → 算「已剪辑」")
        check(M.has_ab_clip([{"grade": "B", "file": str(real)}]), "B 级同样算")
        check(not M.has_ab_clip([{"grade": "C", "file": str(real)}]),
              "只有 C 级 → 不算（C 是档案不是待办，不进人工审核队列）")
        check(not M.has_ab_clip([{"grade": "A", "file": str(gone)}]),
              "A 级但产物文件已不在 → 不算（标记不能给无法兑现的承诺）")
        check(not M.has_ab_clip([{"grade": "A", "file": None}]),
              "A 级但当时切段失败（file=None，只剩静帧）→ 不算")
        check(not M.has_ab_clip([]) and not M.has_ab_clip(None), "零产物 → 不算")
        check(M.has_ab_clip([{"grade": "A", "file": str(gone)}], verify=False),
              "verify=False 只认记录、不碰文件系统（给 UI 现状统计用）")
        check(M.ab_clip_files([{"grade": "A", "file": "x"}, {"grade": "C", "file": "y"},
                               {"grade": "B", "file": None}]) == [Path("x")],
              "ab_clip_files：只取 A/B 且真切出 mp4 的（C 与 file=None 排除）")
    finally:
        shutil.rmtree(d0, ignore_errors=True)

    # ---------- B. 判据：current_subdir 反解 ----------
    check(M.current_subdir(r"G:\need-del\已剪辑\A_x\2024\f.mp4", r"G:\need-del")
          == M.CLIPPED_SUBDIR, "current_subdir：认出子档层")
    check(M.current_subdir(r"G:\need-del\A_x\2024\f.mp4", r"G:\need-del") == "",
          "current_subdir：档位目录下 = 无子档")
    check(M.current_subdir(r"G:\need-del\A_x\2024\已剪辑\f.mp4", r"G:\need-del") == "",
          "current_subdir：只看紧邻回收根那层（原目录树里的同名文件夹不算子档）")
    check(M.SUBDIR_BY_STATE[M.STATE_PLAIN] == "",
          "plain 不进子档（存量 8 千个文件已在档位目录下，没必要全搬一层）")

    # ---------- C. 落点三态 + 存量对齐 ----------
    d, db, bid, ids = _tmp_db(6)
    try:
        cfg = {"work_dir": str(d), "trash_root": str(d / "need-del")}
        rp = len(Path(cfg["trash_root"]).parts)
        hl = d / "hl"
        hl.mkdir()
        clip_a = hl / "v_E01_A_120000_9s.mp4"
        clip_a.write_bytes(b"a")
        clip_c = hl / "v_E01_C_120000_9s.mp4"
        clip_c.write_bytes(b"c")

        def _put(vid):
            p = Path(db.get_video(vid)["path"])
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"v")
            return dict(db.get_video(vid))

        # ① 没剪过 → 批量放行（不走剪辑链路）→ 待保全
        r0 = _put(ids[0])
        ok0, dst0 = M.move_one(db, r0, cfg, "A")
        check(ok0 and Path(dst0).parts[rp] == M.PENDING_SUBDIR,
              f"没剪过的原片 → 落「{M.PENDING_SUBDIR}」（不许混进可删堆）")

        # ② 剪辑链路：刚剪完、只有 C 级。传进去的 row 是**剪辑前的旧快照**
        #    （highlight_at 还是 NULL）—— 必须回库取新鲜行，否则误判成「待保全」
        r1 = _put(ids[1])
        db.mark_highlight(ids[1], "t", 1, 9.0,
                          [{"grade": "C", "file": str(clip_c)}],
                          sent=30, coarse_hits=3, version=M.HIGHLIGHT_TARGET_VERSION)
        ok1, dst1 = M.move_one(db, r1, cfg, "B",
                               clips=[{"grade": "C", "file": str(clip_c)}])
        check(ok1 and M.current_subdir(dst1, cfg["trash_root"]) == "",
              "刚剪完只有 C 级 → 落档位目录（旧快照的 highlight_at 是 NULL，"
              "不回库取新鲜行就会错放进待保全）")

        # ③ 剪出 A 级 → 已剪辑
        r2 = _put(ids[2])
        db.mark_highlight(ids[2], "t", 1, 9.0,
                          [{"grade": "A", "file": str(clip_a)}],
                          sent=30, coarse_hits=5, version=M.HIGHLIGHT_TARGET_VERSION)
        ok2, dst2 = M.move_one(db, r2, cfg, "B",
                               clips=[{"grade": "A", "file": str(clip_a)}])
        check(ok2 and Path(dst2).parts[rp] == M.CLIPPED_SUBDIR,
              f"剪出 A 级 → 落「{M.CLIPPED_SUBDIR}」")
        check(Path(dst2).parts[rp + 1] == M.TIER_NAMES["B"],
              "子档下仍保留原回收档位（两个正交维度都在路径里）")
        r2b = db.get_video(ids[2])
        check(r2b["trash_path"] == dst2 and r2b["trash_tier"] == "B"
              and r2b["human_decision"] == "trash",
              "trash_path 记的是子档内真实落点，档位不变")
        check(ids[2] in {r["id"] for r in db.list_moved()},
              "子档内的文件仍出现在「可撤销移动」队列（撤销能力没被落点破坏）")

        # ④ 存量：先在档位目录，之后补上 A 级产物记录（hlv1）→ 对齐到已剪辑
        r3 = _put(ids[3])
        M.move_one(db, r3, cfg, "A", state=M.STATE_PLAIN)
        db.mark_highlight(ids[3], "t", 1, 9.0,
                          [{"grade": "A", "file": str(clip_a)}],
                          sent=30, coarse_hits=5, version="hlv1")
        before3 = db.get_video(ids[3])["trash_path"]
        ok3, dst3 = M.align_one(db, dict(db.get_video(ids[3])), cfg)
        check(ok3 and M.current_subdir(dst3, cfg["trash_root"]) == M.CLIPPED_SUBDIR
              and Path(dst3).exists() and not Path(before3).exists(),
              "align_one：存量文件搬进子档（同盘 move，原位置已无此文件）")
        r3b = db.get_video(ids[3])
        check(r3b["trash_tier"] == "A" and r3b["trash_path"] == dst3
              and r3b["human_decision"] == "trash",
              "存量搬迁保留档位与处置结论，只更新 trash_path")

        # ⑤ 硬失败 → 待保全（它不在任何队列里，但同样"没保全"）
        r4 = _put(ids[4])
        M.move_one(db, r4, cfg, "B", state=M.STATE_PLAIN)
        db.mark_highlight_error(ids[4], "抽帧失败（测试用）")
        ok4, dst4 = M.align_one(db, dict(db.get_video(ids[4])), cfg)
        check(ok4 and M.current_subdir(dst4, cfg["trash_root"]) == M.PENDING_SUBDIR,
              f"硬失败的行 → 落「{M.PENDING_SUBDIR}」"
              "（队列不收它，但它一样是没落袋的）")

        # ⑥ 粗扫看到过却零产物（hlv1，版本落后）→ 待保全
        r5 = _put(ids[5])
        M.move_one(db, r5, cfg, "A", state=M.STATE_PLAIN)
        db.mark_highlight(ids[5], "t", 0, 0.0, [], sent=30, coarse_hits=7,
                          version="hlv1")
        ok5, dst5 = M.align_one(db, dict(db.get_video(ids[5])), cfg)
        check(ok5 and M.current_subdir(dst5, cfg["trash_root"]) == M.PENDING_SUBDIR,
              "粗扫看到过却零产物（版本落后）→ 待保全（重剪队列收着它）")

        # ⑦ 幂等 + 反向：补剪出 A/B 之后，待保全 → 已剪辑
        okx, msgx = M.align_one(db, dict(db.get_video(ids[5])), cfg)
        check(not okx and "已在" in msgx, f"幂等：已在正确子档内直接跳过（{msgx}）")
        db.mark_highlight(ids[5], "t2", 1, 9.0,
                          [{"grade": "B", "file": str(clip_a)}],
                          sent=30, coarse_hits=7,
                          version=M.HIGHLIGHT_TARGET_VERSION)
        ok6, dst6 = M.align_one(db, dict(db.get_video(ids[5])), cfg)
        check(ok6 and M.current_subdir(dst6, cfg["trash_root"]) == M.CLIPPED_SUBDIR,
              "补剪出 A/B 后：待保全 → 已剪辑（对齐是双向的，跑完保全再对齐一次即可）")

        with db.conn() as _c:
            act = _c.execute(
                "SELECT COUNT(*) FROM action_log WHERE video_id=? AND action='rehome'",
                (ids[3],)).fetchone()[0]
        check(act == 1, "存量搬迁写 action_log（可追溯）")

        # ⑧ 统计（三态计数 + 该不该再搬）
        st = M.recycle_stats(db, cfg, verify=True)
        check(st["total"] == 6 and st["in_trash"][M.STATE_CLIPPED] == 3
              and st["in_trash"][M.STATE_PENDING] == 2
              and st["in_trash"][M.STATE_PLAIN] == 1,
              f"recycle_stats 三态计数（{st['in_trash']}）")
        check(sum(st["to_align"].values()) == 0,
              f"全部就位后 to_align 归零（{st['to_align']}）")

        # ⑧b pending_reasons：待保全的**成因**必须分桶报，且与 is_pending 的
        # 三条出口一一对应。UI 靠它决定引导语 —— 对 error 那批说"去跑剪辑"
        # 是空动作（两条队列都不收它们），这个桶就是为了不写那种话。
        pr = st["pending_reasons"]
        check(sum(pr.values()) == st["in_trash"][M.STATE_PENDING],
              f"pending_reasons 三桶之和 == 待保全总数（{pr} vs "
              f"{st['in_trash'][M.STATE_PENDING]}）")
        check(pr.get(M.PEND_NEVER) == 1 and pr.get(M.PEND_ERROR) == 1
              and pr.get(M.PEND_VERSION) == 0,
              f"pending_reasons 分桶正确（从没剪过=1 / 硬失败=1 / 版本落后=0）：{pr}")
        # 反漂移：每一条被判 pending 的行都必须能归出一个桶，且互斥
        _bad = [r["id"] for r in db.recycle_scan_rows()
                if M.recycle_state(r, verify=True) == M.STATE_PENDING
                and M.pending_reason(r) not in M.PENDING_REASONS]
        check(not _bad, f"pending_reason 的取值都在 PENDING_REASONS 内（异常 id：{_bad}）")
        # 磁盘满那批的形状（at 非空 + version 已到目标 + error 非空）必须归 error，
        # 绝不能被归成 never —— 那个形状正是 2026-10-08 卡死 530 条的形状。
        _shape = {"highlight_at": "2026-10-08 10:00:00", "highlight_error": "OSError",
                  "highlight_version": DBmod.HIGHLIGHT_TARGET_VERSION}
        check(M.pending_reason(_shape) == M.PEND_ERROR,
              "磁盘满半成品的行形状（at 非空 + 版本已到目标 + error 非空）→ 归 error")

        # ⑨ 撤销必须依然可用 —— 子档只是换了个子目录，不该破坏回退
        rr = db.get_video(ids[4])
        ok7, _ = M.undo_move(db, rr)
        rr = db.get_video(ids[4])
        check(ok7 and Path(rr["path"]).exists() and rr["trash_path"] is None,
              "搬迁后的文件仍可撤销回原位（trash_path 不变量未被破坏）")

        # ⑩ 反漂移：**行内判断必须与剪辑队列口径一致**
        # 这是本模块最容易出事的地方 —— 两套判据一旦漂了，UI 会说"可删"而
        # 队列还在排重剪，机主照着删就把没落袋的精华删了。
        pids = db.pending_highlight_ids()
        bad = [r["id"] for r in db.recycle_scan_rows()
               if M.is_pending(r) != (r["id"] in pids)]
        check(not bad, f"行内判断 == 队列口径（不一致的 id：{bad}）")
        check(M.HIGHLIGHT_TARGET_VERSION == DBmod.HIGHLIGHT_TARGET_VERSION,
              "mover 与 db 共用同一个目标版本号常量")
    finally:
        shutil.rmtree(d, ignore_errors=True)

    # ---------- D. UI / 脚本接线（源码级，防改后端忘接前面） ----------
    ui_src = Path("src/ui.py").read_text(encoding="utf-8")
    check("mover.recycle_stats(db, cfg)" in ui_src
          and "mover.align_subdirs(db, cfg)" in ui_src,
          "UI：设置页有「回收目录三堆」卡（现状 + 一键对齐）")
    check("mover.PENDING_SUBDIR" in ui_src and "mover.CLIPPED_SUBDIR" in ui_src,
          "UI：子档名取自 mover 常量（不写死字面量，改名不会漂）")
    check("re_hl_only_first" not in ui_src,
          "UI：不给「全量重剪」勾选框 —— UI 只收症状行；"
          "要全量走 jobs.start_rehighlight(only_dropped=False)")
    # 「零产物失败退回」必须在**回收目录三堆**卡片里（不是剪辑段）：
    # 它的目的是让「待保全」归零，属回收目录的补救动作；挂在剪辑段里，
    # 机主从"为什么待保全不清零"查起要翻过整段参数才找得到，方向是反的。
    _i_recycle = ui_src.find("回收目录三堆")
    _i_retry = ui_src.find("零产物失败记录（可重剪）")
    check(_i_recycle != -1 and _i_retry > _i_recycle,
          "UI：「↩ 退回 N 条零产物失败记录」在「回收目录三堆」卡片内"
          f"（卡片@{_i_recycle} 按钮@{_i_retry}）")
    check(ui_src.count('key="reset_highlight_failed_retry"') == 1,
          "UI：零产物失败退回按钮只有一处（挂两处 = 两个入口 = 迟早只点一个）")
    check("pending_reasons" in ui_src,
          "UI：待保全提示按 pending_reasons 分桶给引导，不写一刀切文案")
    check("先跑完「画面精华剪辑」+「按新口径重剪」" not in ui_src,
          "UI：旧的一刀切提示语已删除（它对磁盘满那批是空动作）")
    ah = Path("scripts/audit_highlight.py").read_text(encoding="utf-8")
    check("HIGHLIGHT_TARGET_VERSION" in ah and "recycle_state" in ah,
          "产物体检脚本复用 db 的目标版本与 mover 的三态判据（不另立口径）")
    check("版本已是目标值却仍为空" in ah,
          "体检脚本单独标出「版本已是目标值却产物为空」的可疑行（这类只能靠 bump 版本号纠偏）")
    m_src = Path("src/mover.py").read_text(encoding="utf-8")
    check('fresh = db.get_video(row["id"])' in m_src,
          "move_one 回库取新鲜行判状态（不信调用方手里那份剪辑前快照）")
    alb = Path("scripts/align_subdirs.py").read_text(encoding="utf-8")
    check("--apply" in alb and "dry-run" in alb.lower(),
          "存量对齐脚本默认只出计划，--apply 才动手")
    check("mover.recycle_state" in alb and "def recycle_state(" in m_src
          and "def is_pending(" in m_src,
          "对齐脚本复用 mover 的判据，不另写一套")


def test_audit_highlight_states():
    """产物体检报告里的"三态"列**只统计已在回收目录的行**。

    三态判据属于回收目录语境（回答"这一条现在能不能删"），对全库行套用会把
    "还没剪过"报成"待保全"。2026-09-13 实测：789 条 `human_decision='keep'`
    里有 392 条从未剪过，全都会被误报 —— 这就是要按 `trash_path` 先过滤的原因。
    症状诊断（`symptom_of`）反过来必须**看全库**，两件事不能混。
    """
    section("产物体检：三态列只在回收目录语境里统计")
    import importlib.util

    p = Path("scripts/audit_highlight.py")
    check(p.exists(), "audit_highlight.py 在位")
    spec = importlib.util.spec_from_file_location("_ah", p)
    ah = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ah)          # 顺带验语法/导入

    rows = [
        {"id": 1, "trash_path": r"G:\need-del\A_高置信_可批量删除\2015\x.mp4",
         "highlight_at": None},                                  # 在回收目录、没剪过
        {"id": 2, "trash_path": None, "highlight_at": None},      # 没搬进回收目录
        {"id": 3, "trash_path": "", "highlight_at": None},        # 空串也算没搬
    ]
    st = ah.recycle_states(rows)
    check(st == {mover.STATE_PENDING: 1},
          f"未搬进回收目录的行进不了三态列（得到 {st}）")

    check(ah.symptom_of({"highlight_at": None}) == "没剪过"
          and ah.symptom_of({"highlight_at": 1, "highlight_clips": "[]",
                             "highlight_coarse_hits": 5}) == "零产物+有命中",
          "symptom_of 仍对全库行生效（症状诊断与回收语境是两件事）")


def test_highlight_batch():
    """L2.8 批处理：队列元素是 sqlite3.Row（没有 .get）时不得崩溃。

    真实任务曾在此挂掉：run_batch 循环里 row 未转 dict，写 manifest 时
    `row.get("trash_path")` 抛 AttributeError；且该行在 try 之外，
    1 条坏数据会炸掉整批（1372 段跑几小时）。用假 db + 假 cutter 复现该路径。
    """
    section("精华剪辑批处理")
    from src import highlighter as H

    # 用真实 sqlite 产出真正的 sqlite3.Row，保证与线上类型完全一致
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE v (id INTEGER, trash_path TEXT)")
    conn.executemany("INSERT INTO v VALUES (?,?)",
                     [(1, r"G:\need-del\a.mp4"), (2, None)])
    rows = conn.execute("SELECT * FROM v").fetchall()
    check(isinstance(rows[0], sqlite3.Row) and not hasattr(rows[0], "get"),
          "队列元素确为 sqlite3.Row（无 .get 方法）")

    marked: list = []
    errs: list = []

    class _DB:
        def highlight_queue(self, limit=5000, batch_id=None):
            return rows

        def mark_highlight(self, vid, ts, n, sec, clips, **kw):
            marked.append((vid, n))

        def mark_highlight_error(self, vid, msg):
            errs.append((vid, msg))

        # 2026-10-04：_dispose_one 的自动保留改走 set_machine_decision（带
        # decided_by='auto'）后，这个替身缺方法会 AttributeError。补上。
        def set_machine_decision(self, vid, dec, *a, **kw):
            pass

        def set_human_decision(self, vid, dec, *a, **kw):
            pass

    class _Cutter(H.HighlightCutter):
        """只 stub 掉耗时的检测/切割，保留 run_batch 的记账与写 manifest 逻辑。"""

        def __init__(self, cfg):
            self.cfg = cfg
            self.h = cfg.get("highlight", {})   # run_batch 的自动处置(_dispose_one)会读

        def cut_one(self, row, hdir, ts, log=None):
            return {"ok": True, "id": row["id"], "stem": "s", "year": "2015",
                    "src": "x", "duration": 10.0, "sec": 1.0, "kept_ratio": 0.1,
                    "stats": {},
                    "clips": [{"grade": "A", "file": "f.mp4", "thumb": "t.jpg"}]}

    cfg = load_config()
    tmp = Path(tempfile.mkdtemp(prefix="hl-batch-"))
    cfg["highlight"]["dir"] = str(tmp)
    try:
        # 审核减负（ADR 0001）：run_batch 剪完会自动处置（_dispose_one）。处置走真实的
        # mover.move_one 会碰文件系统/缺字段 —— 这里把 mover 的移动原语换成假实现，
        # 只验证「批处理记账 + 自动处置」的编排逻辑本身。
        from src import mover as M
        _real_move_one, _real_decide = M.move_one, M.decide_tier
        M.move_one = lambda db, row, cfg, tier, log=None, clips=None: (True, "fake-moved")
        M.decide_tier = lambda row, cfg: "A"
        try:
            res = _Cutter(cfg).run_batch(_DB(), log=lambda s: None)
        finally:
            M.move_one, M.decide_tier = _real_move_one, _real_decide
        check(res["failed"] == 0,
              f"Row 输入不再崩溃（failed={res['failed']}，errors={res['errors']}）")
        check(res["done"] == 2, f"两条都入账（done={res['done']}）")
        check(len(marked) == 2, f"写库 2 次（{len(marked)}）")
        dsp = res.get("disposed", {})
        check(dsp.get("moved") == 2 and dsp.get("not_disposed") == 0,
              f"自动处置：两条无精华候选都移入回收（{dsp}）")
        man = tmp / "manifest.jsonl"
        lines = [l for l in man.read_text(encoding="utf-8").splitlines() if l.strip()]
        check(len(lines) == 2, f"manifest 写全 2 行（{len(lines)}）")
        check(lines and "trash_path" in json.loads(lines[0]),
              "manifest 记录含 trash_path 字段")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_adr0001_cut_then_delete():
    """审核减负（ADR 0001 先剪后删）回归：
      队列谓词放宽 / 0 分桶 / _dispose_one 处置分支 / undo_move 清理不变量 / UI 接线。
    """
    section("审核减负 ADR 0001：0 分桶 / 处置候选队列 / 自动处置")
    from src import highlighter as H

    # ---------- A. 队列谓词放宽：review 桶入队、keep 出队 ----------
    d, db, bid, ids = _tmp_db(6)
    try:
        def _an(v, verdict="trash", score=0.0, trace=None, speech=False, moved=False):
            db.upsert_video_meta(db.get_video(v)["path"], {
                "filename": db.get_video(v)["filename"], "rel_dir": "", "year": "2024",
                "size_bytes": 1, "mtime": 0, "duration_sec": 60,
                "width": 1920, "height": 1080, "video_codec": "hevc",
                "has_audio": 1 if speech else 0, "audio_energy": None,
                "bitrate_kbps": 1000, "meta_ok": 1, "meta_error": None})
            _mark_analyzed(db, v, verdict, score)
            db.update_analysis(v, {"status": "analyzed", "retention_score": score,
                                   "final_score": score, "verdict": verdict,
                                   "decision_trace": trace or {}})
            if speech:
                db.update_audio_review(v, {"audio_checked": 1, "speech_present": 1,
                                           "speech_ratio": 0.2,
                                           "transcript_text": "画外有人声"})
            if moved:
                db.set_human_decision(v, "trash",
                                      trash_path=str(d / "t" / f"v{v}.mp4"), trash_tier="A")

        _an(ids[0], "review", 5.0, speech=True)                 # review 桶未处置 + 语音
        _an(ids[1], "trash", 5.0, moved=True, speech=True)      # 已移入回收 + 语音
        _an(ids[2], "review", 5.0)                               # review 桶（无语音）
        _an(ids[3], "trash", 5.0)                                # 待处置候选
        db.set_human_decision(ids[3], "trash")                   # → 仅逻辑标记 trash 未移动
        _an(ids[4], "review", 5.0, speech=True)                  # review 桶 + 有效语音
        db.set_human_decision(ids[4], "keep")                    # → 人工保留
        _an(ids[5], "trash", 5.0, moved=True)                    # 已移入回收（无语音）
        db.mark_highlight(ids[5], "2026-09-08 12:00:00", 0, 0.0, [])  # 已剪过 → 出队

        hq = {r["id"] for r in db.highlight_queue()}
        check(hq == {ids[0], ids[1], ids[2]},
              f"精华队列放宽：review 桶可入队（{sorted(hq)}）；"
              "逻辑 trash/keep/已剪辑出队")
        aq = {r["id"] for r in db.archive_queue()}
        check(aq == {ids[0], ids[1]},
              f"归档队列放宽：未移入回收的 review 桶 + 有效语音也能归档（{sorted(aq)}）")
        check(all(r["highlight_at"] is None for r in db.highlight_queue())
              and all(r["speech_present"] == 1 for r in db.archive_queue()),
              "过滤条件对已移入回收目录的分支同样生效（OR/AND 优先级 bug 已修复）")
        check(ids[3] not in hq and ids[4] not in hq,
              "human 已表态（trash 未移动 / keep）不再进入处置候选")
    finally:
        shutil.rmtree(d, ignore_errors=True)

    # ---------- B. 放行候选（ADR 0003）：无人物信号 / near_miss 排除 / 分数与处置过滤 ----------
    d, db, bid, ids = _tmp_db(9)
    try:
        # 注意：update_analysis 是**整行覆盖**（未传的列会被写成 NULL）——
        # 部分传参会让 verdict/retention_score 一起消失，测试就测不到想测的分支。
        # 所以这里统一走 _upd，每次都把判定字段补全。
        def _upd(v, **kw):
            base = {"status": "analyzed", "retention_score": 0.0, "final_score": 0.0,
                    "verdict": "trash", "decision_trace": {}, "has_person": 0}
            base.update(kw)
            db.update_analysis(v, base)

        for v in ids:
            _an(v, "trash", 0.0)
            _upd(v)
        # ids[0..2]：0 分且无任何人物信号 → 放行
        # ids[3]   ：near_miss 含人物（"看到了没敢确认"，漏检高发区）→ 必须排除
        _upd(ids[3], decision_trace={"near_miss": {"person": 0.20}})
        # ids[4]   ：has_person=1 → 排除（语义条件的正面判据）
        _upd(ids[4], has_person=1)
        # ids[5]   ：分数超门槛 → 排除
        _upd(ids[5], retention_score=40.0, final_score=40.0)
        # ids[6]   ：verdict=review 且无人物信号、15 分 → 放行（比 ADR 0002 的 0 分桶宽）
        _upd(ids[6], verdict="review", retention_score=15.0, final_score=15.0)
        # ids[7]   ：已人工处置（已移入回收）→ 排除
        db.set_human_decision(ids[7], "trash", trash_path=str(d / "x.mp4"), trash_tier="A")
        # ids[8]   ：meta_ok=0（分析没跑完）→ 排除
        with db.conn() as c:
            c.execute("UPDATE videos SET meta_ok=0 WHERE id=?", (ids[8],))

        q = {r["id"] for r in db.release_queue(limit=100, threshold=30,
                                               require_no_person=True)}
        check(q == {ids[0], ids[1], ids[2], ids[6]},
              f"放行候选命中 {sorted(q)}"
              "（near_miss 有人/检出人/超门槛/已处置/未分析全排除）")
        check(db.count_release(threshold=30, require_no_person=True) == 4,
              f"count_release = {db.count_release(threshold=30, require_no_person=True)}")
        check(db.count_release(batch_id=99999) == 0, "按不存在的批次过滤 → 0")

        # 分数门槛仍生效：门槛 10 → ids[6]（15 分）出局，说明"语义条件"没有吃掉"门槛"
        q10 = {r["id"] for r in db.release_queue(limit=100, threshold=10,
                                                 require_no_person=True)}
        check(q10 == {ids[0], ids[1], ids[2]},
              f"分数门槛生效：15 分的 review 档被挡在门槛外（{sorted(q10)}）")

        # 兼容口径：门槛 0 且关闭语义条件 → 回到 ADR 0002 之前的"仅 0 分桶"
        q0 = {r["id"] for r in db.release_queue(limit=100, threshold=0,
                                                require_no_person=False)}
        check(q0 == {ids[0], ids[1], ids[2], ids[3], ids[4]},
              f"退回兼容口径（门槛 0 + 不看人物）：{sorted(q0)}")

        # 放行候选仍计入待审核（放行前不消失）
        check(db.count_review_queue() >= 4, "放行候选仍计入待审核总数（放行前）")
    finally:
        shutil.rmtree(d, ignore_errors=True)

    # ---------- C. _dispose_one 各分支（mover 原语打桩） ----------
    from src import mover as M
    calls = {"move": 0, "undo": 0, "keep": 0, "move_clips": None}
    _real = (M.move_one, M.undo_move, M.decide_tier)

    def _fake_move(db, row, cfg, tier, log=None, clips=None, state=None, by="auto"):
        # clips 是「已剪辑」子档的落点依据 —— 处置链路必须把**本次新剪出**的产物
        # 传下来（row 里那份是剪辑前的旧快照）。这里顺带把它记下来做断言。
        calls["move"] += 1
        calls["move_clips"] = clips
        calls["move_by"] = by
        return True, "fake-moved"

    def _fake_undo(db, row, log=None, by="human"):
        calls["undo"] += 1
        calls["undo_by"] = by
        return True, "fake-undone"

    def _fake_tier(row, cfg):
        return "A"

    class _DB:
        def set_human_decision(self, vid, dec, **kw):
            calls["keep"] += 1

        def set_machine_decision(self, vid, dec, **kw):
            calls["keep"] += 1

    dbx = _DB()
    M.move_one, M.undo_move, M.decide_tier = _fake_move, _fake_undo, _fake_tier
    try:
        def _cut(**hcfg):
            cutter = object.__new__(H.HighlightCutter)   # 跳过 __init__（不碰 ffmpeg/模型）
            cutter.cfg = {"audio": {"enabled": hcfg.pop("audio_on", False)},
                          "highlight": {"keep_full_ratio": 0.6,
                                        "auto_dispose": hcfg.pop("auto", True)}}
            cutter.h = cutter.cfg["highlight"]
            return cutter

        def _row(**kw):
            base = {"id": 1, "human_decision": None, "has_audio": 0,
                    "audio_checked": None, "path": "v.mp4", "retention_score": 0,
                    "verdict": "trash"}
            base.update(kw)
            return base

        def _r(ratio, ok=True, clips=None):
            return {"ok": ok, "kept_ratio": ratio, "clips": clips or []}

        c = _cut()
        check(c._dispose_one(dbx, _row(), _r(0.5, ok=False))[0] == "skip",
              "dispose：剪辑失败 → skip（不动任何状态）")
        check(c._dispose_one(dbx, _row(human_decision="keep"), _r(0.9))[0]
              == "not_disposed" and not calls["move"],
              "dispose：人工已 keep → 不处置（产物只是额外保险）")
        check(_cut(auto=False)._dispose_one(dbx, _row(), _r(0.9))[0]
              == "not_disposed",
              "dispose：auto_dispose 关闭 → 留给人工")

        calls["keep"] = 0
        act, note = c._dispose_one(dbx, _row(), _r(0.85))
        check(act == "keep_full" and calls["keep"] == 1 and not calls["move"],
              f"dispose：精华占比 85% > 60% → 自动整片保留（{note}）")

        # 音频口径统一为"事后补"（ADR 0004）：不再拦"有音轨但没复核过"的候选 ——
        # 音频队列已放宽为处置候选，移入回收后仍能补复核/补归档（trash_path 回落）。
        calls["move"] = 0
        act, note = _cut(audio_on=True)._dispose_one(
            dbx, _row(has_audio=1, audio_checked=None), _r(0.1))
        check(act == "moved" and calls["move"] == 1,
              f"dispose：有音轨未复核 → 照常移入（音频走事后补，不拦住）（{note}）")

        calls["move"] = 0
        act, note = _cut(audio_on=True)._dispose_one(
            dbx, _row(has_audio=1, audio_checked=1), _r(0.1))
        check(act == "moved" and calls["move"] == 1,
              f"dispose：音频已复核（audio_checked=1）→ 剪完自动移入回收（{note}）")

        calls["undo"] = 0
        act, note = c._dispose_one(dbx, _row(human_decision="trash"), _r(0.85))
        check(act == "undo_keep" and calls["undo"] == 1,
              f"dispose：已放行但整片有内容 → 撤销移动捞回（{note}）")
        # ★ decided_by 溯源：机器捞回必须记 'auto'。undo_move 的默认是 'human'
        # （撤销多数由人触发），这里不显式传就会把机器行为伪装成人工决策，
        # 抽查页"机器判得准不准"的统计真源被污染 —— 全项目最隐蔽的一个坑。
        check(calls.get("undo_by") == "auto",
              f"★机器捞回 undo_move(by='auto')（实得 {calls.get('undo_by')!r}）")

        # 落点依据必须是**本次新剪出**的 clips：row 里那份是剪辑前的旧快照，
        # 拿它判断会把刚剪出 A/B 的原片错放进普通档位（「已剪辑」标记就废了）
        calls["move"] = 0
        _clip = [{"grade": "A", "file": "x.mp4", "dur": 9.0}]
        act, _ = _cut(audio_on=True)._dispose_one(
            dbx, _row(has_audio=0, audio_checked=1), _r(0.1, clips=_clip))
        check(act == "moved" and calls["move_clips"] == _clip,
              "dispose：把本次新剪出的 clips 透传给 move_one（决定「已剪辑」落点）")

        calls["move"] = 0
        act, note = c._dispose_one(dbx, _row(human_decision="trash"), _r(0.1))
        check(act == "not_disposed" and calls["move"] == 0,
              f"dispose：已处置为 trash 且无高占比 → 不再重复移动（{note}）")
    finally:
        M.move_one, M.undo_move, M.decide_tier = _real

    # ---------- D. undo_move：真库真文件，清 trash_path 不变量 ----------
    d, db, bid, ids = _tmp_db(1)
    try:
        vid = ids[0]
        src = Path(db.get_video(vid)["path"])
        src.write_bytes(b"fake-video")
        cfg = {"work_dir": str(d), "trash_root": str(d / "need-del")}
        row = db.get_video(vid)
        ok, _ = mover.move_one(db, dict(row), cfg, "A")
        row = db.get_video(vid)
        check(ok and row["human_decision"] == "trash"
              and row["trash_path"] and not Path(row["path"]).exists()
              and Path(row["trash_path"]).exists(),
              "move_one：文件移入回收，human=trash + trash_path 落库")

        ok2, _ = mover.undo_move(db, row)
        row = db.get_video(vid)
        check(ok2 and Path(row["path"]).exists()
              and not (row["trash_path"] and Path(row["trash_path"]).exists()),
              "undo_move：文件回到原位")
        check(row["human_decision"] == "keep" and row["trash_path"] is None
              and row["trash_tier"] is None,
              "undo_move：keep 行 trash_path/trash_tier 清空（不残留失效回收路径）")
        check(vid not in {r["id"] for r in db.highlight_queue()}
              and vid not in {r["id"] for r in db.archive_queue()}
              and vid not in {r["id"] for r in db.list_moved()},
              "undo 后不再进入任何处置候选/可撤销队列（无二次处理）")

        # 处置翻转：keep 后再标记 trash（无路径）不得继承旧路径
        db.set_human_decision(vid, "trash")
        row = db.get_video(vid)
        check(row["human_decision"] == "trash" and row["trash_path"] is None,
              "处置翻转不继承失效 trash_path（曾因 COALESCE 残留错报清单）")
    finally:
        shutil.rmtree(d, ignore_errors=True)

    # ---------- E. UI 接线（源码级，防改 UI 忘接后端） ----------
    ui_src = Path("src/ui.py").read_text(encoding="utf-8")
    db_src = Path("src/db.py").read_text(encoding="utf-8")
    check("_render_release_candidates" not in ui_src,
          "★「一键放行」卡已从 UI 移除（v3 阶段 4）—— 放行是**程序动作**，"
          "由一键执行自动跑；留在人工页上等于把机器该做的又交回给人")
    check('一键放行' not in ui_src,
          "UI 不再有「一键放行」按钮（能力保留在 jobs._run_release）")
    # 放行的真实入口：一键执行的第 5 阶段
    check("_run_release" in Path("src/jobs.py").read_text(encoding="utf-8")
          and "release" in Path("src/jobs.py").read_text(encoding="utf-8"),
          "jobs：放行保留在一键执行链路里（程序自动跑）")
    check("move_all" in Path("src/mover.py").read_text(encoding="utf-8"),
          "mover：move_all 仍是批量移动的原语")
    check("_PENDING_OR_TRASHED" in db_src
          and "human_decision IS NULL AND verdict IN ('trash','review')" in db_src,
          "db：archive/highlight 共用放宽后的处置候选谓词")
    check('60%' in ui_src and '整片保留' in ui_src and '先剪后删' in ui_src,
          "UI：设置页精华剪辑 caption 说明自动保留/自动移入规则")
    check('r.get("disposed", {})' in ui_src or "disposed" in ui_src,
          "UI：任务完成摘要展示自动处置统计")


def test_adr0004_pipeline():
    """一键执行（ADR 0004）回归：阶段顺序 / 前置总检 / 阶段调度 / 失败即停 / UI 接线。"""
    section("一键执行 ADR 0004：阶段串联 / 前置检查 / 失败即停")
    import copy

    from src import jobs as J
    from src.config import DEFAULT_CONFIG

    # ---------- A. 阶段顺序由依赖决定，不是随意排的 ----------
    keys = [k for k, _ in J.PIPELINE_STAGES]
    check(keys == ["scan", "analyze", "audio", "archive", "release", "highlight",
                   "align"],
          f"阶段顺序：{keys}")
    check(keys.index("analyze") < keys.index("audio") < keys.index("archive"),
          "依赖：分析先于音频复核，音频复核先于语音归档")
    check(keys.index("release") < keys.index("highlight"),
          "依赖：放行必须在剪辑之前（反序会被剪辑的自动处置清空成空操作）")
    check(keys.index("highlight") < keys.index("align"),
          "依赖：三堆对齐必须在剪辑之后（对齐按保全状态归档，早于剪辑跑等于白搬）")

    # ---------- B. 前置总检：缺什么拦什么，缺一个就拒绝启动 ----------
    base = copy.deepcopy(DEFAULT_CONFIG)
    base["work_dir"] = ""
    base["source_root"] = ""
    base["trash_root"] = ""
    base["audio"]["enabled"] = False
    allon = {"stages": {k: True for k in keys}}
    probs = J.pipeline_precheck(base, allon)
    check(any("work_dir" in p for p in probs)
          and any("source_root" in p for p in probs)
          and any("trash_root" in p for p in probs),
          f"前置检查：缺 work_dir / 源目录 / 回收目录都被拦（共 {len(probs)} 条）")
    check(any("音频复核未启用" in p for p in probs),
          "前置检查：勾了音频阶段但 enabled=false → 明确提示（并告知可取消勾选）")

    base["work_dir"] = "."
    check(J.pipeline_precheck(base, {"stages": {k: False for k in keys}}) == [],
          "前置检查：阶段全不勾 → 只剩 work_dir 一条硬条件，不再误报")
    check(J.pipeline_precheck(base, {"stages": {k: (k == "analyze") for k in keys}}) == [],
          "前置检查：只跑分析时不误报目录/引擎缺失（按勾选范围查）")

    # ---------- C. 阶段调度：顺序 / 跳过 / 失败即停 / 停止 ----------
    cfg = {"work_dir": "."}
    opts = {"stages": {"scan": True, "analyze": True, "audio": True,
                       "archive": False, "release": True, "highlight": True,
                       "align": True},
            "scan_limit": 10, "scan_with_audio": False, "stage_limit": 100}
    real_stage = J._run_stage
    calls = []

    def _fake_stage(job, db, cfg, key, opts):
        calls.append(key)
        job.result = {"fake": key}
        return f"{key} 完成"

    job = J.Job("pipeline")
    J._run_stage = _fake_stage
    try:
        J._run_pipeline(job, None, cfg, opts)
    finally:
        J._run_stage = real_stage
    check(calls == ["scan", "analyze", "audio", "release", "highlight", "align"],
          f"调度：未勾选的 archive 被跳过，其余按依赖顺序跑（{calls}）")
    check(job.stage_count == 6 and job.stage_index == 6,
          f"面板阶段计数：{job.stage_index}/{job.stage_count}（7 阶段里 archive 未勾 → 分母 6）")
    check([s["key"] for s in job.result["stages"]] == calls
          and not job.result["aborted"],
          "结果：逐阶段摘要齐全、未标记中断")

    def _boom(job, db, cfg, key, opts):
        calls.append(key)
        if key == "audio":
            raise J.PipelineError("引擎未就绪")
        return "ok"

    calls.clear()
    job2 = J.Job("pipeline")
    err = ""
    J._run_stage = _boom
    try:
        J._run_pipeline(job2, None, cfg, opts)
    except J.PipelineError as e:      # noqa: PERF203 故意接住，验证"停在哪"
        err = str(e)
    finally:
        J._run_stage = real_stage
    check(calls == ["scan", "analyze", "audio"] and "音频复核" in err,
          f"失败即停：音频复核抛错后 release/highlight 一个都没跑（{calls}）")
    check(job2.result.get("aborted") and job2.result.get("stopped_at") == "音频复核",
          "失败后结果写明停在哪一步（面板据此提示「修好再点一次」）")
    check(job2.result["stages"][-1]["status"] == "error",
          "失败阶段的摘要状态为 error（面板按 ❌ 渲染）")

    def _stop_after_first(job, db, cfg, key, opts):
        calls.append(key)
        job._stop.set()               # 模拟用户在第一阶段跑完后点了停止
        return "ok"

    calls.clear()
    job3 = J.Job("pipeline")
    J._run_stage = _stop_after_first
    try:
        J._run_pipeline(job3, None, cfg, opts)
    finally:
        J._run_stage = real_stage
    check(calls == ["scan"], f"停止：当前阶段收尾后不再进入下一阶段（{calls}）")

    # ---------- D. 启动入口：前置不过就拒绝，且不留僵尸 running ----------
    ok, msg = J.start_pipeline(None, base,
                               {"stages": {k: (k == "audio") for k in keys}})
    check(not ok and "音频复核未启用" in msg,
          f"start_pipeline：前置不过直接拒绝并列出原因（{msg.splitlines()[0]}）")
    check(not J.get_pipeline_job().running, "被拒绝时不留下 running 状态的僵尸任务")

    # ---------- E. any_running 必须把流水线算进去 ----------
    pj = J.get_pipeline_job()
    pj.status = "running"
    pj._thread = type("T", (), {"is_alive": lambda self: True})()
    check(J.any_running(),
          "any_running：一键执行在跑时，其它启动按钮/放行按钮必须一起禁用")
    pj.__init__("pipeline")           # 复位，避免污染后续测试
    check(not J.any_running(), "复位后不再算运行中")

    # ---------- F. UI / jobs 接线（源码级，防改后端忘接线） ----------
    ui_src = Path("src/ui.py").read_text(encoding="utf-8")
    jobs_src = Path("src/jobs.py").read_text(encoding="utf-8")
    check("def _render_pipeline" in ui_src and "_render_pipeline(db, cfg)" in ui_src
          and "⚡ 一键执行" in ui_src,
          "UI：设置页置顶「一键执行」区")
    check("jobsmod.start_pipeline(db, cfg, opts)" in ui_src
          and "⚡ 一键执行（跑完到「回收目录就绪」）" in ui_src,
          "UI：一个大按钮 → start_pipeline")
    check("_render_step_by_step(db, cfg, limit)" in ui_src
          and "高级 / 分步执行" in ui_src,
          "UI：原分散段落折叠进「高级 / 分步」")
    check("{job.stage_index}/{job.stage_count}" in ui_src
          and "逐阶段列结果" in ui_src,
          "UI：面板显示当前阶段与逐阶段结果（否则 3/7 的进度条会被误读）")
    check("_PIPELINE = \"pipeline\"" in jobs_src and "def start_pipeline" in jobs_src
          and "def pipeline_precheck" in jobs_src and "PIPELINE_STAGES" in jobs_src,
          "jobs：流水线任务类型 / 前置检查 / 启动入口齐备")
    check('if key == "align":' in jobs_src and "align_subdirs" in jobs_src
          and 'st.get("align", True)' in jobs_src,
          "jobs：三堆对齐阶段复用 mover.align_subdirs（与手点「对齐子档」同一入口），"
          "且前置检查把 align 纳入 trash_root 硬条件")
    check("待归位" in ui_src and "PIPELINE_STAGES[0:4]" in ui_src,
          "UI：一键执行区显示「待归位」指标、七个阶段开关两行排布（4+3）")


def main() -> int:
    cfg = load_config()
    test_misjudgment(cfg)
    test_tier(cfg)
    test_batch_progress()
    test_paging()
    test_reports(cfg)
    test_jobs()
    test_hotkey_binding()
    test_no_delete_calls()
    test_model_verify_sizes()
    test_config_resilience()
    test_audio_review()
    test_archive()
    test_grid_audio_text()
    test_grid_video_play()
    test_rerun_analysis()
    test_review_queue_selfheal()
    test_review_jump_nav()
    test_coarse_suspect_kept()
    test_isolated_hit_not_dropped()
    test_hl_version_single_source()
    test_c_cut_mp4()
    test_coarse_zero_hits_no_crash()
    test_c_not_in_review()
    test_l1_quality_scoring()
    test_l1_floor_safety()
    test_l1_mode_switch()
    test_decided_by()
    test_spotcheck_sampling()
    test_spotcheck_rejudge()
    test_no_clip_split()
    test_rehighlight_idempotent()
    test_failed_retry_recovers_halfdone()
    test_highlight_disk_guard()
    test_recycle_subdirs()
    test_audit_highlight_states()
    test_highlight_batch()
    test_adr0001_cut_then_delete()
    test_adr0004_pipeline()

    print()
    print("=" * 66)
    print(f"结果：{'全部通过' if _failures == 0 else f'{_failures} 项失败'}")
    print("=" * 66)
    return 1 if _failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
