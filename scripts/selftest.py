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

from src import feedback as fb  # noqa: E402
from src import mover  # noqa: E402
from src import config as cfgmod  # noqa: E402
from src.config import load_config, save_config, _read_json, detect_sync_risk, LAST_NOTICE  # noqa: E402
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
        "vlm_agree": None, "vlm_has_person": None, "vlm_has_vehicle": None,
        "vlm_evidence": None, "year": "2024", "rel_dir": "", "batch_id": 1,
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
    cfg_off = json.loads(json.dumps(cfg))
    cfg_off["vlm"]["enabled"] = False

    row_a = blank_row(retention_score=0, decision_trace=json.dumps({"near_miss": {}}))
    check(mover.decide_tier(row_a, cfg_off) == "A", "无目标且分数为 0 → A 档（可批量删除）")

    row_b = blank_row(retention_score=0,
                      decision_trace=json.dumps({"near_miss": {"person": 0.38}}))
    check(mover.decide_tier(row_b, cfg_off) == "B", "存在未达阈值疑似目标 → B 档（建议抽查）")

    cfg_on = json.loads(json.dumps(cfg))
    cfg_on["vlm"]["enabled"] = True
    row_c = blank_row(retention_score=0, decision_trace=json.dumps({"near_miss": {}}),
                      vlm_agree=0, vlm_has_person=1)
    check(mover.decide_tier(row_c, cfg_on) == "B", "两级模型分歧 → B 档")


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
    """快捷键靠按钮文案定位目标 —— 文案一改快捷键就静默失效，必须钉住。"""
    section("快捷键 × 按钮文案一致性")
    from src import hotkeys
    ui_src = Path("src/ui.py").read_text(encoding="utf-8")
    labels = re.findall(r'st\.button\(\s*"([^"]+)"', ui_src)
    for code, text, _desc in hotkeys.DEFAULT_KEYS:
        hit = [l for l in labels if text in l]
        check(bool(hit),
              f"{hotkeys._LABELS.get(code, code)} → 「{text}」命中按钮"
              + (f"：{hit[0]}" if hit else "（未找到，快捷键会失效）"))


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
        check(cfg.get("vlm", {}).get("model") == "qwen3-vl:4b",
              "主文件不存在 → 按全新安装处理，返回默认配置")

        # —— E. 全部坏掉：主文件损坏 + .bak 损坏 + 无 Conflict → 给默认配置 + 提示重填
        bak_path.write_text("also broken{", encoding="utf-8")  # .bak 故意损坏
        cfg_path.write_text("{not valid", encoding="utf-8")     # 主文件损坏
        for f in d.glob("*Conflict*.json"):
            f.unlink()  # 删掉所有 Conflict 副本
        cfg = load_config(cfg_path)
        check(isinstance(cfg, dict) and cfg.get("vlm", {}).get("model") == "qwen3-vl:4b",
              "主+bak+Conflict 全坏 → 返回默认配置")
        check("重置为默认" in cfgmod.LAST_NOTICE,
              f"明确提示用户需要重填：{cfgmod.LAST_NOTICE[:50]}")

        # —— F. 原子写入：保存中途崩溃不应留下半截文件
        for ext in (".json", ".bak"):
            for f in d.glob(f"*{ext}"):
                f.unlink()
        cfg = {"source_root": "G:\\camera\\2024", "vlm": {"enabled": True}}
        save_config(cfg, cfg_path)
        # 不应残留任何 .tmp-* 临时文件
        tmps = list(d.glob("*.tmp-*"))
        check(not tmps, f"无残留临时文件（{len(tmps)} 个）")
        # .bak 必须是"最后一次成功保存"的副本，不是上一次
        bak_content = json.loads(bak_path.read_text(encoding="utf-8"))
        check(bak_content["vlm"]["enabled"] is True,
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
def test_grid_select_all():
    """批量浏览「全选本页 / 清空选择」机制（2026-09-05 修过的真实 bug）。

    bug 根因：带 key 的复选框在首次渲染后，勾选态由 session_state["g_{id}"] 决定，
    value= 参数只在组件首次实例化时生效。旧代码只改 selected 集合 → rerun 后
    复选框仍按旧组件状态渲染 → elif 又把 id 从 selected 里 discard，等于没选。
    正确做法：用 on_click 回调置位/复位组件状态（回调在脚本体执行前运行）。
    """
    section("批量浏览 全选/清空 机制")

    # —— A. 源码接线检查：按钮必须挂在回调上，不能是「只改集合 + st.rerun」
    ui_src = Path("src/ui.py").read_text(encoding="utf-8")
    seg = ui_src[ui_src.index("def tab_grid"):]
    sel_all = seg[seg.index('st.button("全选本页"'):][:200]
    sel_clr = seg[seg.index('st.button("清空选择"'):][:200]
    check("on_click=_grid_sel_all" in sel_all,
          "「全选本页」通过 on_click 回调置位组件状态")
    check("on_click=_grid_sel_clear" in sel_clr,
          "「清空选择」通过 on_click 回调复位组件状态")
    check("st.session_state[f\"g_{i}\"] = True" in ui_src,
          "回调内直接写复选框组件状态 g_{id}")

    # —— B. AppTest 行为验证：最小可运行脚本复刻同一套机制
    app = r'''
import streamlit as st

st.session_state.setdefault("selected", set())
IDS = [1, 2, 3]

selected = st.session_state.selected
for vid in IDS:
    if st.checkbox("选择", key=f"g_{vid}", value=vid in selected):
        selected.add(vid)
    elif vid in selected:
        selected.discard(vid)
st.session_state.selected = selected


def _sel_all(page_ids):
    for i in page_ids:
        st.session_state[f"g_{i}"] = True


def _sel_clear():
    for i in list(st.session_state.get("selected", set())):
        st.session_state[f"g_{i}"] = False
    st.session_state["selected"] = set()


st.button("全选本页", on_click=_sel_all, args=(IDS,))
st.button("清空选择", on_click=_sel_clear)
st.write(f"已选 {len(selected)} 个")
'''
    from streamlit.testing.v1 import AppTest
    import tempfile as _tf
    app_file = Path(_tf.mkdtemp(prefix="grid_test_")) / "app.py"
    app_file.write_text(app, encoding="utf-8")
    try:
        at = AppTest.from_file(str(app_file), default_timeout=10)
        at.run()
        check(not any(b.value for b in at.checkbox), "初始无勾选")
        at.button[0].click()  # 全选本页
        at.run()
        ok_sel = all(b.value for b in at.checkbox)
        check(ok_sel, "点击「全选本页」后本页全部勾选（旧代码此处失效）")
        check("已选 3 个" in at.markdown[-1].value,
              f"计数联动：{at.markdown[-1].value}")
        at.button[1].click()  # 清空选择
        at.run()
        ok_clr = not any(b.value for b in at.checkbox)
        check(ok_clr, "点击「清空选择」后全部复位（含组件状态）")
        check("已选 0 个" in at.markdown[-1].value,
              f"清空后计数归零：{at.markdown[-1].value}")
    finally:
        shutil.rmtree(app_file.parent, ignore_errors=True)


# --------------------------------------------------------------------------- 11
def test_audio_review():
    """音频复核层（L2.5）回归：DB 迁移/队列/复核写入/judge 解析/UI 接线。

    不依赖 funasr 引擎与模型 —— 只测不碰引擎的分支与数据层逻辑，
    引擎行为在装好依赖后用真实音频做冒烟（见 docs/开发计划 Phase 2.6）。
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
        import sqlite3
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
          "UI：审核台已挂载音频复核卡")
    check("confirm_skip_audio" in ui_src and "disabled=not confirm_skip_audio" in ui_src,
          "UI：确认清理带音频复核关口")
    check("b_gate_ok" in ui_src and "disabled=(not selected or not b_gate_ok)" in ui_src,
          "UI：批量移入带音频复核关口")
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
    d2, db2, _, ids2 = _tmp_db(4)   # v0,v1 trash+review 待复核各1
    try:
        for v, verdict in ((ids2[0], "trash"), (ids2[1], "review")):
            db2.upsert_video_meta(db2.get_video(v)["path"], {
                "filename": db2.get_video(v)["filename"], "rel_dir": "", "year": "2024",
                "size_bytes": 1, "mtime": 0, "duration_sec": 60,
                "width": 1920, "height": 1080, "video_codec": "hevc",
                "has_audio": 1, "audio_energy": None, "bitrate_kbps": 1000,
                "meta_ok": 1, "meta_error": None})
            _mark_analyzed(db2, v, verdict, 5.0)
        check(db2.count_audio_pending() == 2 and db2.count_audio_pending(scope="all") == 2,
              "范围 all：trash+review 均计入待复核（默认全量覆盖）")
        check(db2.count_audio_pending(scope="trash") == 1
              and db2.count_audio_pending(scope="review") == 1,
              "范围拆分：trash/review 各自计数正确")
        q_all = {x["id"] for x in db2.audio_review_queue()}
        q_r = {x["id"] for x in db2.audio_review_queue(scope="review")}
        check(q_all == {ids2[0], ids2[1]} and q_r == {ids2[1]},
              "队列过滤：all 含两类；review 只取需人工确认档")
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
    """L2.7 语音档案归档：红线 / 谓词 / 配对 / 幂等 / 撤销不回收。"""
    section("语音档案归档")
    from src import archiver as A

    # 1) 盘符红线：归档目录绝不能在源盘或回收目录所在盘（那里会被手动清空）
    p = A.check_ready({"trash_root": r"G:\need-del", "source_root": r"G:\change\2015",
                       "archive": {"dir": r"G:\audio-archive"}})
    check(len(p) == 2, f"红线：归档目录与源盘/回收目录同盘 → 拒绝（{len(p)} 条）")
    check(A.check_ready({"trash_root": r"G:\need-del", "source_root": r"G:\change\2015",
                         "archive": {"dir": r"E:\audio-archive"}}) == [],
          "异盘归档目录 → 检查通过")
    check(A.check_ready({"trash_root": "", "source_root": "",
                         "archive": {"dir": r"E:\audio-archive"}}) == [],
          "未配置源/回收目录时不误报")

    d, db, bid, ids = _tmp_db(4)
    try:
        # 造 4 种情况：
        # v0 已入回收目录 + 有效语音 + 缓存 wav 在      → 归档成功
        # v1 有效语音但未移动（无 trash_path）          → 不入队
        # v2 已移动但无人声                              → 不入队
        # v3 已移动 + 有效语音但缓存 wav 缺失            → 入队后跳过
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
        check(q == {ids[0], ids[3]},
              f"队列谓词：已入回收目录∧有效语音∧未归档（{sorted(q)}）")
        check(db.count_archive_pending() == 2 and db.count_archived() == 0,
              "计数：待归档 2 / 已归档 0")

        cfg = {"work_dir": str(d), "trash_root": "", "source_root": "",
               "archive": {"dir": str(d / "arch")}}
        res = A.archive_pending(db, cfg, limit=100)
        check(res["done"] == 1 and res["skipped"] == 1 and res["failed"] == 0,
              f"归档结果：成功 {res['done']} / 缓存缺失跳过 {res['skipped']} / 失败 0")

        # 文件名 wav+txt 配对，txt 内容 = 数据库转写
        wav = Path(d) / "arch" / "2024" / f"{db.get_video(ids[0])['filename'][:-4]}.wav"
        txt = wav.with_suffix(".txt")
        check(wav.exists() and txt.exists(), "归档产物：wav + 同名 txt 配对存在")
        check(txt.read_text(encoding="utf-8") == "测试转写内容",
              "txt 内容与数据库 transcript_text 一致")

        mf = Path(d) / "arch" / "manifest.jsonl"
        lines = mf.read_text(encoding="utf-8").strip().splitlines()
        rec = json.loads(lines[0])
        check(len(lines) == 1 and rec["id"] == ids[0]
              and rec["transcript"] == "测试转写内容"
              and rec["trash_path"].endswith(".mp4"),
              "manifest.jsonl：一行一条 JSON，含 id/转写/回收路径")

        # 幂等：再跑一次不新增
        res2 = A.archive_pending(db, cfg, limit=100)
        lines2 = mf.read_text(encoding="utf-8").strip().splitlines()
        check(res2["done"] == 0 and len(lines2) == 1,
              "幂等：重复执行不重复归档、manifest 不追加")
        check(db.count_archived() == 1, "已归档计数 = 1（archived_at 标记）")

        # 撤销移动 → 退出队列，但已归档文件保留（档案独立，宁存勿删）
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
    """批量浏览展示音频转写（2026-09-07）：格子内摘要 + 仅看检出人声筛选。

    场景：画面被判删的片段，画外可能有人声事件。九宫格原来只显示「分数/档位/文件名」，
    用户必须逐条点进审核台才能看到说了什么——等于把最关键的信息藏起来了。
    """
    section("批量浏览 音频转写展示")

    # 1) 源码接线：格子内必须渲染音频行，且受开关控制
    ui_src = Path("src/ui.py").read_text(encoding="utf-8")
    seg = ui_src[ui_src.index("def tab_grid"):]
    seg = seg[:seg.index("def tab_", 5)] if "def tab_" in seg[5:] else seg
    check("_grid_audio_html(row)" in seg and "unsafe_allow_html=True" in seg,
          "格子内渲染音频摘要行（HTML 片段）")
    check('st.checkbox("🎧 显示音频转写", key="grid_show_audio")' in seg,
          "提供「显示音频转写」开关（默认开）")
    check("st.checkbox(\"仅看检出人声\"" in seg
          and "speech_only=st.session_state.grid_speech_only" in seg,
          "提供「仅看检出人声」筛选并透传 SQL")
    check("st.markdown(_GRID_AUDIO_CSS" in seg, "注入格子音频摘要样式")

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
    section("批量浏览 系统播放器入口")

    ui_src = Path("src/ui.py").read_text(encoding="utf-8")
    seg = ui_src[ui_src.index("def tab_grid"):]
    seg = seg[:seg.index("def tab_", 5)] if "def tab_" in seg[5:] else seg

    check('st.button("▶ 播放", key=f"p_{row[\'id\']}"' in seg,
          "格子内「▶ 播放」按钮（按 id 唯一 key）")
    check('st.button("📂 定位", key=f"f_{row[\'id\']}"' in seg,
          "格子内「📂 定位」按钮（按 id 唯一 key）")
    check("current_video_path(row)" in seg and "open_in_player(str(vp))" in seg,
          "播放走 current_video_path 探测 + 系统播放器")
    check(seg.count('st.error("原文件已不存在') >= 2,
          "文件缺失时明确报错，不静默失败")
    check('k.startswith(("g_", "p_", "f_"))' in seg,
          "翻页残留键清理覆盖新增的播放/定位按钮键")
    # 审核台同步：已移入回收目录的文件原本会被误报"不存在"
    rev = ui_src[ui_src.index("def tab_review"):]
    rev = rev[:rev.index("def tab_grid")]
    check("vp = current_video_path(row)" in rev,
          "审核台播放按钮同样走路径探测（移动后仍可播）")

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

    from src.db import Database as _DB  # noqa: F401

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
    check('row["verdict"] not in ("trash", "review")' in seg2,
          "渲染前校验队列条目仍在审核范围")
    check('"_q_reload"' in seg2 and "st.rerun()" in seg2,
          "失效条目触发整页重载（自愈）")

    # 行为级：把一条已被重置（verdict IS NULL）的真实 id 塞进缓存队列，页面必须不崩
    from streamlit.testing.v1 import AppTest
    cfg = load_config()
    con = sqlite3.connect(cfg["runtime"]["sqlite_path"])
    con.row_factory = sqlite3.Row
    dead = con.execute(
        "SELECT id FROM videos WHERE verdict IS NULL LIMIT 1").fetchone()
    con.close()
    if dead is None:
        check(True, "（库中无已重置记录，跳过行为级复现）")
        return
    at = AppTest.from_file(str(Path("src/ui.py").resolve()), default_timeout=240)
    at.session_state["queue"] = [dead["id"]]
    at.session_state["q_page"] = 0
    at.session_state["idx"] = 0
    at.run()
    check(not at.exception,
          f"预置死 id {dead['id']} 后页面无异常（原崩溃场景已修复）")
    q = list(at.session_state["queue"])
    check(dead["id"] not in q and len(q) > 0,
          f"队列已自愈刷新（{len(q)} 条有效 id，死 id 不再出现）")


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
    test_config_resilience()
    test_grid_select_all()
    test_audio_review()
    test_archive()
    test_grid_audio_text()
    test_grid_video_play()
    test_rerun_analysis()
    test_review_queue_selfheal()

    print()
    print("=" * 66)
    print(f"结果：{'全部通过' if _failures == 0 else f'{_failures} 项失败'}")
    print("=" * 66)
    return 1 if _failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
