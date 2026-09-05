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

    print()
    print("=" * 66)
    print(f"结果：{'全部通过' if _failures == 0 else f'{_failures} 项失败'}")
    print("=" * 66)
    return 1 if _failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
