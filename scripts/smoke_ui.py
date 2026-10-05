"""UI 渲染冒烟：用 Streamlit 官方 AppTest 把整页跑一遍，抓渲染期异常。

为什么需要它：源码级接线检查（selftest 里的字符串断言）只能证明"函数被调用了"，
证明不了"页面能渲染出来" —— 重复 widget key、未定义变量、StreamlitAPIException
这类问题只在真正跑一次渲染时才暴露。

只读：只点「跳到分数 ≥」这一个按钮（纯 COUNT 查询 + 队列定位，不写库），
其余按钮一概不点 —— 因此不会写库、不会落盘配置。

点它的理由：审核台的跳转回调曾把形参 `filt` 写成 `fil`，一按就 NameError，
而源码级断言查不出这类"只在回调里才执行"的错误 —— 必须真点一次。

跑在**一次性临时库**上（靠 ui.main() 的 VIDEO_SEARCH_DB_OVERRIDE 钩子注入），
不碰项目真实库。原因与 selftest 同一个：真实库可能不存在（新机首次装 / 库被清），
或为空（尚未跑扫描）—— 空库时精华复核页的「无精华三义」整块不渲染（`nb["total"]==0`
是唯一的渲染条件），断言就会假失败。所以这里自己造几条"已剪辑 + 零产物"的行。
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Windows GBK 终端编不了码 emoji（按钮文案里有 ⚡ / 🗑 等），
# 让所有 print 输出能落地。
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

from streamlit.testing.v1 import AppTest  # noqa: E402


def _seed_temp_db() -> str:
    """造一个带"已剪辑 + 零产物"行的临时库，返回路径。

    必须造出这几类行，冒烟的每一项断言才有东西可看：
      - 零产物（clips='[]'）× 3 种分义 ⇒ 精华复核页的「三义」表才渲染
      - 有 A 级 mp4 产物 ⇒ 「已剪辑」档位有内容、精华抽查入口有样本
      - 未处置的 review 候选 ⇒ 抽查页 L1 入口有样本
      - l1_version='qv2' 的行 ⇒ 抽查页 L1 入口的**唯一**抽样源
      - 一行 legacy（l1_version 为 NULL）⇒ 用来验证抽样把它排除在外
    """
    import json
    from src.db import Database

    path = str(Path(tempfile.mkdtemp(prefix="smoke-")) / "smoke.db")
    db = Database(path)
    rows = [
        # (filename, verdict, human_decision, clips, sent, coarse_hits, l1ver)
        ("empty-detected.mp4", "trash", "trash", [], 30, 0, "qv2"),
        ("empty-skipped.mp4", "trash", "trash", [], 0, 0, "qv2"),
        ("empty-dropped.mp4", "trash", "trash", [], 12, 3, "qv2"),
        ("with-ab.mp4", "review", "trash",
         [{"grade": "A", "file": "G:/h/a1.mp4", "dur": 9.0, "thumb": "t.jpg",
           "thumbs": ["t.jpg"], "conf": 0.8, "n_person": 1, "score": 88.0,
           "index": 1, "start": 1.0, "end": 10.0, "level": 2, "shot_at": None,
           "ver": "hlv3"}], 40, 2, "qv2"),
        ("pending-review.mp4", "review", None, None, 25, 1, "qv2"),
        # 旧公式的行：抽查页不该抽到它
        ("legacy-row.mp4", "trash", "trash", [], 10, 1, None),
    ]
    with db.conn() as c:
        for fn, verdict, hd, clips, sent, hits, l1ver in rows:
            c.execute(
                """INSERT INTO videos (
                   path, filename, year, meta_ok, status, verdict,
                   retention_score, final_score, has_person, motion_area_ratio,
                   sampled_frames, human_decision, trash_path, trash_tier,
                   highlight_at, highlight_count, highlight_sec, highlight_clips,
                   highlight_sent, highlight_coarse_hits, highlight_version,
                   ab_clips_count, highlight_best_grade, l1_version)
                   VALUES (:path, :filename, :year, 1, 'analyzed', :verdict,
                           25.0, 25.0, 0, 0.1, 15, :human_decision,
                           :trash_path, :trash_tier, :highlight_at,
                           :highlight_count, :highlight_sec, :highlight_clips,
                           :sent, :hits, 'hlv3', :ab, :best, :l1ver)""",
                {
                    "path": f"G:/src/{fn}", "filename": fn, "year": "2015",
                    "verdict": verdict, "human_decision": hd,
                    "trash_path": f"G:/need-del/{fn}" if hd == "trash" else None,
                    "trash_tier": "A" if hd == "trash" else None,
                    "highlight_at": "2026-10-04 12:00:00",
                    "highlight_count": len(clips) if clips is not None else 0,
                    "highlight_sec": 0.0 if clips is None else 9.0,
                    "highlight_clips": json.dumps(clips, ensure_ascii=False)
                                       if clips is not None else "[]",
                    "sent": sent, "hits": hits,
                    "ab": 1 if clips else 0, "best": "A" if clips else None,
                    "l1ver": l1ver,
                })
    return path


def main() -> int:
    tmp_db = _seed_temp_db()
    old = os.environ.get("VIDEO_SEARCH_DB_OVERRIDE")
    os.environ["VIDEO_SEARCH_DB_OVERRIDE"] = tmp_db
    try:
        return _run()
    finally:
        if old is None:
            os.environ.pop("VIDEO_SEARCH_DB_OVERRIDE", None)
        else:
            os.environ["VIDEO_SEARCH_DB_OVERRIDE"] = old


def _run() -> int:
    at = AppTest.from_file(str(ROOT / "src" / "ui.py"), default_timeout=300)
    at.run()
    if at.exception:
        print("❌ 渲染抛异常：")
        for e in at.exception:
            print(f"   {e.value}")
        return 1

    labels = [b.label for b in at.button]
    want = ["⚡ 一键执行（跑完到「回收目录就绪」）", "💾 保存设置"]
    missing = [w for w in want if w not in labels]
    print(f"按钮总数：{len(labels)}")
    print("一键执行区按钮：" + ("，".join(w for w in want if w in labels) or "（无）"))
    if missing:
        print(f"❌ 缺少按钮：{missing}")
        return 1

    markdowns = " ".join(m.value for m in at.markdown)
    if "一键执行" not in markdowns:
        print("❌ 页面上没有「一键执行」区块标题")
        return 1

    # 七个阶段开关 + 关键输入框是否都在
    cbs = [c.label for c in at.checkbox]
    need_cb = ["扫描元数据", "AI 分析", "音频复核", "语音档案归档", "批量放行",
               "画面精华剪辑", "回收目录三堆对齐"]
    miss_cb = [c for c in need_cb if c not in cbs]
    if miss_cb:
        print(f"❌ 阶段开关缺失：{miss_cb}")
        return 1
    keys = [t.label for t in at.text_input]
    print(f"输入框：{len(keys)}（含源/回收/工作目录/模型/归档/精华）")

    # 抽查页的抽样控件（v3 阶段 4：审核台已重做为抽查页）。
    # ⚠️ 原来这里断言的是「跳到第 N 条 / 跳到分数 ≥X / 九宫格页码」——
    #    那些是**全量人工审核时代**的导航（ADR 0005），程序全驱动之后
    #    人工只抽 N 条看，用不上"跳到第 1500 条"。连同批量浏览整页一起删了。
    #
    # 两个入口用 radio 切换，**一次只渲染选中分支** —— 所以只查当前入口的控件，
    # 另一个入口要切过去再查（这正是选 radio 而不是嵌套 tabs 的理由之一）。
    num_keys = [n.key for n in at.number_input]
    btn_keys = [b.key for b in at.button]
    entry_radio = next((r for r in at.radio if "抽查什么" in r.label), None)
    if entry_radio is None:
        print("❌ 缺少「抽查什么」入口切换（L1 判定 / 已剪精华）")
        return 1
    which = entry_radio.value
    prefix = "sp_l1" if str(which).startswith("🎯") else "sp_ab"
    want = [f"{prefix}_n", f"{prefix}_seed"]
    miss = [k for k in want if k not in num_keys]
    if f"{prefix}_reroll" not in btn_keys:
        miss.append(f"{prefix}_reroll")
    if miss:
        print(f"❌ 抽查入口「{which}」的抽样控件缺失：{miss}")
        return 1
    print(f"抽查控件：入口「{which}」在位"
          f"（抽多少条 / 随机种子 / 换一批）")    # Track 0：三义尺子 —— 精华复核页必须能报出"看到过但被否决"这个数
    if "看到过但被否决" not in markdowns:
        print("❌ 精华复核页没有「无精华」拆三义的表（Track 0）")
        return 1
    print("Track 0/Track C 控件：C 级档案开关 + 无精华三义说明在位")

    # 判定参数区（v3 阶段 3）：口径切换与三档阈值必须在设置页可改。
    # 口径与阈值是**一对** —— 换公式不换阈值会让「建议清理」几乎不可达，
    # 让人在同一个页面里同时看到两者 + 校验，比自己去猜可靠。
    if "sc_mode" not in [s.key for s in at.selectbox]:
        print("❌ 设置页缺少「打分口径」下拉（legacy / quality_v2 切换）")
        return 1
    nums2 = {n.label: n for n in at.number_input}
    need = ["每片段抽帧数", "建议清理 ≤", "需人工确认 <", "批量放行 ≤"]
    miss = [k for k in need if not any(k == lb for lb in nums2)]
    if miss:
        print(f"❌ 判定参数控件缺失：{miss}")
        return 1
    if "💾 保存判定参数" not in labels:
        print("❌ 缺少「保存判定参数」按钮")
        return 1
    print(f"判定参数：口径下拉 + 抽帧数 {nums2['每片段抽帧数'].value} "
          f"+ trash≤{nums2['建议清理 ≤'].value} "
          f"+ review<{nums2['需人工确认 <'].value} "
          f"+ release≤{nums2['批量放行 ≤'].value}（机主拍板 trash 20 / review 35）")

    # 回归：回调里的未定义变量只有真点一次才暴露。
    # 改点「🎲 换一批」—— 旧点的是「跳到分数 ≥」，那个控件已随全量审核
    # 导航一起删除。选「换一批」的理由同上：它会走 session_state 写 +
    # st.rerun()，回调里的未定义变量在这一步才暴露。
    #
    # ⚠️ 绝不能点「该留 / 该删」—— 那两个会**写库**（搬文件、改判定）。
    # 冒烟必须只读。
    _btn = at.button(key=f"{prefix}_reroll")
    if _btn.disabled:
        print(f"⚠️ 入口「{which}」的「换一批」当前被禁用（抽样池为空），跳过点击回归")
    else:
        _btn.click().run()
        if at.exception:
            print("❌ 「换一批」回调抛异常：")
            for e in at.exception:
                print(f"   {e.value}")
            return 1
        print(f"抽查「换一批」（入口「{which}」）回调：无异常（且未写库）")

    print("✅ UI 渲染冒烟通过：整页无异常，一键执行区、七个阶段开关、"
          "抽查抽样控件、判定参数区在位")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
