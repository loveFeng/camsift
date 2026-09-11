"""UI 渲染冒烟：用 Streamlit 官方 AppTest 把整页跑一遍，抓渲染期异常。

为什么需要它：源码级接线检查（selftest 里的字符串断言）只能证明"函数被调用了"，
证明不了"页面能渲染出来" —— 重复 widget key、未定义变量、StreamlitAPIException
这类问题只在真正跑一次渲染时才暴露。

只读：只点「跳到分数 ≥」这一个按钮（纯 COUNT 查询 + 队列定位，不写库），
其余按钮一概不点 —— 因此不会写库、不会落盘配置。

点它的理由：审核台的跳转回调曾把形参 `filt` 写成 `fil`，一按就 NameError，
而源码级断言查不出这类"只在回调里才执行"的错误 —— 必须真点一次。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from streamlit.testing.v1 import AppTest  # noqa: E402


def main() -> int:
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

    # 六个阶段开关 + 关键输入框是否都在
    cbs = [c.label for c in at.checkbox]
    need_cb = ["扫描元数据", "AI 分析", "音频复核", "语音档案归档", "批量放行", "画面精华剪辑"]
    miss_cb = [c for c in need_cb if c not in cbs]
    if miss_cb:
        print(f"❌ 阶段开关缺失：{miss_cb}")
        return 1
    keys = [t.label for t in at.text_input]
    print(f"输入框：{len(keys)}（含源/回收/工作目录/模型/归档/精华）")

    # 审核台 / 批量浏览的跳转与筛选控件（ADR 0002 配套改动）
    num_keys = [n.key for n in at.number_input]
    want_num = ["rv_jump_idx", "rv_jump_score", "rv_min_score",
                "grid_jump_page", "grid_min_score"]
    miss_num = [k for k in want_num if k not in num_keys]
    btn_keys = [b.key for b in at.button]
    want_btn = ["rv_jump_idx_go", "rv_jump_score_go", "rv_filter_clear",
                "grid_jump_go", "grid_first", "grid_last"]
    miss_btn = [k for k in want_btn if k not in btn_keys]
    cbs2 = [c.key for c in at.checkbox]
    miss_cb2 = [k for k in ("rv_person_only", "rv_speech_only", "grid_person_only")
                if k not in cbs2]
    if miss_num or miss_btn or miss_cb2:
        print(f"❌ 跳转/筛选控件缺失：数字框{miss_num} 按钮{miss_btn} 勾选{miss_cb2}")
        return 1
    print("跳转控件：审核台（条号/分数）+ 批量浏览（页码/首页/末页）在位")

    # Track C · C3：C 级档案的显隐开关（默认不占人工队列）
    if "rv_show_c_archive" not in cbs2:
        print("❌ 缺少「含仅 C 级档案」开关（Track C · C3）")
        return 1
    # Track 0：三义尺子 —— 精华复核页必须能报出"看到过但被否决"这个数
    if "看到过但被否决" not in markdowns:
        print("❌ 精华复核页没有「无精华」拆三义的表（Track 0）")
        return 1
    print("Track 0/Track C 控件：C 级档案开关 + 无精华三义说明在位")

    # 回归：回调里的未定义变量只有真点一次才暴露
    at.button(key="rv_jump_score_go").click().run()
    if at.exception:
        print("❌ 「跳到分数 ≥」回调抛异常：")
        for e in at.exception:
            print(f"   {e.value}")
        return 1
    print("审核台「跳到分数 ≥」回调：无异常")

    print("✅ UI 渲染冒烟通过：整页无异常，一键执行区、六个阶段开关、跳转筛选控件在位")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
