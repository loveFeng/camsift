"""
键盘快捷键 —— 审核效率的关键

Streamlit 没有原生快捷键 API，这里用注入到父页面的 JS 监听 keydown，
再按**按钮文本**找到对应按钮并触发点击。

为什么按文本找按钮：Streamlit 不允许给 button 加自定义 id/data 属性，
文本匹配是唯一稳定的锚点。因此本模块的键位表与 ui.py 的按钮文案是耦合的
—— 改按钮文案时记得同步这里。
"""
from __future__ import annotations

import json
from typing import List, Tuple

import streamlit as st
import streamlit.components.v1 as components

# (按键 code, 目标按钮文案片段, 说明)
DEFAULT_KEYS: List[Tuple[str, str, str]] = [
    ("Space", "确认清理", "确认清理（移入回收目录，可撤销）"),
    ("Enter", "误判，这段要保留", "标记误判并自动归因"),
    ("ArrowLeft", "上一条", "上一条"),
    ("ArrowRight", "下一条", "下一条"),
    ("KeyP", "用系统播放器打开", "用系统播放器打开原视频"),
    ("KeyV", "生成网页可播预览", "生成浏览器可播预览"),
]

_JS_TEMPLATE = """
<script>
(function () {{
  const doc = window.parent.document;
  if (doc.__vsHotkeysInstalled) {{ return; }}
  doc.__vsHotkeysInstalled = true;

  const MAP = {mapping};

  function clickByText(txt) {{
    const btns = doc.querySelectorAll('button');
    for (const b of btns) {{
      if (b.innerText && b.innerText.indexOf(txt) !== -1) {{
        b.click();
        return true;
      }}
    }}
    return false;
  }}

  doc.addEventListener('keydown', function (e) {{
    const t = e.target;
    if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable)) {{
      return;  // 正在输入备注，不抢键
    }}
    if (e.ctrlKey || e.altKey || e.metaKey || e.shiftKey) {{ return; }}
    const target = MAP[e.code];
    if (!target) {{ return; }}
    e.preventDefault();
    clickByText(target);
  }}, true);
}})();
</script>
"""

_LABELS = {
    "Space": "空格", "Enter": "回车", "ArrowLeft": "←", "ArrowRight": "→",
    "KeyP": "P", "KeyV": "V",
}


def install(keys: List[Tuple[str, str, str]] | None = None,
            enabled: bool = True) -> None:
    """注入快捷键监听。enabled=False 时不注入（页面按普通方式操作）。"""
    if not enabled:
        return
    keys = keys or DEFAULT_KEYS
    mapping = json.dumps({code: text for code, text, _ in keys}, ensure_ascii=False)
    html = _JS_TEMPLATE.format(mapping=mapping)
    try:
        components.html(html, height=0, width=0)
    except Exception:  # noqa: BLE001
        # 新版 Streamlit 用 st.iframe 取代 components.html（官方标注移除后走这里）。
        # 两个 API 都是把 HTML 渲染进同源 iframe，JS 访问 parent.document 的方式一致。
        try:
            st.iframe(html, height=0)
        except Exception:  # noqa: BLE001
            pass  # 快捷键是增强能力，注入失败不能拖垮页面


def hint_markdown(keys: List[Tuple[str, str, str]] | None = None,
                  enabled: bool = True) -> str:
    """生成快捷键提示文本。"""
    keys = keys or DEFAULT_KEYS
    if not enabled:
        return "`快捷键已关闭`"
    return "　".join(
        f"`{_LABELS.get(code, code)}` {desc}" for code, _, desc in keys
    )
