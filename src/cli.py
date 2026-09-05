"""
命令行入口 —— 分析引擎与 UI 解耦，可在后台长期运行

常用命令
--------
    python -m src.cli init                      生成默认 config.json
    python -m src.cli scan   --root D:/2019 --limit 200
    python -m src.cli analyze --limit 200
    python -m src.cli stats
    python -m src.cli feedback                  查看误判归因与调参建议
    python -m src.cli export --out 清理清单.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import ensure_dirs, load_config, save_config
from .db import Database


def _db(cfg):
    return Database(cfg["runtime"]["sqlite_path"])


def cmd_init(args):
    ensure_dirs()
    cfg = load_config()
    if not Path("config.json").exists():
        save_config(cfg)
        print("已生成 config.json，请先编辑 source_root / trash_root")
    else:
        print("config.json 已存在")
    return 0


def cmd_scan(args):
    from .scanner import Scanner
    cfg = load_config()
    ensure_dirs(cfg)
    db = _db(cfg)
    sc = Scanner(db, cfg)
    res = sc.scan(args.root, limit=args.limit, with_audio_energy=args.audio,
                  log=lambda m: print(" ", m))
    print(f"扫描完成：{res}")
    return 0


def cmd_analyze(args):
    from .analyzer import Analyzer
    cfg = load_config()
    ensure_dirs(cfg)
    db = _db(cfg)
    an = Analyzer(db, cfg)
    if not an.yolo_on:
        print("提示：未启用 YOLO（缺少模型文件），判定仅依赖画面动静。")
        print("      运行 `python scripts/download_model.py` 可下载模型。")
    res = an.run(limit=args.limit, log=lambda m: print(" ", m))
    print(f"分析完成：{res}")
    return 0


def cmd_stats(args):
    cfg = load_config()
    db = _db(cfg)
    s = db.stats()
    for k, v in s.items():
        print(f"  {k:16s} {v}")
    return 0


def cmd_feedback(args):
    cfg = load_config()
    db = _db(cfg)
    from . import feedback as fb
    s = fb.summarize_feedback(db)
    if not s["total"]:
        print("还没有误判记录。")
        return 0
    print(f"累计误判 {s['total']} 条\n")
    print("原因分布：")
    for d in s["distribution"]:
        print(f"  {d['label']:22s} {d['count']:4d}  ({d['ratio']}%)")
    print("\n优化建议：")
    for x in s["suggestions"]:
        print(f"  · {x}")
    return 0


def cmd_export(args):
    cfg = load_config()
    db = _db(cfg)
    p = db.export_csv(args.out)
    print(f"已导出：{p}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="video-search",
                                 description="监控视频 AI 筛选（只分析、分级与移动，永不删除）")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init", help="生成默认配置").set_defaults(func=cmd_init)

    p = sub.add_parser("scan", help="扫描目录并提取元数据")
    p.add_argument("--root", required=True, help="视频源根目录")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--audio", action="store_true", help="同时分析音频能量（较慢）")
    p.set_defaults(func=cmd_scan)

    p = sub.add_parser("analyze", help="运行 AI 分析")
    p.add_argument("--limit", type=int, default=None)
    p.set_defaults(func=cmd_analyze)

    sub.add_parser("stats", help="查看统计").set_defaults(func=cmd_stats)
    sub.add_parser("feedback", help="误判归因与调参建议").set_defaults(func=cmd_feedback)

    p = sub.add_parser("export", help="导出清理清单")
    p.add_argument("--out", default="清理清单.csv")
    p.set_defaults(func=cmd_export)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
