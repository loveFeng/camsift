# Surveillance Video AI Triage Workbench

[中文](README.md) | **English**

Picks out **false-positive motion-detection clips** from surveillance footage and moves them to a recycle directory, graded by confidence. **The program only analyzes, scores, grades, and moves — it never deletes.** The final deletion is always done manually by you.

## Key Capabilities

| Requirement | Where it lives |
|------|---------|
| Review videos on a web page | "Review Workbench" one-by-one (with hotkeys) + "Batch Browse" grid (paginated) |
| Watch job execution | "Execution Progress" page (live panels for background tasks + batch progress + throughput/ETA) |
| Why delete / why keep | Every item shows its **decision-evidence chain** (itemized scoring breakdown) |
| Misjudgment attribution | Flag "misjudged — keep" and the system attributes the cause automatically, keeps the result on screen, and aggregates it on the "Misjudgment Analysis" page |
| 🔊 Audio re-review | For delete candidates: extract audio track → voice detection (FSMN-VAD) → full transcript shown to you → LLM judges usefulness (Alibaba FunASR + ollama/qwen3). Prevents deleting clips where something happens off-camera |
| 🎞 Highlight clipping | Before deletion, cuts out the few seconds where "something is actually happening on screen" (L2.8): two-level detection (coarse 3fps@640 + fine 4fps@1280), A/B/C grade outputs. Together with the speech archive this forms the last line of defense before deletion |
| ⚡ One-click pipeline | Chains "scan → analyze → audio re-review → speech archiving → bulk release → highlight clipping" into a single click, ending with the recycle directory ready (ADR 0004) |

**Scanning and analysis run in background threads**: after starting a job you can switch tabs and keep reviewing; progress refreshes every second, and jobs can be stopped at any time (finishing the current file first; already-written results are never lost).

> 📚 **Want to know how the code works?** Start at **[docs/架构设计.md](docs/架构设计.md)** (architecture doc, in Chinese: layering / data flow / lifecycle / design trade-offs);
> schema and queue predicates in [docs/数据模型.md](docs/数据模型.md); **what to do next** in [docs/待办与风险.md](docs/待办与风险.md).
> Full documentation index: [docs/README.md](docs/README.md).

## Decision Logic (one sentence)

The footage moved — but *what* moved? **No person/vehicle detected, only pixel changes** → classified as environmental false trigger (leaves / light & shadow / headlights), recommended for cleanup. Person/vehicle/animal detected → keep. **Candidates with no person on screen but with an audio track pass through audio re-review before cleanup** (audio events get rescued; see `docs/设计档案-10TB筛选系统.md` §3.6).

Three grades:

- **Recommend cleanup (trash)**: no target and weak visual change
- **Needs human confirmation (review)**: visible motion but no target, or a suspected target below threshold
- **Recommend keep**: person/vehicle/animal detected

## Quick Start

**Double-click `启动工作台.bat`** in the project root. (Script and UI filenames are in Chinese; the app UI itself is Chinese too.)

It automatically: checks the virtualenv → checks dependencies (installs if missing) → checks the YOLO model → checks port availability → starts the service → opens the browser.

Then open **http://localhost:8501**

Stop the service: press `Ctrl+C` in the console window, or double-click `停止服务.bat`.

> Full from-zero setup, complete usage walkthrough, and troubleshooting: **[docs/启动与使用指南.md](docs/启动与使用指南.md)** (in Chinese)

> First run auto-creates two local files, neither of which is committed to git: `.venv/` (virtualenv) and
> `config.json` (your local paths and thresholds. What *is* committed is the `DEFAULT_CONFIG` defaults in
> `src/config.py`; if `config.json` is missing it gets rebuilt from those defaults).

Command-line start (equivalent):

```bash
cd <project dir>   # wherever you cloned this repo
.venv/Scripts/python.exe -m streamlit run src/ui.py --server.port=8501
```

## Workflow

1. **"Settings & Scan" page** — fill in three paths:
   - `source_root`: video source root (external drive, read-only)
   - `trash_root`: graded recycle directory (**must be on the same physical disk as the source**, otherwise moves are refused)
   - `work_dir`: local working directory (database/thumbnails, put it on SSD)
2. Click **🔍 Start Scan (background)**, then **🤖 Start AI Analysis (background)**. Jobs run in the background; the panel below shows progress, speed, and ETA in real time. Click "Stop" anytime.
3. **"Review Workbench"** — review one by one: thumbnails + evidence chain → confirm cleanup / flag as misjudged-keep
   - Hotkeys: `Space` confirm cleanup　`Enter` misjudged-keep　`←` `→` navigate　`P` open in player　`V` generate web preview
   - After flagging a misjudgment, the attribution result stays visible at the top of the page (it doesn't flash by)
4. **"Batch Browse"** — paginated multi-select; after batch-moving to the recycle directory it **auto-generates a manifest CSV and space statistics**
5. **"Highlight Review"** page — browse A/B products (C-grade is archival, safe to skip); the review workbench top bar has a **one-click release for "no-person" candidates**, backstopped by a highlight re-scan
6. **"Misjudgment Analysis"** page — see the distribution of misjudgment causes and tuning suggestions
7. Want it all in one go: the **"⚡ One-Click Pipeline"** at the top of "Settings & Scan" runs the full chain (including audio re-review + speech archiving + release + highlight clipping); when it's done, the only thing left is you manually emptying the recycle directory

> Want it to run automatically on boot? Double-click `开机自启-安装.bat` (starts 30 s after login). While jobs run, the program
> automatically requests "prevent system sleep" — details in [docs/启动与使用指南.md](docs/启动与使用指南.md) §5.2 (in Chinese).

> After changing thresholds, click "Save Config" **before** starting jobs — otherwise the old config is used.

## Repository Layout

```
src/
  config.py     config (thresholds / paths / scoring weights)
  db.py         SQLite data layer (incl. raw detections and decision-evidence chain)
  scanner.py    ffprobe metadata scan (serial disk reads, resumable)
  analyzer.py   frame extraction + motion detection + YOLO object detection + scoring
  audioreview.py  audio re-review: VAD voice detection + ASR transcript + LLM usefulness (L2.5)
  archiver.py   speech archive: rescue audio + transcripts before deletion (L2.7)
  highlighter.py  highlight clipping: coarse + fine scan, A/B/C grades (L2.8)
  feedback.py   misjudgment attribution (re-examines raw detections, classifies causes, suggests tuning)
  mover.py      graded move + undo + manifest/space stats (contains no deletion of any kind)
  jobs.py       background job manager (dedicated threads + progress + stop, process-level state)
  hotkeys.py    keyboard hotkeys (injects JS that locates buttons by label text)
  cli.py        command-line entry (for running headless without the UI)
  ui.py         Streamlit review workbench
scripts/
  selftest.py        core-logic self-tests (regression; hundreds of assertions, incl. a code audit that src/ has no delete calls)
  smoke_ui.py        UI render smoke test (Streamlit AppTest runs the full page)
  smoke_audio.py     audio engine smoke test (real Chinese speech vs. pure noise)
  download_model.py  download YOLO ONNX model (--spec picks yolov8/yolo11 n/s/m; two verification gates per input size)
  download_audio_models.py  download FunASR models (fsmn-vad / paraformer-zh / ct-punc, ~1.2 GB)
  setup_audio_engine.py     install L2.5 audio engine deps (works around funasr's numpy<2 pin)
  compare_models.py  model comparison experiment: does switching model / input size help, on the same hand-labeled batch
  align_subdirs.py  existing-data alignment: file existing recycle-dir originals into "clipped" / "pending-salvage" subdirs (dry-run by default, --apply to move)
  audit_requirements.py  read-only audit: the four requirements vs. live DB numbers (reproducibility entry point)
  audit_highlight.py  product audit: lists rows that "were detected but have zero products / still-frame only" (bucketed by symptom × version, flags suspicious "already at target version yet empty" rows); --recut N samples a re-run to estimate rescue rate (read-only on the DB)
  compile_period.py  periodic compilations: stitch highlight clips into weekly/monthly reels (prototype, see docs/周期合辑方案.md)
  _legacy/           one-off diagnostic / backfill scripts, retired, unmaintained
docs/                documentation index: docs/README.md; implementation docs start at docs/架构设计.md;
                     the single todo/risk register: docs/待办与风险.md
```

## Command Line (headless analysis)

First create the virtualenv and install dependencies (the `.bat` scripts do this automatically; from the CLI you do it manually):

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

```bash
.venv/Scripts/python.exe -m src.cli scan    --root D:/CCTV/2024 --limit 500
.venv/Scripts/python.exe -m src.cli analyze --limit 500
.venv/Scripts/python.exe -m src.cli stats
.venv/Scripts/python.exe -m src.cli feedback
.venv/Scripts/python.exe -m src.cli export  --out cleanup_list.csv
```

## Key Design Decisions

- **Raw detections are saved** (including below-threshold ones): otherwise "why was this misjudged" is unanswerable after the fact.
- **Undecodable clips default to keep**: proprietary codecs (H.265+ / Smart H.265) are never trashed for technical failure.
- **Source drive stays read-only the whole time**: database and thumbnails all live on local SSD; the source disk is written exactly once, when a move executes.
- **Graded recycle directory**: `<trash root>/A_高置信_可批量删除/<year>/…` and `B_低置信_建议抽查/<year>/…` (original directory structure preserved to avoid name-collision overwrites) — the folder names tell you what's safe to delete in bulk.
- **The program never deletes**: `scripts/selftest.py` includes a code audit asserting there are no `os.remove` / `shutil.rmtree` / `unlink` calls under `src/`.
