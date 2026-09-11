@echo off
setlocal EnableDelayedExpansion
title 监控视频 AI 筛选工作台 - 启动器
cd /d "%~dp0"
rem 把系统目录提到 PATH 最前面：本机装有 Git/Anaconda，
rem 它们的 timeout.exe / find / sort 会顶替系统同名命令导致脚本行为异常。
set "PATH=%SystemRoot%\System32;%SystemRoot%;%SystemRoot%\System32\Wbem;%SystemRoot%\System32\WindowsPowerShell\v1.0;%PATH%"
echo ============================================================
echo    监控视频 AI 筛选工作台  -  启动器
echo ============================================================
echo.
rem ============================================================
rem  [1/5] 虚拟环境
rem ============================================================
set "PY=.venv\Scripts\python.exe"
if exist "%PY%" goto :deps
echo [1/5] 首次运行：正在创建虚拟环境...
set "BASEPY="
if exist "C:\Users\Administrator\.workbuddy\binaries\python\versions\3.13.12\python.exe" (
    set "BASEPY=C:\Users\Administrator\.workbuddy\binaries\python\versions\3.13.12\python.exe"
) else (
    where python >nul 2>&1 && set "BASEPY=python"
)
if not defined BASEPY (
    echo     [错误] 找不到 Python，请先安装 Python 3.10 以上版本并勾选 Add to PATH
    echo.
    pause
    exit /b 1
)
"%BASEPY%" -m venv .venv
if errorlevel 1 (
    echo     [错误] 虚拟环境创建失败
    pause
    exit /b 1
)
echo     虚拟环境已创建
rem ============================================================
rem  [2/5] 依赖
rem ============================================================
:deps
echo [1/5] 虚拟环境已就绪
echo [2/5] 检查依赖...
"%PY%" -c "import streamlit, cv2, numpy" >nul 2>&1
if errorlevel 1 (
    echo     首次安装依赖（约 1-2 分钟，使用清华镜像源）...
    "%PY%" -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
    if errorlevel 1 (
        echo.
        echo     [错误] 依赖安装失败。可尝试手动执行：
        echo         .venv\Scripts\python.exe -m pip install -r requirements.txt
        echo.
        pause
        exit /b 1
    )
)
echo     依赖就绪
rem ============================================================
rem  [3/5] YOLO 模型
rem ============================================================
echo [3/5] 检查 YOLO 模型...
if exist "data\models\yolov8n.onnx" (
    echo     模型已就绪
    goto :port
)
echo     未找到模型（约 12MB）。
echo     没有模型也能运行，但只会做画面动静检测 —— 认不出人/车，判定准确度大打折扣。
choice /c YN /t 20 /d Y /m "     现在下载模型？(Y/N)"
if errorlevel 2 (
    echo     跳过下载，稍后可在网页「设置与扫描」页看到提示，或运行 download_model.bat
    goto :port
)
echo     正在下载（如被网络中断，脚本会自动断点续传重试）...
"%PY%" scripts\download_model.py
echo.
rem ============================================================
rem  [4/5] 端口占用
rem ============================================================
:port
echo [4/5] 检查端口 8501...
set "OLDPID="
for /f "tokens=5" %%P in ('netstat -ano ^| findstr ":8501" ^| findstr "LISTENING"') do set "OLDPID=%%P"
if not defined OLDPID goto :start
echo     端口已被占用（PID !OLDPID!）—— 通常是上次没关掉的工作台。
choice /c YN /t 15 /d Y /m "     关闭旧服务并重新启动？(Y/N)"
if errorlevel 2 (
    echo     保持旧服务运行，改为直接打开浏览器...
    start "" http://localhost:8501
    ping -n 3 127.0.0.1 >nul
    exit /b 0
)
taskkill /PID !OLDPID! /F >nul 2>&1
ping -n 3 127.0.0.1 >nul
echo     旧服务已关闭
rem ============================================================
rem  [5/5] 启动服务
rem ============================================================
:start
echo [5/5] 启动服务...
echo.
echo ============================================================
echo   工作台即将启动，浏览器会自动打开：
echo.
echo        http://localhost:8501
echo.
echo   · 这个窗口要一直开着 —— 关掉它就等于停止服务
echo   · 停止服务：在本窗口按 Ctrl+C，或运行 停止服务.bat
echo   · 首次使用请先到「设置与扫描」页填写视频源目录与回收目录
echo ============================================================
echo.
rem 前台运行：服务日志直接打在本窗口，出问题能一眼看到。
rem 不加 --server.headless，让 Streamlit 自己打开浏览器（比脚本延时再 start 更可靠）。
"%PY%" -m streamlit run src/ui.py --server.port=8501 --server.address=localhost --browser.gatherUsageStats=false
echo.
echo 服务已停止。
pause
endlocal
