@echo off
setlocal EnableDelayedExpansion
title 停止监控视频筛选工作台
cd /d "%~dp0"
rem 把系统目录提到 PATH 最前面：本机装有 Git/Anaconda，
rem 它们的 timeout.exe / find / sort 会顶替系统同名命令导致脚本行为异常。
set "PATH=%SystemRoot%\System32;%SystemRoot%;%SystemRoot%\System32\Wbem;%SystemRoot%\System32\WindowsPowerShell\v1.0;%PATH%"
echo 正在停止工作台...
echo.
set "FOUND=0"
for /f "tokens=5" %%P in ('netstat -ano ^| findstr ":8501" ^| findstr "LISTENING"') do (
    echo   关闭服务进程 PID %%P
    taskkill /PID %%P /F >nul 2>&1
    set "FOUND=1"
)
if "!FOUND!"=="0" (
    echo   未检测到运行中的工作台（端口 8501 空闲）
) else (
    echo.
    echo   已停止。数据已保存在 data\videos.db，下次启动会接着上次的进度。
)
echo.
ping -n 4 127.0.0.1 >nul
endlocal
