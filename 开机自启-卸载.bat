@echo off
setlocal
title 监控视频 AI 筛选工作台 - 取消开机自启
cd /d "%~dp0"
set "PATH=%SystemRoot%\System32;%SystemRoot%;%SystemRoot%\System32\Wbem;%SystemRoot%\System32\WindowsPowerShell\v1.0;%PATH%"

set "TASKNAME=监控视频AI筛选工作台"

echo ============================================================
echo    取消开机自启
echo ============================================================
echo.

schtasks /Query /TN "%TASKNAME%" >nul 2>&1
if errorlevel 1 (
    echo [提示] 没有找到自启任务（可能本来就未安装），无需处理。
    echo.
    pause
    exit /b 0
)

schtasks /Delete /TN "%TASKNAME%" /F
if errorlevel 1 (
    echo.
    echo [错误] 删除失败。可尝试：右键本文件 -^> 以管理员身份运行。
    echo.
    pause
    exit /b 1
)

echo.
echo [完成] 开机自启已取消。工作台仍可手动用「启动工作台.bat」启动。
echo.
pause
endlocal
