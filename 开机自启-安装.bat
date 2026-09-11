@echo off
setlocal
title 监控视频 AI 筛选工作台 - 安装开机自启
cd /d "%~dp0"
set "PATH=%SystemRoot%\System32;%SystemRoot%;%SystemRoot%\System32\Wbem;%SystemRoot%\System32\WindowsPowerShell\v1.0;%PATH%"

set "TASKNAME=监控视频AI筛选工作台"
set "LAUNCHER=%~dp0启动工作台.bat"

echo ============================================================
echo    安装开机自启（登录后自动启动工作台）
echo ============================================================
echo.
echo  任务名称：%TASKNAME%
echo  目标脚本：%LAUNCHER%
echo.
echo  说明：
echo   · 使用 Windows 计划任务，登录后延迟 30 秒启动
echo   · 启动后会弹出一个控制台窗口，关掉窗口 = 停止服务
echo   · 想取消：运行「开机自启-卸载.bat」
echo.
pause

schtasks /Query /TN "%TASKNAME%" >nul 2>&1
if not errorlevel 1 (
    echo [提示] 已存在同名任务，先删除旧的...
    schtasks /Delete /TN "%TASKNAME%" /F >nul 2>&1
)

schtasks /Create /TN "%TASKNAME%" /TR "\"%LAUNCHER%\"" /SC ONLOGON /RL LIMITED /DELAY 0000:30 /F
if errorlevel 1 (
    echo.
    echo [错误] 创建失败。可尝试：右键本文件 -^> 以管理员身份运行。
    echo.
    pause
    exit /b 1
)

echo.
echo [完成] 开机自启已安装。下次登录系统时会自动启动工作台。
echo.
echo 立即试跑一次（可不重启验证）：schtasks /Run /TN "%TASKNAME%"
echo.
pause
endlocal
