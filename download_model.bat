@echo off
setlocal EnableDelayedExpansion
title 下载 YOLO 检测模型
cd /d "%~dp0"
rem 把系统目录提到 PATH 最前面：本机装有 Git/Anaconda，
rem 它们的 timeout.exe / find / sort 会顶替系统同名命令导致脚本行为异常。
set "PATH=%SystemRoot%\System32;%SystemRoot%;%SystemRoot%\System32\Wbem;%SystemRoot%\System32\WindowsPowerShell\v1.0;%PATH%"
echo ============================================================
echo   下载 YOLOv8n ONNX 检测模型（约 12MB）
echo ============================================================
echo.
if not exist ".venv\Scripts\python.exe" (
    echo [错误] 未找到虚拟环境，请先运行 启动工作台.bat 完成初始化
    echo.
    pause
    exit /b 1
)
if exist "data\models\yolov8n.onnx" (
    echo 模型已存在，无需重复下载。
    choice /c YN /t 10 /d N /m "是否删除后重新下载？(Y/N)"
    if errorlevel 2 (
        exit /b 0
    )
    del /f /q "data\models\yolov8n.onnx" >nul 2>&1
)
echo 下载中（网络不稳定时会自动断点续传重试）...
echo.
.venv\Scripts\python.exe scripts\download_model.py
echo.
if exist "data\models\yolov8n.onnx" (
    echo 模型已就绪：data\models\yolov8n.onnx
) else (
    echo [提示] 自动下载失败。可手动下载后放到此位置：
    echo     data\models\yolov8n.onnx
    echo 下载地址：
    echo     https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.onnx
)
echo.
pause
endlocal
