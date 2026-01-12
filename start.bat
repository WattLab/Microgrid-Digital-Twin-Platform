@echo off
chcp 65001 > nul
cd /d "%~dp0"

echo ==================================================
echo    微电网数字孪生智能调度平台
echo    Microgrid Digital Twin Smart Scheduling
echo ==================================================
echo.
echo 正在启动服务器...
echo.

REM 关闭已有的Python进程
taskkill /F /IM python.exe 2>nul

REM 等待进程结束
timeout /t 2 /nobreak > nul

REM 启动服务器
echo 启动FastAPI服务器...
start "MicrogridServer" cmd /c "cd /d "%~dp0" && python main.py"

REM 等待服务器启动
timeout /t 4 /nobreak > nul

REM 打开浏览器
echo 正在打开可视化界面...
start "" "http://localhost:8000/visualization"

echo.
echo ==================================================
echo 服务已启动！
echo 可视化界面: http://localhost:8000/visualization
echo API文档: http://localhost:8000/docs
echo ==================================================
pause
