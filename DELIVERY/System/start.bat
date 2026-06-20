@echo off
echo ========================================
echo TBM盾构机关键掘进参数实时预测系统启动脚本
echo ========================================

echo 正在检查Python环境...
python --version
if %errorlevel% neq 0 (
    echo 错误: 未找到Python环境
    pause
    exit /b 1
)

echo.
echo 正在安装依赖包...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo 警告: 依赖包安装可能有问题，但继续启动...
)

echo.
echo 正在启动Web服务器...
echo 系统将在 http://localhost:5000 启动
echo 按 Ctrl+C 停止服务器
echo.

python app.py

pause
