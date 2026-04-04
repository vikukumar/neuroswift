@echo off
setlocal

where py >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    py -m venv venv
) else (
    python -m venv venv
)

call venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --default-timeout=1000 --retries=10

endlocal
