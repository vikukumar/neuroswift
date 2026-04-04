@echo off
setlocal

call setup_env.bat
if errorlevel 1 goto :error

call venv\Scripts\activate.bat
if errorlevel 1 goto :error

python train_small_llm.py --data-path examples\data\neuroswift_corpus.txt --output-dir artifacts\neuroswift-tiny
if errorlevel 1 goto :error

python test_llm.py --model-dir artifacts\neuroswift-tiny --prompt "neuroswift "
if errorlevel 1 goto :error

endlocal
exit /b 0

:error
echo.
echo Workflow failed. See the error output above.
endlocal
exit /b 1
