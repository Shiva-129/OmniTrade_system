@echo off
set PYTHONPATH=%PYTHONPATH%;.
echo Starting OmniTrade Observer...
python src/observer.py
pause
