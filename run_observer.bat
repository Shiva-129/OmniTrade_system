@echo off
set PYTHONPATH=%PYTHONPATH%;.
echo Starting OmniTrade Observer...
python -m src.observer
pause
