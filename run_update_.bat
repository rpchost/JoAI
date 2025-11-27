@echo off
REM run_update.bat — Windows Task Scheduler Runner

REM Change to project directory
cd /d C:\DATA\App\Rpchost\JoAI

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Run the update script
python update_data_local.py >> logs\update.log 2>&1

REM Deactivate
deactivate

exit