@echo off
echo Starting Forecasting Dashboard...
echo.

cd /d "%~dp0"
streamlit run app/dashboard.py

pause 