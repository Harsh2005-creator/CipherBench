@echo off
REM CipherBench Quick Start Script for Windows

echo ==========================================
echo CipherBench Project Setup
echo ==========================================
echo.

REM Check Python installation
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher
    pause
    exit /b 1
)

echo [1/5] Python detected:
python --version
echo.

REM Check MySQL installation
mysql --version >nul 2>&1
if errorlevel 1 (
    echo WARNING: MySQL CLI not found
    echo Please ensure MySQL is installed and running
    echo.
) else (
    echo [2/5] MySQL detected:
    mysql --version
    echo.
)

REM Install dependencies
echo [3/5] Installing Python dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)
echo Dependencies installed successfully!
echo.

REM Test database connection
echo [4/5] Testing database connection...
python -c "from database.db_config import test_connection; test_connection()"
if errorlevel 1 (
    echo.
    echo WARNING: Database connection failed
    echo Please configure config.yaml with correct MySQL credentials
    echo and create the database using: mysql -u root -p ^< database/schema.sql
    echo.
) else (
    echo Database connection successful!
    echo.
)

REM Test data loading
echo [5/5] Testing data loader...
python utils/data_loader.py
if errorlevel 1 (
    echo.
    echo WARNING: Data loading test failed
    echo Please ensure data files are in data/binary and data/multiclass directories
    echo.
)

echo.
echo ==========================================
echo Setup Complete!
echo ==========================================
echo.
echo Next steps:
echo 1. Configure MySQL password in config.yaml
echo 2. Create database: mysql -u root -p ^< database/schema.sql
echo 3. Train models: python train.py --model all --task multiclass --size 512KB
echo 4. Launch dashboard: streamlit run dashboard/streamlit_app.py
echo 5. Start API: python api/app.py
echo.
echo For more information, see README.md
echo.
pause
