@echo off
echo ========================================================
echo   PNEUMONIA DETECTION PIPELINE RUNNER
echo ========================================================
echo.

echo [1/4] Running Exploratory Data Analysis (EDA)...
python notebooks/01_eda.py
if %errorlevel% neq 0 (
    echo [ERROR] EDA failed!
    pause
    exit /b %errorlevel%
)
echo.

echo [2/4] Training Models (This will take 1-3 hours depending on your RTX 3050)...
python run_training.py
if %errorlevel% neq 0 (
    echo [ERROR] Training failed!
    pause
    exit /b %errorlevel%
)
echo.

echo [3/4] Evaluating Models and Generating Plots...
python run_evaluation.py
if %errorlevel% neq 0 (
    echo [ERROR] Evaluation failed!
    pause
    exit /b %errorlevel%
)
echo.

echo [4/4] Starting Streamlit App...
echo (Press Ctrl+C to stop the app later)
streamlit run app/streamlit_app.py
