@echo off
title Solitaire Update
setlocal enabledelayedexpansion

echo.
echo   Solitaire Update
echo   ================
echo.

:: Find where this script lives (the update package)
set "UPDATEDIR=%~dp0"

:: -------------------------------------------------------------------
:: Python detection cascade: python -> python3 -> py launcher
:: -------------------------------------------------------------------
set "PYTHON="
where python >nul 2>&1 && set "PYTHON=python"
if not defined PYTHON (
    where python3 >nul 2>&1 && set "PYTHON=python3"
)
if not defined PYTHON (
    where py >nul 2>&1 && set "PYTHON=py"
)

:: -------------------------------------------------------------------
:: Try Python updater first (full safety gates)
:: -------------------------------------------------------------------
if defined PYTHON (
    echo   Using Python: %PYTHON%
    echo.

    :: Check if update.py exists in the package
    if exist "%UPDATEDIR%update.py" (
        %PYTHON% "%UPDATEDIR%update.py" "%UPDATEDIR%"
        if !ERRORLEVEL! EQU 0 (
            goto :end
        ) else (
            echo.
            echo   [!!] Python updater exited with error code !ERRORLEVEL!
            echo   [!!] Falling back to PowerShell updater...
            echo.
        )
    ) else (
        echo   [!!] update.py not found in package, using PowerShell fallback...
        echo.
    )
)

:: -------------------------------------------------------------------
:: PowerShell fallback (no Python required)
:: -------------------------------------------------------------------
if exist "%UPDATEDIR%update.ps1" (
    echo   Using PowerShell fallback...
    echo.
    powershell -ExecutionPolicy Bypass -NoProfile -File "%UPDATEDIR%update.ps1" "%UPDATEDIR%"
    if !ERRORLEVEL! EQU 0 (
        goto :end
    ) else (
        echo.
        echo   [!!] PowerShell updater also failed.
        echo   [!!] Please contact support for help.
        echo.
        goto :end
    )
)

:: -------------------------------------------------------------------
:: Neither Python nor PowerShell script available
:: -------------------------------------------------------------------
echo.
echo   [!!] Could not find Python or PowerShell update scripts.
echo   [!!] Ensure update.py and/or update.ps1 are in the same folder as this script.
echo.

:end
pause
