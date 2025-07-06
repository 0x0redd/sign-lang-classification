@echo off
echo Compiling LaTeX document...
echo.

REM Check if pdflatex is available
where pdflatex >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: pdflatex not found. Please install a LaTeX distribution like MiKTeX or TeX Live.
    echo.
    echo For Windows, you can download MiKTeX from: https://miktex.org/download
    pause
    exit /b 1
)

echo Found pdflatex. Starting compilation...
echo.

REM First compilation pass
echo Running first pdflatex pass...
pdflatex -interaction=nonstopmode rapport.tex
if %errorlevel% neq 0 (
    echo ERROR: First compilation pass failed. Check rapport.log for details.
    pause
    exit /b 1
)

echo.
echo Running second pdflatex pass for references...
pdflatex -interaction=nonstopmode rapport.tex
if %errorlevel% neq 0 (
    echo ERROR: Second compilation pass failed. Check rapport.log for details.
    pause
    exit /b 1
)

echo.
echo SUCCESS: rapport.pdf has been generated!
echo.

REM Clean up auxiliary files (optional)
set /p cleanup="Clean up auxiliary files? (y/n): "
if /i "%cleanup%"=="y" (
    del *.aux *.log *.toc *.lof *.lot *.lol *.out 2>nul
    echo Auxiliary files cleaned up.
)

echo.
echo Compilation complete!
pause
