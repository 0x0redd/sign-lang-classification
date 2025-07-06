#!/bin/bash

echo "Compiling LaTeX document..."
echo

# Check if pdflatex is available
if ! command -v pdflatex &> /dev/null; then
    echo "ERROR: pdflatex not found. Please install a LaTeX distribution."
    echo
    echo "For Ubuntu/Debian: sudo apt-get install texlive-full"
    echo "For macOS: Install MacTeX from https://www.tug.org/mactex/"
    echo "For other Linux: Install texlive package from your distribution"
    exit 1
fi

echo "Found pdflatex. Starting compilation..."
echo

# First compilation pass
echo "Running first pdflatex pass..."
if ! pdflatex -interaction=nonstopmode rapport.tex; then
    echo "ERROR: First compilation pass failed. Check rapport.log for details."
    exit 1
fi

echo
echo "Running second pdflatex pass for references..."
if ! pdflatex -interaction=nonstopmode rapport.tex; then
    echo "ERROR: Second compilation pass failed. Check rapport.log for details."
    exit 1
fi

echo
echo "SUCCESS: rapport.pdf has been generated!"
echo

# Clean up auxiliary files (optional)
read -p "Clean up auxiliary files? (y/n): " cleanup
if [[ $cleanup == "y" || $cleanup == "Y" ]]; then
    rm -f *.aux *.log *.toc *.lof *.lot *.lol *.out
    echo "Auxiliary files cleaned up."
fi

echo
echo "Compilation complete!"
