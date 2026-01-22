#!/bin/bash

# Script to compile LaTeX document using Docker
# Usage: ./compile-docker.sh

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "================================================"
echo "LaTeX Docker Compilation Script"
echo "================================================"
echo ""

if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed."
    echo "Please install Docker from: https://docs.docker.com/get-docker/"
    exit 1
fi
if [ ! -f "$SCRIPT_DIR/assignment.tex" ]; then
    echo "Error: assignment.tex not found in $SCRIPT_DIR"
    exit 1
fi

echo "Building Docker image (this may take a few minutes first time)..."
docker build -f "$SCRIPT_DIR/Dockerfile.latex" -t latex-compiler "$SCRIPT_DIR"

echo ""
echo "Compiling LaTeX document (pass 1/2)..."
docker run --rm -v "$SCRIPT_DIR:/work" latex-compiler pdflatex -interaction=nonstopmode assignment.tex

echo ""
echo "Compiling LaTeX document (pass 2/2)..."
docker run --rm -v "$SCRIPT_DIR:/work" latex-compiler pdflatex -interaction=nonstopmode assignment.tex

echo ""
echo "================================================"
echo "Compilation complete!"
echo "Output: $SCRIPT_DIR/assignment.pdf"
echo "================================================"
if [ -f "$SCRIPT_DIR/assignment.pdf" ]; then
    echo ""
    echo "PDF generated successfully!"
    
    if command -v xdg-open &> /dev/null; then
        echo "Opening PDF..."
        xdg-open "$SCRIPT_DIR/assignment.pdf" 2>/dev/null || true
    fi
else
    echo ""
    echo "Warning: PDF file was not generated. Check the output above for errors."
    exit 1
fi
