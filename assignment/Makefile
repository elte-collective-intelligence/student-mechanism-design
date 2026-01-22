# Makefile for compiling assignment.tex

.PHONY: all clean view docker docker-compose help

# Default target
all: assignment.pdf

# Compile the PDF (requires pdflatex installed)
assignment.pdf: assignment.tex
	pdflatex -interaction=nonstopmode assignment.tex
	pdflatex -interaction=nonstopmode assignment.tex

# Compile using Docker (no LaTeX installation needed)
docker:
	./compile-docker.sh

# Compile using docker-compose (no LaTeX installation needed)
docker-compose:
	docker-compose up --build

# Clean auxiliary files
clean:
	rm -f *.aux *.log *.out *.toc *.synctex.gz

# Clean everything including the PDF
distclean: clean
	rm -f assignment.pdf

# View the PDF (Linux)
view: assignment.pdf
	xdg-open assignment.pdf 2>/dev/null || evince assignment.pdf 2>/dev/null || okular assignment.pdf

# Quick compile (single pass)
quick:
	pdflatex -interaction=nonstopmode assignment.tex

# Help message
help:
	@echo "Available targets:"
	@echo "  make            - Compile PDF (requires pdflatex)"
	@echo "  make docker     - Compile using Docker (recommended if no LaTeX)"
	@echo "  make docker-compose - Compile using docker-compose"
	@echo "  make quick      - Single-pass compile (faster)"
	@echo "  make clean      - Remove auxiliary files"
	@echo "  make distclean  - Remove all generated files"
	@echo "  make view       - Open PDF in viewer"
	@echo "  make help       - Show this help message"
