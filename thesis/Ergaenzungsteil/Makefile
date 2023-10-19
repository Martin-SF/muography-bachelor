all: full

TeXOptions = -lualatex \
			 -interaction=nonstopmode \
			 -halt-on-error \
			 -synctex=1 \
			 -output-directory=build

full: FORCE | build
	latexmk $(TeXOptions) thesis.tex
	cp build/thesis.pdf ../2022_Bachelor_Ergänzung_Schoenfeld_Martin.pdf

fast: FORCE | build
	# rm -f thesis.pdf
	lualatex --output-directory=build --interaction=nonstopmode --synctex=1 thesis.tex

build:
	mkdir -p build/

clean:
	rm -rf build

FORCE:
	
.PHONY: all full FORCE build clean fast fast2noethe thesis