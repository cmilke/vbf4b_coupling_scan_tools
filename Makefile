base = $(shell basename $$PWD)
target = $(base).pdf
sources = *.tex
compile = lualatex -jobname=$(base) -output-directory='pdfout' -halt-on-error -pdf


main: main.tex pdfout
	$(compile) $<

clean:
	rm pdfout/* &> /dev/null
