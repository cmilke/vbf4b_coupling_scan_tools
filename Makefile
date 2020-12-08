base = $(shell basename $$PWD)
target = $(base).pdf
sources = *.tex
compile = lualatex -jobname=$(base) -output-directory='out' -halt-on-error -pdf


main: main.tex out
	$(compile) $<

clean:
	rm out/* &> /dev/null
