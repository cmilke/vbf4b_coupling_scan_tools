base = $(shell basename $$PWD)
target = $(base).pdf
sources = *.tex
compile = lualatex -jobname=$(base) -output-directory='out' -halt-on-error -pdf

$(target): $(sources) out
	$(compile) main.tex 

out:
	mkdir out

sync:
	git add images/remote

desync:
	rm -r images/remote

hard_desync:
	git rm -rf images/remote

clean:
	rm out/* &> /dev/null
