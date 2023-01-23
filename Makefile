# compile a report by its name

%:
	pdflatex $@.tex

clean:
	rm *.aux *.*latex* *.fls *.log *.xdv