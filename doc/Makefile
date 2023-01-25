# compile a report by its name

%: 
	mkdir -p img/$@
	python3 $@.py
	pdflatex $@.tex

clean:
	rm *.aux *.*latex* *.fls *.log *.xdv