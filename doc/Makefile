# compile a report by its name

%:
	jupyter nbconvert --to script $@.ipynb
	junix -f $@.ipynb -o img
	pdflatex $@.tex

clean:
	rm *.aux *.*latex* *.fls *.log *.xdv ./*.py