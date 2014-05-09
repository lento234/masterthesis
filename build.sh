
if [ "$1" = "final" ]; then
	pdflatex -synctex=1 -interaction=nonstopmode thesis.tex
	bibtex thesis.aux
	makeindex thesis.idx
	./sort_symb.sh
	pdflatex -synctex=1 -interaction=nonstopmode thesis.tex
	pdflatex -synctex=1 -interaction=nonstopmode thesis.tex
	echo "Final version done!"
else
	pdflatex -synctex=1 -interaction=nonstopmode thesis.tex
	echo "Draft version done!"
fi
