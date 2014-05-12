
if [ "$1" = "final" ]; then
	echo "Final Version ..."
	pdflatex -synctex=1 -interaction=nonstopmode -shell-escape thesis.tex
	bibtex thesis.aux
	makeindex thesis.idx
	./sort_symb.sh
	pdflatex -synctex=1 -interaction=nonstopmode -shell-escape thesis.tex
	pdflatex -synctex=1 -interaction=nonstopmode -shell-escape thesis.tex
	echo "Final version...................Done."
else
	pdflatex -synctex=1 -interaction=nonstopmode -shell-escape thesis.tex
	echo "Draft version...................Done."
fi
