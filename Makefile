all:
	./build.sh

final:
	./build.sh final

clean:
	rm *.aux *.bbl *.blg *.idx *.ilg *.ind *.lof *.log *.lot *.out *.toc
	rm *.nlo *.nls *.pyg *.synctex.gz
	rm chapters/*.aux
