notebooks: 1-intro.ipynb 2-Corpus-Linguistics.ipynb

html: 1-intro.html 2-Corpus-Linguistics.html

all: notebooks html

1: 1-intro.html

2: 2-Corpus-Linguistics.html

1-intro.ipynb: 1-intro.md
	notedown -o notebooks/$@ $<
	jupyter nbconvert --to notebook --execute --allow-errors --output=$@ notebooks/$@
	rm notebooks/wikipedia_content_analysis.html
	rm notebooks/sometextfile.txt

1-intro.html: 1-intro.ipynb
	jupyter nbconvert --to html --output=../html/$@ notebooks/$<

2-Corpus-Linguistics.ipynb: 2-Corpus-Linguistics.md
	notedown -o notebooks/$@ $<
	jupyter nbconvert --to notebook --execute --allow-errors --output=$@ notebooks/$@

2-Corpus-Linguistics.html: 2-Corpus-Linguistics.ipynb
	jupyter nbconvert --to html --output=../html/$@ notebooks/$<
