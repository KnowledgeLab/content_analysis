notebooks: 1-intro.ipynb

html: 1-intro.html

all: notebooks html

1-intro.ipynb: 1-intro.md
	notedown -o notebooks/$@ $<
	jupyter nbconvert --to notebook --execute --allow-errors --output=$@ notebooks/$@
	rm notebooks/wikipedia_content_analysis.html

1-intro.html: 1-intro.ipynb
	jupyter nbconvert --to html --output=../html/$@ notebooks/$<
