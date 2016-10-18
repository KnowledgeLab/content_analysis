notebooks: 1-intro.ipynb

html: 1-intro.html

all: notebooks html

1-intro.ipynb: 1-intro.md
	notedown --run -o notebooks/$@ $<
	rm wikipedia_content_analysis.html

1-intro.html: 1-intro.ipynb
	jupyter nbconvert --allow-errors --to html --output=../html/$@ notebooks/$<
