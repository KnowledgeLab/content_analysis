notebooks: 1-intro.ipynb

all: notebooks

1-intro.ipynb: 1-intro.md
	notedown --run -o notebooks/$@ $<
