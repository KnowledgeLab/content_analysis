# Layout

+ Searching text for keywords
+ Distribution of terms
+ Correlation
+
+ From last year:
    + Word frequencies
    + Conditional frequencies
    + Statistically significant collocations
    + Distinguishing or Important words and phrases (Wordls!)
        + tfidf
    + POS-tagged words and phrases
    + Lemmatized words and phrases
        + stemmers
    + Dictionary-based annotations.

+ divergences
    + kale

+ Sources
    + US senate press releases
        + e.g. [http://www.reid.senate.gov/press_releases](http://www.reid.senate.gov/press_releases)
    + Tumblr
    + Literature

# Week 2 - Corpus Linguistics

Intro stuff

For this notebook we will be using the following packages

```python
import nltk
import pandas
import matplotlib.pyplot as plt
%matplotlib inline  
```
# Getting our corpuses

To get started we will need some targets, lets start by downloading one of the corpuses from `nltk`. Lets take a look at how that works.

first we can get a list of corpuses avaible from the Gutenburg corpus

```python
print(nltk.corpus.gutenberg.fileids())
print(len(nltk.corpus.gutenberg.fileids()))
```

We can also look at the individual works

```python
nltk.corpus.gutenberg.raw('shakespeare-macbeth.txt')[:1000]
```

All the listed works have been nicely marked up and classified for us so we can do much better than just looking at raw text.

```python
print(nltk.corpus.gutenberg.words('shakespeare-macbeth.txt'))
print(nltk.corpus.gutenberg.sents('shakespeare-macbeth.txt'))
```

If we want to do some analysis we can start by simply counting the number of times each word occurs.

```python
def wordCounter(wordLst):
    wordCounts = {}
    for word in wordLst:
        #We usually need to normalize the case
        wLower = word.lower()
        if wLower in wordCounts:
            wordCounts[wLower] += 1
        else:
            wordCounts[wLower] = 1
    #convert to DataFrame
    countsForFrame = {'word' : [], 'count' : []}
    for w, c in wordCounts.items():
        countsForFrame['word'].append(w)
        countsForFrame['count'].append(c)
    return pandas.DataFrame(countsForFrame)

countedWords = wordCounter(nltk.corpus.gutenberg.words('shakespeare-macbeth.txt'))
countedWords
```

Notice how `wordCounter()` is not a very complicated function. That is because the hard parts have already been done by `nltk`. If we were using unprocessed text we would have to tokenize and determine what to do with the non-word characters.

Lets plot our counts and see what it looks like.

First we need to sort the words by count.

```python
#Doing this in place as we don't need the unsorted DataFrame
countedWords.sort_values('count', ascending=False, inplace=True)
countedWords
```

```python
plt.plot(countedWords['count'])
```
