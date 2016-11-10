# Layout

+ Searching text for keywords
+ Distribution of terms
+ Correlation

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
import requests
import nltk
import pandas
import matplotlib.pyplot as plt
%matplotlib inline  

import json
import urllib.parse #For joining urls
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
countedWords[:10]
```

Notice how `wordCounter()` is not a very complicated function. That is because the hard parts have already been done by `nltk`. If we were using unprocessed text we would have to tokenize and determine what to do with the non-word characters.

Lets plot our counts and see what it looks like.

First we need to sort the words by count.

```python
#Doing this in place as we don't need the unsorted DataFrame
countedWords.sort_values('count', ascending=False, inplace=True)
countedWords[:10]
```

```python
plt.plot(range(len(countedWords)), countedWords['count'])
plt.show()
```

This shows the likelihood of a word occurring is inversely proportional to its rank, this effect is called [Zipf's Law](https://en.wikipedia.org/wiki/Zipf%27s_law).

What does the distribution of word lengths look like?

There are many other properties of words we can look at. First lets look at concordance.

To do this we need to load the text into a `ConcordanceIndex`
```python
macbethIndex = nltk.text.ConcordanceIndex(nltk.corpus.gutenberg.words('shakespeare-macbeth.txt'))
```

Then we can get all the words that cooccur with a word, lets look at `'macbeth'`.

```python
macbethIndex.print_concordance('macbeth')
```

weird, `'macbeth'` doesn't occur anywhere in the the text. What happened?

`ConcordanceIndex` is case sensitive, lets try looking for `'Macbeth'`

```python
macbethIndex.print_concordance('Macbeth')
```

That's a lot better

what about something a lot less frequent

```python
print(countedWords[countedWords['word'] == 'Donalbaine'])
macbethIndex.print_concordance('Donalbaine')
```

# Getting press releases

First we need to understand the GitHub API

requests are made to `'https://api.github.com/'` and responses are in JSON, similar to Tumblr's API.

We will get the information on [github.com/lintool/GrimmerSenatePressReleases](https://github.com/lintool/GrimmerSenatePressReleases) as it contains a nice set documents.

```
r = requests.get('https://api.github.com/repos/lintool/GrimmerSenatePressReleases')
senateReleasesData = json.loads(r.text)
print(senateReleasesData.keys())
print(senateReleasesData['description'])
```

What we are interested in is the `'contents_url'`

```python
print(senateReleasesData['contents_url'])
```

We can use this to get any, or all of the files from the repo

```python
r= requests.get('https://api.github.com/repos/lintool/GrimmerSenatePressReleases/contents/raw/Whitehouse')
whitehouseLinks = json.loads(r.text)
whitehouseLinks[0]
```

Now we have a list of information about Whitehouse press releases. Lets look at one of them.

```python
r = requests.get(whitehouseLinks[0]['download_url'])
whitehouseRelease = r.text
print(whitehouseRelease[:1000])
len(whitehouseRelease)
```

Now we have a blob of text we first need to tokenize it.

```python
whTokens = nltk.word_tokenize(whitehouseRelease)
whTokens[:10]
```

`whTokens` is a list of 'words' but it is not perfect. It is better than `.split(' ')` and there are ways to improve it further, but for now it is good enough.

To use this in `nltk` we can convert it into a `Text`.

```python
whText = nltk.Text(whTokens)
```

Then we can do further analysis.

Lets look at few things

```python
print(whText.collocations())
print(whText.common_contexts('stem'))
print(whText.count('stem'))
```

```python
whText.dispersion_plot(['stem', 'cell', 'federal' ,'Lila', 'Barber', 'Whitehouse'])
```

As we have a large number of these records we can instead load them into a `TextCollection`, but first we will need to download them.
