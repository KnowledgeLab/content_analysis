# Layout

+ Intro algebra
    + news
    + physics
+ doc2vec
    + press releases
+ score function
    + gensim
+ dimensions example, e.g. he-she

```python
#All these packages need to be installed from pip
import gensim#For word2vec, etc
import requests #For downloading our datasets
import nltk #For stop words and stemmers
import numpy as np #For arrays
import pandas #Gives us DataFrames
import matplotlib.pyplot as plt #For graphics
import seaborn #Makes the graphics look nicer

#This 'magic' command makes the plots work better
#in the notebook, don't use it outside of a notebook.
#Also you can ignore the warning
%matplotlib inline

import os #For looking through files
import os.path #For managing file paths

```

# Intro

intro stuff ...

# Getting our corpuses

Instead of downloading our corpora, we have download them ahead of time, a subset of the [senate press releases](https://github.com/lintool/GrimmerSenatePressReleases) are in `data/grimmerPressReleases`. So we will load them into a DataFrame, to do this first we need to define a function to convert directories of text files into DataFrames.

```python
def loadDir(targetDir, category):
    allFileNames = os.listdir(targetDir)
    #We need to make them into useable paths and filter out hidden files
    filePaths = [os.path.join(targetDir, fname) for fname in allFileNames if fname[0] != '.']

    #The dict that will become the DataFrame
    senDict = {
        'category' : [category] * len(filePaths),
        'filePath' : [],
        'text' : [],
    }

    for fPath in filePaths:
        with open(fPath) as f:
            senDict['text'].append(f.read())
            senDict['filePath'].append(fPath)

    return pandas.DataFrame(senDict)
```

Now we can use the function in all the directories in `data/grimmerPressReleases`

```python
dataDir = 'data/grimmerPressReleases'

senReleasesDF = pandas.DataFrame()

for senatorName in [d for d in os.listdir(dataDir) if d[0] != '.']:
    senPath = os.path.join(dataDir, senatorName)
    senReleasesDF = senReleasesDF.append(loadDir(senPath, senatorName), ignore_index = True)

senReleasesDF[:100:10]
```

# Stemming is taking a really long time do to the size of the dataset, so it's been disabled for now

We also want to remove stop words and stem, but tokenizing requires two steps. Word2Vec wants to know the sentence structure as well as simply the words, so the tokenizing is slightly different this time.

```python
#Define the same function as last week
def normlizeTokens(tokenLst, stopwordLst = None, stemmer = None, lemmer = None):
    #We can use a generator here as we just need to iterate over it

    #Lowering the case and removing non-words
    workingIter = (w.lower() for w in tokenLst if w.isalpha())

    #Now we can use the semmer, if provided
    if stemmer is not None:
        workingIter = (stemmer.stem(w) for w in workingIter)

    #And the lemmer
    if lemmer is not None:
        workingIter = (lemmer.lemmatize(w) for w in workingIter)

    #And remove the stopwords
    if stopwordLst is not None:
        workingIter = (w for w in workingIter if w not in stopwordLst)
    #We will return a list with the stopwords removed
    return list(workingIter)

#initialize our stemmer and our stop words
stop_words_nltk = nltk.corpus.stopwords.words('english')
snowball = nltk.stem.snowball.SnowballStemmer('english')

#Apply our functions, notice each row is a list of lists now
senReleasesDF['tokenized_sents'] = senReleasesDF['text'].apply(lambda x: [nltk.word_tokenize(s) for s in nltk.sent_tokenize(x)])
senReleasesDF['normalized_sents'] = senReleasesDF['tokenized_sents'].apply(lambda x: [normlizeTokens(s, stopwordLst = stop_words_nltk, stemmer = None) for s in x])

senReleasesDF[:100:10]
```


#Word2Vec

We will be using the gensim implementation of [Word2Vec](https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.Word2Vec).

To load our data our data we give all the sentences to the trainer

```python
senReleasesW2V = gensim.models.word2vec.Word2Vec(senReleasesDF['tokenized_sents'].sum())
```

Now we can look at a few things

```
senReleasesW2V.most_similar('president')
```

Or find relationships


```python
#Man is to senator as woman is to ...
senReleasesW2V.most_similar(positive = ['woman', 'senator'], negative = ['man'], topn = 5)
#Chinatown is a noble profession
```

Get the vector

```python
senReleasesW2V['president']
```

Get all the vectors

```python
senReleasesW2V.syn0
```

Find what doesn't fit

```python
senReleasesW2V.doesnt_match(['she', 'he', 'her', 'him', 'washington'])
```


Or save for use later

```python
senReleasesW2V.save("data/senpressreleasesWORD2Vec")
```

# APS abstracts

```python
apsDF = pandas.read_csv('data/APSabstracts1950s.csv', index_col = 0)
apsDF['tokenized_sents'] = apsDF['abstract'].apply(lambda x: [nltk.word_tokenize(s) for s in nltk.sent_tokenize(x)])
apsDF['normalized_sents'] = apsDF['tokenized_sents'].apply(lambda x: [normlizeTokens(s, stopwordLst = stop_words_nltk, stemmer = None) for s in x])

apsW2V = gensim.models.word2vec.Word2Vec(apsDF['tokenized_sents'].sum())
```
