# Layout

Notebook plan (next week):
Clustering documents
hierarchical clustering
k-means clustering
visualization of methods of cluster identification
topic modeling with many/few topics (adjust alpha, beta)
topic model visualization
topic model interpretation/statistics
matrix of topic similarity corpora
topic model extension - correlated, dynamic, author

+ Opening
+ packages
    + scikit-learn
    + gensim
+ Introduce ML
+ Training
    + Use sklearn dataset examples
+ Extraction
    + Probabilistic models
    + Deterministic models
    + stop lists
    + cleaning
    + Performance
    + [sklearn](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.cluster)
        + [Tfidf](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
        + [KMeans](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
        + Hierarchical
    + [gensim](http://radimrehurek.com/gensim/apiref.html)
        + [LDA](https://radimrehurek.com/gensim/models/ldamodel.html)
            + Expansions on LDA
        + [word2vec](https://radimrehurek.com/gensim/models/word2vec.html)
        + Doc2Vec
    + Implement our own models
        + regressions

# Week 3 - Clustering

Intro stuff ...

For this notebook we will be using the following packages

```python
#All these packages need to be installed from pip
#These are all for the ML/cluster detection
import sklearn
import sklearn.feature_extraction.text
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.datasets
import sklearn.cluster

import gensim#For topic modeling
import nltk #the Natural Language Toolkit
import requests #For downloading our datasets
import numpy as np
import pandas #gives us DataFrames
import matplotlib.pyplot as plt #For graphics
import seaborn as sns #Makes the graphics look nicer

#This 'magic' command makes the plots work better
#in the notebook, don't use it outside of a notebook.
#Also you can ignore the warning, it
%matplotlib inline

import metaknowledge as mk

import time
import json
```

# Getting our corpuses

We can get a dataset to work on from sklearn

```python
#data_home argument will let you change the download location

newsgroups = sklearn.datasets.fetch_20newsgroups(subset='train')
print(dir(newsgroups))
```

We can get the categories with `target_names` or the actual files with `filenames`
```python
print(newsgroups.target_names)
print(len(newsgroups.data))
```

lets reduce our dataset for this analysis and drop some of the extraneous information

```python
categories = ['comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos']
newsgroups = sklearn.datasets.fetch_20newsgroups(subset='train', categories = categories, remove=['headers', 'footers', 'quotes'])
```

The contents of the files are stored in `data`

```python
print(len(newsgroups.data))
print("\n".join(newsgroups.data[2].split("\n")[:15]))
```

We can start by converting the documents into count vectors

```python
#First it needs to be initialized
ngCountVectorizer = sklearn.feature_extraction.text.CountVectorizer()
#Then trained
newsgroupsVects = ngCountVectorizer.fit_transform(newsgroups.data)
print(newsgroupsVects.shape)
```

This gives us a matrix with each row a document and each column a word, the matrix is mostly zeros, so it is stored as a sparse matrix

```python
newsgroupsVects
```

But we can use the normal operations on it or even, convert it to normal matrix

```python
newsgroupsVects[:10,:20].todense()
```

We can also lookup the indices of different words using the Vectorizer

```python
ngCountVectorizer.vocabulary_.get('vector')
```

But there are some more interesting things to do

Lets started with [term frequency–inverse document frequency](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html)(tf-idf)

```python
#initialize
newsgroupsTFTransformer = sklearn.feature_extraction.text.TfidfTransformer().fit(newsgroupsVects)
#train
newsgroupsTF = newsgroupsTFTransformer.transform(newsgroupsVects)
print(newsgroupsTF.shape)
```

This gives us the tf-idf for each word in each text

```python
list(zip(ngCountVectorizer.vocabulary_.keys(), newsgroupsTF.data))[:20]
```

Lots of garbage from unique words and stopwords, but it is a start. We should normally filter out stop words, stem and lem our data before vectorizering, or we can instead tf-idf to filter our data, for exact explanation of all the options look [here](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html), generally, we've limited it to at most 5000 words, as well as limited it to words with at least 3 occurrences, and that aren't in more than half the documents.

```python
#initialize
ngTFVectorizer = sklearn.feature_extraction.text.TfidfVectorizer(max_df=0.5, max_features=5000, min_df=3, stop_words='english', norm='l2')
#train
newsgroupsTFVects = ngTFVectorizer.fit_transform(newsgroups.data)
```

Lets look at the matrix

```python
newsgroupsTFVects
```

Its much smaller now, only 5000 words, but the same number of documents

We can still look at the words

```python
try:
    print(ngTFVectorizer.vocabulary_['vector'])
except KeyError:
    print('vector is missing')
    print('The available words are: {} ...'.format(list(ngTFVectorizer.vocabulary_.keys())[:10]))
```

This is a good matrix to start finding clusters with though

#K-means

Lets start with k-means

To do this we will need to know how many clusters we're looking for

```python
numClusters = len(set(newsgroups.target_names))
numClusters
```

Then we can initialize our cluster finder

```python
#k-means++ is a better way of finding the starting points
#We could also try providing our own
km = sklearn.cluster.KMeans(n_clusters=numClusters, init='k-means++')
```

And now we can calculate the clusters

```python
km.fit(newsgroupsTFVects)
```

Once we have the clusters there are a variety of metrics that sklearn provides, we will look at a few

```python
print("The available metrics are: {}".format([s for s in dir(sklearn.metrics) if s[0] != '_']))
print("for our clusters:")
print("Homogeneity: {:0.3f}".format(sklearn.metrics.homogeneity_score(newsgroups.target, km.labels_)))
print("Completeness: {:0.3f}".format(sklearn.metrics.completeness_score(newsgroups.target, km.labels_)))
print("V-measure: {:0.3f}".format(sklearn.metrics.v_measure_score(newsgroups.target, km.labels_)))
```

We can also look at the contents of the clusters

```python
terms = vectorizer.get_feature_names()
print("Top terms per cluster:")
order_centroids = km.cluster_centers_.argsort()[:, ::-1]
for i in range(true_k):
    print("Cluster %d:" % i)
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind])
    print('\n')
```

#Hierarchical Clustering

Instead of looking at the matrix of documents to words, we can look at how the words relate the documents to each other.

To do this we will take our matrix of word counts per document `newsgroupsTFVects` and


# Gensim

To do topic modeling we will again be using data from the [grimmer press releases corpus](ttps://github.com/lintool/GrimmerSenatePressReleases). Lets start by defining the same function as last lesson and loading a few press releases from Obama into a DataFrame.

```python
def getGithubFiles(target, maxFiles = 100):
    #We are setting a max so our examples don't take too long to run
    #For converting to a DataFrame
    releasesDict = {
        'name' : [], #The name of the file
        'text' : [], #The text of the file, watch out for binary files
        'path' : [], #The path in the git repo to the file
        'html_url' : [], #The url to see the file on Github
        'download_url' : [], #The url to download the file
    }

    #Get the directory information from Github
    r = requests.get(target)

    #Check for rate limiting
    if r.status_code != 200:
        raise RuntimeError("Github didn't like your request, you have probably been rate limited.")
    filesLst = json.loads(r.text)

    for fileDict in filesLst[:maxFiles]:
        #These are provided by the directory
        releasesDict['name'].append(fileDict['name'])
        releasesDict['path'].append(fileDict['path'])
        releasesDict['html_url'].append(fileDict['html_url'])
        releasesDict['download_url'].append(fileDict['download_url'])

        #We need to download the text though
        text = requests.get(fileDict['download_url']).text
        releasesDict['text'].append(text)

    return pandas.DataFrame(releasesDict)

obReleases = getGithubFiles('https://api.github.com/repos/lintool/GrimmerSenatePressReleases/contents/raw/Obama', maxFiles = 20)
obReleases[:5]
```

Now we have the files we can tokenize and normalize

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

#Apply our functions
obReleases['tokenized_text'] = obReleases['text'].apply(lambda x: nltk.word_tokenize(x))
obReleases['normalized_tokens'] = obReleases['tokenized_text'].apply(lambda x: normlizeTokens(x, stopwordLst = stop_words_nltk, stemmer = snowball))
```

To use the texts with gensim we need to create a `corpua` object, this takes a few steps. First we create a `Dictioanry` that maps tokens to ids.

```python
dictionary = gensim.corpora.Dictionary(obReleases['normalized_tokens'])
```

Then for each of the texts we create a list of tuples containing: each token and its count. We will only use the first half of our dataset for now, and will leave the second half to test with.

```python
corpus = [dictionary.doc2bow(text) for text in obReleases['normalized_tokens'][:10]]
```

Then we serialize the corpus as a file and load it. This is an important step when the corpus is large.

```python
gensim.corpora.MmCorpus.serialize('obama.mm', corpus)
obmm = gensim.corpora.MmCorpus('obama.mm')
```

Now we have a correctly formatted corpura that we can use for some topic models

```
oblda = gensim.models.ldamodel.LdaModel(corpus=obmm, id2word=dictionary, num_topics=10)
```

We can check how well different texts belong to different topics, heres one of the texts from the training set

```python
ob1Bow = dictionary.doc2bow(obReleases['normalized_tokens'][0])
ob1lda = oblda[ob1Bow]
ob1lda
```

and one from the withheld set

```python
ob11Bow = dictionary.doc2bow(obReleases['normalized_tokens'][11])
ob11lda = oblda[ob1Bow]
ob11lda
```
