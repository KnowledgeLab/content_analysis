# Week 8 - Semantic Networks

intro...

# Note

Getting igraph to make plots was a lot of work, the steps I ended up with are:

+ install `cairocffi`, remove `pycairo` if present
+ install `igraph` version `0.7.1` from the github repo, **Do not** use the pypi version, there is a [known issue](https://github.com/igraph/python-igraph/issues/89)

For this notebook we will be using the following packages.

``` python
import nltk
import sklearn
import pandas
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import igraph as ig
```

We will primarily be dealing with graphs in this notebook, so lets first go over how to use them.

To start with lets create an undirected graph


``` python
g = ig.Graph()
g
```

We can add vertices, they are all numbered starting at 0 so this will add vertices `0`, `1` and `2`.

```
g.add_vertices(3)
```

Now we have 3 vertices

``` python
g.vcount()
```

Or if we want to get more information about the graph

```python
print(g.summary())
```

We can give vertices names, or even give the graph a name.

```
g.vs['name'] = ['a', 'b', 'c'] #vs stands for VertexSeq(uence)
```

Now we can get vertices by name

```python
g.vs.find('a')
```

Still pretty boring though

Lets add a couple of edges, notice we can use names or ids

``` python
g.add_edges([(0,1), (1,2), ('a', 'c')])
print(g.summary())
```

Notice the summary has changed

We can also give the edges properties, but instead of names we give weights

```
print("Before weighted", g.is_weighted())
g.es['a', 'b']['weight'] = 4
g.es['a', 'c']['weight'] = 10
print("After weighted", g.is_weighted())
```

Let's visualize it

```python
ig.plot(g)
```

Very exciting

There are a large number of things to do with the graph once we have created it, but we have to move on to using them now.

First lets load our data, the Grimmer corpus

``` python
senReleasesDF = pandas.read_csv('data/senReleasesTraining.csv', index_col = 0)
senReleasesDF[:5]
```

We will be extracting sentences, as well as tokenizing and stemming

```python
def normlizeTokens(tokenLst, stopwordLst = None, stemmer = None, lemmer = None, vocab = None):
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
    if vocab is not None:
        vocab_str = '|'.join(vocab)
        workingIter = (w for w in workingIter if re.match(vocab_str, w))

    return list(workingIter)

stop_words_nltk = nltk.corpus.stopwords.words('english')
snowball = nltk.stem.snowball.SnowballStemmer('english')
wordnet = nltk.stem.WordNetLemmatizer()
```

``` python
senReleasesDF['tokenized_sents'] = senReleasesDF['text'].apply(lambda x: [nltk.word_tokenize(s) for s in nltk.sent_tokenize(x)])
senReleasesDF['normalized_sents'] = senReleasesDF['tokenized_sents'].apply(lambda x: [normlizeTokens(s, stopwordLst = stop_words_nltk, stemmer = None) for s in x])

senReleasesDF[:5]
```
