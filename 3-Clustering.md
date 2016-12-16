# Layout

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
import sklearn
import sklearn.feature_extraction.text
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.datasets
import sklearn.cluster

import gensim
import nltk
import numpy as np
import pandas as pd
import metaknowledge as mk

import time
```

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

The contents are stored in `data`

```python
print(len(newsgroups.data))
print("\n".join(newsgroups.data[2].split("\n")[:15]))
```

```python
count_vect = sklearn.feature_extraction.text.CountVectorizer()
X_train_counts = count_vect.fit_transform(newsgroups.data)
print(X_train_counts.shape)
print(count_vect.vocabulary_.get('algorithm'))
```

```python
tf_transformer = sklearn.feature_extraction.text.TfidfTransformer(use_idf=True).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)
print(X_train_tf.shape)
```

```python
tfidf_transformer = sklearn.feature_extraction.text.TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
print(X_train_tfidf.shape)
```

```python
list(zip(count_vect.vocabulary_.keys(), X_train_tfidf.data))[:15]
```

Lots of garabge from unique words and stopwords

```python
vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(max_df=0.5, max_features=10000, min_df=3, stop_words='english', norm='l2', use_idf=True)
X = vectorizer.fit_transform(newsgroups.data)
list(zip(vectorizer.get_feature_names()[3000:3010], X.data[3000:3010]))
```

```python
true_k = np.unique(newsgroups.target_names).shape[0]
true_k
```

```python
km = sklearn.cluster.KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1, verbose=1)

print("Clustering sparse data with {}".format(km))
t0 = time.time()
km.fit(X) #km.fit(X_train_tfidf)
print("done in {:0.3f}s".format(time.time() - t0))

print("Homogeneity: {:0.3f}".format(sklearn.metrics.homogeneity_score(newsgroups.target, km.labels_)))
print("Completeness: {:0.3f}".format(sklearn.metrics.completeness_score(newsgroups.target, km.labels_)))
print("V-measure: {:0.3f}".format(sklearn.metrics.v_measure_score(newsgroups.target, km.labels_)))
print("Adjusted Rand-Index: {:.3f}".format(sklearn.metrics.adjusted_rand_score(newsgroups.target, km.labels_)))
print("Silhouette Coefficient: {:0.3f}".format(sklearn.metrics.silhouette_score(X, newsgroups.target, sample_size=1000)))
```

```python
sklearn.metrics.homogeneity_score??
```

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


# Gensim

loading abstracts from raw wos data

```python
RC = mk.RecordCollection('../data/imetricrecs.txt')
recsData = {'abstract' : [], 'id' : [], 'authors' : []}
for R in RC:
    if R.get('abstract') is not None:
        recsData['abstract'].append(R.get('abstract'))
        recsData['id'].append(R.id)
        recsData['authors'].append(R.get('authorsFull'))
imetric_abstracts = pd.DataFrame(recsData)
imetric_abstracts[:10]
```

Lets tokenize and filter the abstracts a bit


```python
stoplist = set('for a of the and to in'.split())
def abstractFilter(abString):
    sents = nltk.sent_tokenize(abString)
    texts = [word for sent in sents for word in sent.lower().split() if word not in stoplist]
    return texts

imetric_abstracts['abs'] = imetric_abstracts['abstract'].apply(abstractFilter)
imetric_abstracts[:10]
```

```python
bigram = gensim.models.Phrases(imetric_abstracts['abs'])
bigrammed = (bigram[imetric_abstracts['abs']])
trigram = gensim.models.Phrases(bigrammed)
trigrammed = (trigram[bigrammed])
```

```python
modelSaveLoc = '../imetricsmodel'

start = time.time()
model = gensim.models.Word2Vec(trigrammed, workers=4, batch_words=10000)

for iteration in range(10):
    model.train(trigrammed)

vocab_matrix = model.syn0
vocabulary = model.index2word

model.save(modelSaveLoc)

end = time.time()
print(end - start)
```

```python
model = gensim.models.Word2Vec.load(modelSaveLoc)
model['renewable']
```
