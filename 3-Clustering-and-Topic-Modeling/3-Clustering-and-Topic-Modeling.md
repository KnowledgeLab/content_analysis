# Week 3 - Clustering & Topic Modeling

This week, we take a text corpus that we have developed, and we first break it
into discrete document chunks through a process known as clustering or
partitioning. We will pilot this here both with a well-known *flat* clustering
method, `kmeans`, and also a *hierarchical* approach, `Ward's (minimum variance)
method`. We will demonstrate a simple (graphical) approach to identifying
optimal cluster number, the sillhouette method, and evaluate the quality of
unsupervised clusters on labeled data. Next, we will explore a method of content
clustering called topic modeling. This statistical technique models and
computationally induces *topics* from data, which are sparse distributions over
(nonexclusive clusters of) words, from which documents can formally be described
as sparse mixtures. We will explore these topics and consider their utility for
understanding trends within a corpus. Finally, we will consider how to construct
models that take document cluster and topic loading as predictive features.

For this notebook we will be using the following packages

```python
#All these packages need to be installed from pip
#These are all for the cluster detection
import sklearn
import sklearn.feature_extraction.text
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.datasets
import sklearn.cluster
import sklearn.decomposition
import sklearn.metrics

import scipy #For hierarchical clustering and some visuals
#import scipy.cluster.hierarchy
import gensim#For topic modeling
import nltk #the Natural Language Toolkit
nltk.data.path.append('../data/nltk_data')
import requests #For downloading our datasets
import numpy as np #for arrays
import pandas #gives us DataFrames
import matplotlib.pyplot as plt #For graphics
import matplotlib.cm #Still for graphics
import seaborn as sns #Makes the graphics look nicer

#This 'magic' command makes the plots work better
#in the notebook, don't use it outside of a notebook.
#Also you can ignore the warning, it
%matplotlib inline

import json
```

# Getting our corpora

To begin, we will use a well known corpus of testing documents from the *20
Newsgroups corpus*, a dataset commonly used to illustrate text applications of
text clustering and classification. This comes packaged with sklearn and
comprises approximately 20,000 newsgroup documents, partitioned (nearly) evenly
across 20 newsgroups. It was originally collected by Ken Lang, probably for his
1995 *Newsweeder: Learning to filter netnews* paper. The data is organized into
20 distinct newsgroups, each corresponding to a different topic. Some of the
newsgroups are very closely related (e.g. comp.sys.ibm.pc.hardware /
comp.sys.mac.hardware), while others are unrelated (e.g misc.forsale /
soc.religion.christian).

```python
newsgroups = sklearn.datasets.fetch_20newsgroups(subset='train', data_home = '../data/scikit_learn_data')
print(dir(newsgroups))
```

We can ascertain the categories with `target_names` or the actual files with
`filenames`

```python
print(newsgroups.target_names)
print(len(newsgroups.data))
```

We will start by converting the provided data into pandas DataFrames.

First we reduce our dataset for this analysis by dropping some extraneous
information and converting it into a DataFrame.

```python
newsgroupsCategories = ['comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos']

newsgroupsDF = pandas.DataFrame(columns = ['text', 'category', 'source_file'])

for category in newsgroupsCategories:
    print("Fetching data for: {}".format(category))
    ng = sklearn.datasets.fetch_20newsgroups(subset='train', categories = [category], remove=['headers', 'footers', 'quotes'], data_home = '../data/scikit_learn_data/')
    newsgroupsDF = newsgroupsDF.append(pandas.DataFrame({'text' : ng.data, 'category' : [category] * len(ng.data), 'source_file' : ng.filenames}), ignore_index=True)

#Creating an explicit index column for later

#newsgroupsDF['index'] = range(len(newsgroupsDF))
#newsgroupsDF.set_index('index', inplace = True)
print(len(newsgroupsDF))
newsgroupsDF[:10]
```

Next, we can convert the documents into word count vectors (e.g.,
*soc.religion.christian message a* might contain 3 mentions of "church", 2 of
"jesus", 1 of "religion", etc., yielding a CountVector=[3,2,1,...])

```python
#First it needs to be initialized
ngCountVectorizer = sklearn.feature_extraction.text.CountVectorizer()
#Then trained
newsgroupsVects = ngCountVectorizer.fit_transform(newsgroupsDF['text'])
print(newsgroupsVects.shape)
```

This gives us a matrix with row a document and each column a word. The matrix is
mostly zeros, so we store it as a sparse matrix, a data structure that contains
and indexes only the nonzero entries.

```python
newsgroupsVects
```

We can use the normal operations on this sparse matrix or convert it to normal
matrix (not recommended for large sparse matrices :-)

```python
newsgroupsVects[:10,:20].todense()
```

We can also lookup the indices of different words using the Vectorizer

```python
ngCountVectorizer.vocabulary_.get('vector')
```

There are some more interesting things to do...

Lets start with [term frequency–inverse document frequency](http://scikit-learn.
org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.ht
ml)(tf-idf), a method for weighting document-distinguishing words.

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

At first glance, there appears to be a lot of garbage littering this unordered
list with unique words and stopwords. Note, however, that words like *apple*,
*rgb*, and *voltage* distinguish this newsgroup document, while stopwords post a
much lower weight. Note that we could filter out stop words, stem and lem our
data before vectorizering, or we can instead use tf-idf to filter our data (or
**both**). For exact explanation of all options look [here](http://scikit-learn.
org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.htm
l). To prune this matrix of features, we now limit our word vector to 1000 words
with at least 3 occurrences, which do not occur in more than half of the
documents. There is an extensive science and art to feature engineering for
machine learning applications like clustering.

```python
#initialize
ngTFVectorizer = sklearn.feature_extraction.text.TfidfVectorizer(max_df=0.5, max_features=1000, min_df=3, stop_words='english', norm='l2')
#train
newsgroupsTFVects = ngTFVectorizer.fit_transform(newsgroupsDF['text'])
```

Lets look at the matrix

```python
newsgroupsTFVects
```

The matrix is much smaller now, only 1000 words, but the same number of
documents

We can still look at the words:

```python
try:
    print(ngTFVectorizer.vocabulary_['vector'])
except KeyError:
    print('vector is missing')
    print('The available words are: {} ...'.format(list(ngTFVectorizer.vocabulary_.keys())[:10]))
```

This is a reasonable matrix of features with which to begin identifying
clusters.

# Flat Clustering with K-means

Lets start with k-means, an approach that begins with random clusters of
predefined number, then iterates cluster reassignment and evaluates the new
clusters relative to an objective function, recursively.

To do this we will need to know how many clusters we are looking for. Here the
*true number* of clusters is 4. Of course, in most cases you would not know the
number in advance.

```python
numClusters = len(set(newsgroupsDF['category']))
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

Once we have the clusters, we can evaluate them with a variety of metrics that
sklearn provides. We will look at a few.

```python
print("The available metrics are: {}".format([s for s in dir(sklearn.metrics) if s[0] != '_']))
print("for our clusters:")
print("Homogeneity: {:0.3f}".format(sklearn.metrics.homogeneity_score(newsgroupsDF['category'], km.labels_)))
print("Completeness: {:0.3f}".format(sklearn.metrics.completeness_score(newsgroupsDF['category'], km.labels_)))
print("V-measure: {:0.3f}".format(sklearn.metrics.v_measure_score(newsgroupsDF['category'], km.labels_)))
```

We can also look at the contents of the clusters:

```python
terms = ngTFVectorizer.get_feature_names()
print("Top terms per cluster:")
order_centroids = km.cluster_centers_.argsort()[:, ::-1]
for i in range(numClusters):
    print("Cluster %d:" % i)
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind])
    print('\n')
```

Let's construct a visualization of the clusters. First, we will first reduce the
dimensionality of the data using principal components analysis (PCA).

```python
PCA = sklearn.decomposition.PCA
pca = PCA(n_components = 2).fit(newsgroupsTFVects.toarray())
reduced_data = pca.transform(newsgroupsTFVects.toarray())
```

The cell below is optional. It allows you to do a biplot

```python
components = pca.components_
keyword_ids = list(set(order_centroids[:,:10].flatten())) #Get the ids of the most distinguishing words(features) from your kmeans model.
words = [terms[i] for i in keyword_ids]#Turn the ids into words.
x = components[:,keyword_ids][0,:] #Find the coordinates of those words in your biplot.
y = components[:,keyword_ids][1,:]
```

Then, let's build a color map for the true labels.

```python
colordict = {
'comp.sys.mac.hardware': 'red',
'comp.windows.x': 'orange',
'misc.forsale': 'green',
'rec.autos': 'blue',
    }
colors = [colordict[c] for c in newsgroupsDF['category']]
print("The categories' colors are:\n{}".format(colordict.items()))
```

Let's plot the data using the true labels as the colors of our data points.

```python
fig = plt.figure(figsize = (10,6))
ax = fig.add_subplot(111)
ax.set_frame_on(False)
ax.scatter(reduced_data[:, 0], reduced_data[:, 1], color = colors, alpha = 0.5, label = colors)
plt.xticks(())
plt.yticks(())
plt.title('True Classes')
plt.show()
```

One nice thing about PCA is that we can also do a biplot and map our feature
vectors to the same space.

```python
fig = plt.figure(figsize = (16,9))
ax = fig.add_subplot(111)
ax.set_frame_on(False)
ax.scatter(reduced_data[:, 0], reduced_data[:, 1], color = colors, alpha = 0.3, label = colors)
for i, word in enumerate(words):
    ax.annotate(word, (x[i],y[i]))
plt.xticks(())
plt.yticks(())
plt.title('True Classes')
plt.show()
```

Let's do it again with predicted clusters.

```python
colors_p = [colordict[newsgroupsCategories[l]] for l in km.labels_]
```

```python
fig = plt.figure(figsize = (10,6))
ax = fig.add_subplot(111)
ax.set_frame_on(False)
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], color = colors_p, alpha = 0.5)
plt.xticks(())
plt.yticks(())
plt.title('Predicted Clusters\n k = 4')
plt.show()
```

Let's try with 3 clusters.

```python
km3 = sklearn.cluster.KMeans(n_clusters= 3, init='k-means++')
km3.fit(newsgroupsTFVects.toarray())
```

# Selecting Cluster Number

Now we demonstrate the Silhouette method, one approach by which optimal number
of clusters can be ascertained. Many other methods exist (e.g., Bayesian
Information Criteria or BIC, the visual "elbow criteria", etc.)

First we will define a helper function



```python
def plotSilhouette(n_clusters, X):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize = (15,5))
    
    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])
    clusterer = sklearn.cluster.KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(X)
    
    silhouette_avg = sklearn.metrics.silhouette_score(X, cluster_labels)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = sklearn.metrics.silhouette_samples(X, cluster_labels)

    y_lower = 10
    
    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = matplotlib.cm.spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        y_lower = y_upper + 10
    
    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = matplotlib.cm.spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(reduced_data[:, 0], reduced_data[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                c=colors)

    # Labeling the clusters
    centers = clusterer.cluster_centers_
    projected_centers = pca.transform(centers)
    # Draw white circles at cluster centers
    ax2.scatter(projected_centers[:, 0], projected_centers[:, 1],
                marker='o', c="white", alpha=1, s=200)

    for i, c in enumerate(projected_centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1, s=50)

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("PC 1")
    ax2.set_ylabel("PC 2")

    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')
    plt.show()
    print("For n_clusters = {}, The average silhouette_score is : {:.3f}".format(n_clusters, silhouette_avg))
```

Now we can examine a few different numbers of clusters

```python
X = newsgroupsTFVects.toarray()
plotSilhouette(3, X)
```

```python
X = newsgroupsTFVects.toarray()
plotSilhouette(4, X)
```

```python
X = newsgroupsTFVects.toarray()
plotSilhouette(5, X)
```

```python
X = newsgroupsTFVects.toarray()
plotSilhouette(6, X)
```

Interestingly, the silhouette scores above suggests that 3 is a better number of
clusters than 4, which would be accurate if we (reasonsably) grouped the two
computer-themed groups.

# Getting arbitrary text data

Lets start by defining the same function as last lesson and loading a few press
releases from 10 different senators into a DataFrame. The code to do this is
below, but commented out as we've already downloaded the data to the data
directory.

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

targetSenator = 'Kennedy'# = ['Voinovich', 'Obama', 'Whitehouse', 'Snowe', 'Rockefeller', 'Murkowski', 'McCain', 'Kyl', 'Baucus', 'Frist']
"""
#Uncomment this to download your own data
senReleasesTraining = pandas.DataFrame()

print("Fetching {}'s data".format(targetSenator))
targetDF = getGithubFiles('https://api.github.com/repos/lintool/GrimmerSenatePressReleases/contents/raw/{}'.format(targetSenator), maxFiles = 2000)
targetDF['targetSenator'] = targetSenator
senReleasesTraining = senReleasesTraining.append(targetDF, ignore_index = True)

#Watch out for weird lines when converting to csv
#one of them had to be removed from the Kennedy data so it could be re-read
senReleasesTraining.to_csv("data/senReleasesTraining.csv")
"""

senReleasesTraining = pandas.read_csv("../data/senReleasesTraining.csv")

senReleasesTraining[:5]
```

Now we have the files we can tokenize and normalize

The normalized text is good, but we know that the texts will have a large amount
of overlap so we can use tf-idf to remove some of the most frequent words.
Before doing that, there is one empty cell, let's remove that.

```python
senReleasesTraining = senReleasesTraining.dropna(axis=0, how='any')
```

```python
#Similar parameters to before, but stricter max df and no max num occurrences
senTFVectorizer = sklearn.feature_extraction.text.TfidfVectorizer(max_df=100, min_df=2, stop_words='english', norm='l2')
senTFVects = senTFVectorizer.fit_transform(senReleasesTraining['text'])
senTFVectorizer.vocabulary_.get('senat', 'Missing "Senate"')
```

# Clustering with our new data

One nice thing about using DataFrames for everything is that we can quickly
convert code from one input to another. Below we are redoing the cluster
detection with our senate data. If you setup your DataFrame the same way it
should be able to run on this code, without much work.

First we will define what we will be working with

```python
targetDF = senReleasesTraining
textColumn = 'text'
numCategories = 3
```

Tf-IDf vectorizing

```python
exampleTFVectorizer = sklearn.feature_extraction.text.TfidfVectorizer(max_df=0.5, max_features=1000, min_df=3, stop_words='english', norm='l2')
#train
exampleTFVects = ngTFVectorizer.fit_transform(targetDF[textColumn])
```

Running k means

```python
exampleKM = sklearn.cluster.KMeans(n_clusters = numCategories, init='k-means++')
exampleKM.fit(exampleTFVects)
```

And visualize, this is more up to you, but we will do one

```python
examplePCA = sklearn.decomposition.PCA(n_components = 2).fit(exampleTFVects.toarray())
reducedPCA_data = examplePCA.transform(exampleTFVects.toarray())

colors = list(plt.cm.rainbow(np.linspace(0,1, numCategories)))
colors_p = [colors[l] for l in exampleKM.labels_]
```

```python
fig = plt.figure(1)
ax = fig.add_subplot(111)
ax.set_frame_on(False)
plt.scatter(reducedPCA_data[:, 0], reducedPCA_data[:, 1], color = colors_p, alpha = 0.5)
plt.xticks(())
plt.yticks(())
plt.title('Predicted Clusters\n k = {}'.format(numCategories))
plt.show()
```

In this case, there are probably two clusters. You can check with a Silhouette
analysis.

# Hierarchical Clustering with Wald's Method

Next we approach a hierchical clustering method, which proposes nested clusters
at any resolution (at the finest resolution, every document is its own cluster).

Here we must begin by calculating how similar the documents are to one another.

As a first pass, we take our matrix of word counts per document
`newsgroupsTFVects` and create a word occurrence matrix measuring how similar
the documents are to each other based on their number of shared words. (Note one
could perform the converse operation, a document occurrence matrix measuring how
similar  words are to each other based on their number of collocated documents).

```python
newsgroupsCoocMat = newsgroupsTFVects * newsgroupsTFVects.T
#set the diagonal to 0 since we don't care how similar texts are to themselves
newsgroupsCoocMat.setdiag(0)
#Another way of relating the texts is with their cosine similarity
#newsgroupsCosinMat1 = 1 - sklearn.metrics.pairwise.cosine_similarity(newsgroupsTFVects)
#But generally word occurrence is more accurate

```

Now we can compute a tree of nested clusters. Here we will only look at the
first 100 texts.

```python
linkage_matrix = scipy.cluster.hierarchy.ward(newsgroupsCoocMat[:100, :100].toarray())
linkage_matrix[:10]
```

Now we can visualize the tree

```python
ax = scipy.cluster.hierarchy.dendrogram(linkage_matrix)
```

This plot may seem somewhat unwieldy. To make it easier to read, we can cut the
tree after a number of branchings.

```python
ax = scipy.cluster.hierarchy.dendrogram(linkage_matrix, p=4, truncate_mode='level')
```

By default, the tree is colored to show the clusters based on their ['distance']
(https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.cluster.hiera
rchy.dendrogram.html#scipy.cluster.hierarchy.dendrogram) from one another, but
there are other ways of forming hierarchical clusters.

Another approach involves cutting the tree into `n` branches. We can do this
with [`fcluster()`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.c
luster.hierarchy.fcluster.html#scipy.cluster.hierarchy.fcluster). Lets break the
tree into 4 clusters

```python
hierarchicalClusters = scipy.cluster.hierarchy.fcluster(linkage_matrix, 4, 'maxclust')
hierarchicalClusters
```

This gives us an array giving each element of `linkage_matrix`'s cluster. The
leader function below is actually quite misleading. In this case, the ids it
returns are actually not document ids but non-singleton clusters. You can ignore
this cell.

```python
clusterLeaders = scipy.cluster.hierarchy.leaders(linkage_matrix, hierarchicalClusters)
clusterLeaders
```

# Now let's do it with our new data

We can also do hierarchical clustering with the Senate data. Let's start by
creating the linkage matrix:

```python
exampleCoocMat = exampleTFVects * exampleTFVects.T
exampleCoocMat.setdiag(0)
examplelinkage_matrix = scipy.cluster.hierarchy.ward(exampleCoocMat[:100, :100].toarray())
```

And visualize the tree:

```python
ax = scipy.cluster.hierarchy.dendrogram(examplelinkage_matrix, p=5, truncate_mode='level')
```

Sometimes, it's not very interesting :-).

# Gensim

To do topic modeling we will also be using data from the [grimmer press releases
corpus](ttps://github.com/lintool/GrimmerSenatePressReleases). To use the texts
with gensim we need to create a `corpua` object, this takes a few steps. First
we create a `Dictionary` that maps tokens to ids.

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
senReleasesTraining['tokenized_text'] = senReleasesTraining['text'].apply(lambda x: nltk.word_tokenize(x))
senReleasesTraining['normalized_tokens'] = senReleasesTraining['tokenized_text'].apply(lambda x: normlizeTokens(x, stopwordLst = stop_words_nltk, stemmer = snowball))

senReleasesTraining[::100]
```

```python
def dropMissing(wordLst, vocab):
    return [w for w in wordLst if w in vocab]

senReleasesTraining['reduced_tokens'] = senReleasesTraining['normalized_tokens'].apply(lambda x: dropMissing(x, senTFVectorizer.vocabulary_.keys()))
```

```python
dictionary = gensim.corpora.Dictionary(senReleasesTraining['reduced_tokens'])
```

Then for each of the texts we create a list of tuples containing each token and
its count. We will only use the first half of our dataset for now and will save
the remainder for testing.

```python
corpus = [dictionary.doc2bow(text) for text in senReleasesTraining['reduced_tokens']]
```

Then we serialize the corpus as a file and load it. This is an important step
when the corpus is large.

```python
gensim.corpora.MmCorpus.serialize('senate.mm', corpus)
senmm = gensim.corpora.MmCorpus('senate.mm')
```

Now we have a correctly formatted corpus that we can use for topic modeling and
induction.

```python
senlda = gensim.models.ldamodel.LdaModel(corpus=senmm, id2word=dictionary, num_topics=10, alpha='auto', eta='auto')
```

We can inspect the degree to which distinct texts load on different topics. Here
is one of the texts from the training set:

```python
sen1Bow = dictionary.doc2bow(senReleasesTraining['reduced_tokens'][0])
sen1lda = senlda[sen1Bow]
print("The topics of the text: {}".format(senReleasesTraining['name'][0]))
print("are: {}".format(sen1lda))
```

We can now see which topics our model predicts press releases load on and make
this into a `dataFrame` for later analysis.

```python
ldaDF = pandas.DataFrame({
        'name' : senReleasesTraining['name'],
        'topics' : [senlda[dictionary.doc2bow(l)] for l in senReleasesTraining['reduced_tokens']]
    })
```

This is a bit unwieldy so lets make each topic its own column:

```python
#Dict to temporally hold the probabilities
topicsProbDict = {i : [0] * len(ldaDF) for i in range(senlda.num_topics)}

#Load them into the dict
for index, topicTuples in enumerate(ldaDF['topics']):
    for topicNum, prob in topicTuples:
        topicsProbDict[topicNum][index] = prob

#Update the DataFrame
for topicNum in range(senlda.num_topics):
    ldaDF['topic_{}'.format(topicNum)] = topicsProbDict[topicNum]

ldaDF[1::100]
```

Now let's visualize this for several (e.g., 10) documents in the corpus. First
we'll subset the data:

```python
ldaDFV = ldaDF[:10][['topic_%d' %x for x in range(10)]]
ldaDFVisN = ldaDF[:10][['name']]
ldaDFVis = ldaDFV.as_matrix(columns=None)
ldaDFVisNames = ldaDFVisN.as_matrix(columns=None)
ldaDFV
```

First we can visualize as a stacked bar chart:

```python
N = 10
ind = np.arange(N)
K = senlda.num_topics  # N documents, K topics
ind = np.arange(N)  # the x-axis locations for the novels
width = 0.5  # the width of the bars
plots = []
height_cumulative = np.zeros(N)

for k in range(K):
    color = plt.cm.coolwarm(k/K, 1)
    if k == 0:
        p = plt.bar(ind, ldaDFVis[:, k], width, color=color)
    else:
        p = plt.bar(ind, ldaDFVis[:, k], width, bottom=height_cumulative, color=color)
    height_cumulative += ldaDFVis[:, k]
    plots.append(p)
    

plt.ylim((0, 1))  # proportions sum to 1, so the height of the stacked bars is 1
plt.ylabel('Topics')

plt.title('Topics in Press Releases')
plt.xticks(ind+width/2, ldaDFVisNames, rotation='vertical')

plt.yticks(np.arange(0, 1, 10))
topic_labels = ['Topic #{}'.format(k) for k in range(K)]
plt.legend([p[0] for p in plots], topic_labels, loc='center left', frameon=True,  bbox_to_anchor = (1, .5))

plt.show()
```

We can also visualize as a heat map:

```python
plt.pcolor(ldaDFVis, norm=None, cmap='Blues')
plt.yticks(np.arange(ldaDFVis.shape[0])+0.5, ldaDFVisNames);
plt.xticks(np.arange(ldaDFVis.shape[1])+0.5, topic_labels);

# flip the y-axis so the texts are in the order we anticipate (Austen first, then Brontë)
plt.gca().invert_yaxis()

# rotate the ticks on the x-axis
plt.xticks(rotation=90)

# add a legend
plt.colorbar(cmap='Blues')
plt.tight_layout()  # fixes margins
plt.show()
```

We can also look at the top words from each topic to get a sense of the semantic
(or syntactic) domain they represent. To look at the terms with the highest LDA
weight in topic `1` we can do the following:

```python
senlda.show_topic(1)
```

And if we want to make a dataFrame:

```python
topicsDict = {}
for topicNum in range(senlda.num_topics):
    topicWords = [w for w, p in senlda.show_topic(topicNum)]
    topicsDict['Topic_{}'.format(topicNum)] = topicWords

wordRanksDF = pandas.DataFrame(topicsDict)
wordRanksDF
```

We can see that several of the topics have the same top words, but there are
definitely differences. We can try and make the topics more distinct by changing
the $\alpha$ and $\eta$ parameters of the model. $\alpha$ controls the sparsity
of document-topic loadings, and $\eta$ controls the sparsity of topic-word
loadings.

We can make a visualization of the distribution of words over any single topic.

```python
topic1_df = pandas.DataFrame(senlda.show_topic(1, topn=50))
plt.figure()
topic1_df.plot.bar(legend = False)
plt.title('Probability Distribution of Words, Topic 1')
plt.show()
```

See how different $\eta$ values can change the shape of the distribution.

```python
senlda1 = gensim.models.ldamodel.LdaModel(corpus=senmm, id2word=dictionary, num_topics=10, eta = 0.00001)
senlda2 = gensim.models.ldamodel.LdaModel(corpus=senmm, id2word=dictionary, num_topics=10, eta = 0.9)
```

```python
topic11_df = pandas.DataFrame(senlda1.show_topic(1, topn=50))
topic21_df = pandas.DataFrame(senlda2.show_topic(1, topn=50))

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.set_size_inches(18, 7)
topic11_df.plot.bar(legend = False, ax = ax1, title = '$\eta$  = 0.00001')
topic21_df.plot.bar(legend = False, ax = ax2, title = '$\eta$  = 0.9')
plt.show()
```
