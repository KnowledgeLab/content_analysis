# Week 4 - Word Embeddings

This week, we build on last week's topic modeling techniques by taking a text
corpus we have developed, specifying an underlying number of dimensions, and
training a model with a neural network auto-encoder (one of Google's word2vec
algorithms) that best describes corpus words in their local linguistic contexts,
and exploring their locations in the resulting space to learn about the
discursive culture that produced them. Documents here are represented as densely
indexed locations in dimensions, rather than sparse mixtures of topics (as in
LDA topic modeling), so that distances between those documents (and words) are
consistently superior, though they require the full vector of dimension loadings
(rather than just a few selected topic loadings) to describe. We will explore
these spaces to understand complex, semantic relationships between words, index
documents with descriptive words, identify the likelihood that a given document
would have been produced by a given vector model, and explore how semantic
categories can help us understand the cultures that produced them.

For this notebook we will be using the following packages

```python
#All these packages need to be installed from pip
import gensim#For word2vec, etc
import requests #For downloading our datasets
import nltk #For stop words and stemmers
nltk.data.path.append('../data/nltk_data')
import numpy as np #For arrays
import pandas #Gives us DataFrames
import matplotlib.pyplot as plt #For graphics
import seaborn #Makes the graphics look nicer
import sklearn.metrics.pairwise #For cosine similarity
import sklearn.manifold #For T-SNE
import sklearn.decomposition #For PCA

#This 'magic' command makes the plots work better
#in the notebook, don't use it outside of a notebook.
#Also you can ignore the warning
%matplotlib inline

import os #For looking through files
import os.path #For managing file paths
```

# Getting our corpora

Instead of downloading our corpora, we have download them in advance; a subset
of the [senate press
releases](https://github.com/lintool/GrimmerSenatePressReleases) are in
`grimmerPressReleases`. We will load them into a DataFrame, but first we need to
define a function to convert directories of text files into DataFrames:

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

Now we can use the function in all the directories in
`data/grimmerPressReleases`

```python
dataDir = '../data/grimmerPressReleases'

senReleasesDF = pandas.DataFrame()

for senatorName in [d for d in os.listdir(dataDir) if d[0] != '.']:
    senPath = os.path.join(dataDir, senatorName)
    senReleasesDF = senReleasesDF.append(loadDir(senPath, senatorName), ignore_index = True)

senReleasesDF[:100:10]
```

We also want to remove stop words and stem. Tokenizing requires two steps.
Word2Vec needs to retain the sentence structure so as to capture a "continuous
bag of words (CBOW)" and all of the skip-grams within a word window. The
algorithm tries to preserve the distances induced by one of these two local
structures. This is very different from clustering and LDA topic modeling which
extract unordered words alone. As such, tokenizing is slightly more involved.

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

#initialize our stemmer and our stop words
stop_words_nltk = nltk.corpus.stopwords.words('english')
snowball = nltk.stem.snowball.SnowballStemmer('english')
wordnet = nltk.stem.WordNetLemmatizer()
```

```python
#Apply our functions, notice each row is a list of lists now
senReleasesDF['tokenized_sents'] = senReleasesDF['text'].apply(lambda x: [nltk.word_tokenize(s) for s in nltk.sent_tokenize(x)])
senReleasesDF['normalized_sents'] = senReleasesDF['tokenized_sents'].apply(lambda x: [normlizeTokens(s, stopwordLst = stop_words_nltk, stemmer = None) for s in x])

senReleasesDF[:100:10]
```

# Word2Vec

We will be using the gensim implementation of [Word2Vec](https://radimrehurek.co
m/gensim/models/word2vec.html#gensim.models.word2vec.Word2Vec).

To load our data our data we give all the sentences to the trainer:

```python
senReleasesW2V = gensim.models.word2vec.Word2Vec(senReleasesDF['normalized_sents'].sum())
```

Inside the word2vec object the words each have a vector. To access the vector
directly, use the square braces (`__getitem__`) method:

```python
senReleasesW2V['president'][:10] #Shortening because it's very large
```

If you want the full matrix, `syn0` stores all the vectors:

```python
senReleasesW2V.wv.syn0
```

Then, `index2word` lets you translate from the matrix to words

```python
senReleasesW2V.wv.index2word[10]
```

Now we can look at a few things that come from the word vectors. The first is to
find similar vectors (cosine similarity):

```python
senReleasesW2V.most_similar('president')
```

```python
senReleasesW2V.most_similar('war')
```

Find which word least matches the others within a word set (cosine similarity):

```python
senReleasesW2V.doesnt_match(['administration', 'administrations', 'presidents', 'president', 'washington'])
```

Find which word best matches the result of a semantic *equation* (here, we seek
the words whose vectors best fit the missing entry from the equation: **X + Y -
Z = _**.

```python
senReleasesW2V.most_similar(positive=['clinton', 'republican'], negative = ['democrat'])
```

Here we see that **Clinton + Republican - Democrat = Bush**. In other words, in
this dataset, **Clinton** is to **Democrat** as **Bush** is to **Republican**.
Whoah!

We can also save the vectors for later use:

```python
senReleasesW2V.save("senpressreleasesWORD2Vec")
```

We can also use dimension reduction to visulize the vectors. We will start by
selecting a subset we want to plot. Let's look at the top words from the set:

```python
numWords = 50
targetWords = senReleasesW2V.wv.index2word[:numWords]
```

We can then extract their vectors and create our own smaller matrix that
preserved the distances from the original:

```python
wordsSubMatrix = []
for word in targetWords:
    wordsSubMatrix.append(senReleasesW2V[word])
wordsSubMatrix = np.array(wordsSubMatrix)
wordsSubMatrix
```

Then we can use PCA to reduce the dimesions (e.g., to 50), and T-SNE to project
them down to the two we will visualize. We note that this is nondeterministic
process, and so you can repeat and achieve alternative
projectsions/visualizations of the words:

```python
pcaWords = sklearn.decomposition.PCA(n_components = 50).fit(wordsSubMatrix)
reducedPCA_data = pcaWords.transform(wordsSubMatrix)
#T-SNE is theoretically better, but you should experiment
tsneWords = sklearn.manifold.TSNE(n_components = 2).fit_transform(reducedPCA_data)
```

We now can plot the points

```python
fig = plt.figure(figsize = (10,6))
ax = fig.add_subplot(111)
ax.set_frame_on(False)
plt.scatter(tsneWords[:, 0], tsneWords[:, 1], alpha = 0)#Making the points invisible 
for i, word in enumerate(targetWords):
    ax.annotate(word, (tsneWords[:, 0][i],tsneWords[:, 1][i]), size =  20 * (numWords - i) / numWords)
plt.xticks(())
plt.yticks(())
plt.show()
```

My visualization above puts ``said`` next to ``congress`` and ``bill`` near
``act``. ``health`` is beside ``care`` and ``national`` abuts ``security``.

# Doc2Vec

Instead of just looking at just how words embed within in the space, we can look
at how the different documents relate to each other within the space. First lets
load our data--abstracts of most U.S. physics papers from the 1950s.

```python
apsDF = pandas.read_csv('../data/APSabstracts1950s.csv', index_col = 0)
apsDF[:10]
```

We will load these as documents into Word2Vec, but first we need to normalize
and pick some tags

```python
keywords = ['photomagnetoelectric', 'quantum', 'boltzmann', 'proton', 'positron', 'feynman', 'classical', 'relativity']
```

```python
apsDF['tokenized_words'] = apsDF['abstract'].apply(lambda x: nltk.word_tokenize(x))
apsDF['normalized_words'] = apsDF['tokenized_words'].apply(lambda x: normlizeTokens(x, stopwordLst = stop_words_nltk))
```

```python
taggedDocs = []
for index, row in apsDF.iterrows():
    #Just doing a simple keyword assignment
    docKeywords = [s for s in keywords if s in row['normalized_words']]
    docKeywords.append(row['copyrightYear'])
    docKeywords.append(row['doi']) #This lets us extract individual documnets since doi's are unique
    taggedDocs.append(gensim.models.doc2vec.LabeledSentence(words = row['normalized_words'], tags = docKeywords))
apsDF['TaggedAbstracts'] = taggedDocs
```

Now we can train a Doc2Vec model:

```python
apsD2V = gensim.models.doc2vec.Doc2Vec(apsDF['TaggedAbstracts'], size = 100) #Limiting to 100 dimensions
```

We can get vectors for the tags/documents, just as we did with words. Documents
are actually the centroids (high dimensional average points) of their words.

```python
apsD2V.docvecs[1952]
```

The words can still be accessed in the same way:

```python
apsD2V['atom']
```

We can still use the ``most_similar`` command to perform simple semantic
equations:

```python
apsD2V.most_similar(positive = ['atom','electrons'], negative = ['electron'], topn = 1)
```

This is interesting. **Electron** is to **electrons** as **atom** is to
**atoms**. Another way to understand this, developed below is: **electrons -
electron** induces a singular to plural dimension, so when we subtract
**electron** from **atom** and add **electrons**, we get **atoms**!

```python
apsD2V.most_similar(positive = ['einstein','law'], negative = ['equation'], topn = 1)
```

In other words **Einstein** minus **equation** plus **law** equals
**Meissner**--Walthur Meissner studied mechanical engineering and physics ...
and was more likely to produce a "law" than a "equation", like the Meissner
effect, the damping of the magnetic field in superconductors. If we built our
word-embedding with a bigger corpus like the entire arXiv, a massive repository
of physics preprints, we would see many more such relationships like **gravity -
Newton + Einstein = relativity**.

We can also compute all of these *by hand*--explicitly wth vector algebra:

```python
sklearn.metrics.pairwise.cosine_similarity(apsD2V['electron'].reshape(1,-1), apsD2V['positron'].reshape(1,-1))
#We reorient the vectors with .reshape(1, -1) so that they can be computed without a warning in sklearn
```

In the doc2vec model, the documents have vectors just as the words do, so that
we can compare documents with each other and also with words (similar to how a
search engine locates a webpage with a query). First, we will calculate the
distance between a word and documents in the dataset:

```python
apsD2V.docvecs.most_similar([ apsD2V['electron'] ], topn=5 )
```

If we search for the first of these on the web (these are doi codes), we find
the following...a pretty good match:

```python
from IPython.display import Image
Image("../data/PhysRev.98.875.jpg", width=1000, height=1000)
```

Now let's go the other way around and find words most similar to this document:

```python
apsD2V.most_similar( [ apsD2V.docvecs['10.1103/PhysRev.98.875'] ], topn=5) 
```

We can even look for documents most like a query composed of multiple words:

```python
apsD2V.docvecs.most_similar([ apsD2V['electron']+apsD2V['positron']+apsD2V['neutron']], topn=5 )
```

Now let's plot some words and documents against one another with a heatmap:

```python
heatmapMatrix = []
for tagOuter in keywords:
    column = []
    tagVec = apsD2V.docvecs[tagOuter].reshape(1, -1)
    for tagInner in keywords:
        column.append(sklearn.metrics.pairwise.cosine_similarity(tagVec, apsD2V.docvecs[tagInner].reshape(1, -1))[0][0])
    heatmapMatrix.append(column)
heatmapMatrix = np.array(heatmapMatrix)
```

```python
fig, ax = plt.subplots()
hmap = ax.pcolor(heatmapMatrix, cmap='terrain')
cbar = plt.colorbar(hmap)

cbar.set_label('cosine similarity', rotation=270)
a = ax.set_xticks(np.arange(heatmapMatrix.shape[1]) + 0.5, minor=False)
a = ax.set_yticks(np.arange(heatmapMatrix.shape[0]) + 0.5, minor=False)

a = ax.set_xticklabels(keywords, minor=False, rotation=270)
a = ax.set_yticklabels(keywords, minor=False)
```

Now let's look at a heatmap of similarities between the first ten documents in
the corpus:

```python
targetDocs = apsDF['doi'][:10]

heatmapMatrixD = []

for tagOuter in targetDocs:
    column = []
    tagVec = apsD2V.docvecs[tagOuter].reshape(1, -1)
    for tagInner in targetDocs:
        column.append(sklearn.metrics.pairwise.cosine_similarity(tagVec, apsD2V.docvecs[tagInner].reshape(1, -1))[0][0])
    heatmapMatrixD.append(column)
heatmapMatrixD = np.array(heatmapMatrixD)
```

```python
fig, ax = plt.subplots()
hmap = ax.pcolor(heatmapMatrixD, cmap='terrain')
cbar = plt.colorbar(hmap)

cbar.set_label('cosine similarity', rotation=270)
a = ax.set_xticks(np.arange(heatmapMatrixD.shape[1]) + 0.5, minor=False)
a = ax.set_yticks(np.arange(heatmapMatrixD.shape[0]) + 0.5, minor=False)

a = ax.set_xticklabels(targetDocs, minor=False, rotation=270)
a = ax.set_yticklabels(targetDocs, minor=False)
```

Now let's look at a heatmap of similarities between the first ten documents and
our keywords:

```python
heatmapMatrixC = []

for tagOuter in targetDocs:
    column = []
    tagVec = apsD2V.docvecs[tagOuter].reshape(1, -1)
    for tagInner in keywords:
        column.append(sklearn.metrics.pairwise.cosine_similarity(tagVec, apsD2V.docvecs[tagInner].reshape(1, -1))[0][0])
    heatmapMatrixC.append(column)
heatmapMatrixC = np.array(heatmapMatrixC)
```

```python
fig, ax = plt.subplots()
hmap = ax.pcolor(heatmapMatrixC, cmap='terrain')
cbar = plt.colorbar(hmap)

cbar.set_label('cosine similarity', rotation=270)
a = ax.set_xticks(np.arange(heatmapMatrixC.shape[1]) + 0.5, minor=False)
a = ax.set_yticks(np.arange(heatmapMatrixC.shape[0]) + 0.5, minor=False)

a = ax.set_xticklabels(keywords, minor=False, rotation=270)
a = ax.set_yticklabels(targetDocs, minor=False)
```

We will save the model in case we would like to use it again.

```python
apsD2V.save('apsW2V')
```

We can later load it:

```python
#apsD2V = gensim.models.word2vec.Word2Vec.load('data/apsW2V')
```

# Linguistic Change

Below is code that aligns the dimensions of multiple embeddings arrayed over
time or some other dimension and allow identification of semantic chanage as the
word vectors change their loadings for focal words. This code comes from the
approach piloted at Stanford by William Hamilton, Daniel Jurafsky and Jure
Lescovec [here](https://arxiv.org/pdf/1605.09096.pdf).

```python
def smart_procrustes_align_gensim(base_embed, other_embed, words=None):
    """Procrustes align two gensim word2vec models (to allow for comparison between same word across models).
    Code ported from HistWords <https://github.com/williamleif/histwords> by William Hamilton <wleif@stanford.edu>.
    (With help from William. Thank you!)
    First, intersect the vocabularies (see `intersection_align_gensim` documentation).
    Then do the alignment on the other_embed model.
    Replace the other_embed model's syn0 and syn0norm numpy matrices with the aligned version.
    Return other_embed.
    If `words` is set, intersect the two models' vocabulary with the vocabulary in words (see `intersection_align_gensim` documentation).
    """

    # make sure vocabulary and indices are aligned
    in_base_embed, in_other_embed = intersection_align_gensim(base_embed, other_embed, words=words)

    # get the embedding matrices
    base_vecs = in_base_embed.syn0norm
    other_vecs = in_other_embed.syn0norm

    # just a matrix dot product with numpy
    m = other_vecs.T.dot(base_vecs) 
    # SVD method from numpy
    u, _, v = np.linalg.svd(m)
    # another matrix operation
    ortho = u.dot(v) 
    # Replace original array with modified one
    # i.e. multiplying the embedding matrix (syn0norm)by "ortho"
    other_embed.syn0norm = other_embed.syn0 = (other_embed.syn0norm).dot(ortho)
    return other_embed
    
def intersection_align_gensim(m1,m2, words=None):
    """
    Intersect two gensim word2vec models, m1 and m2.
    Only the shared vocabulary between them is kept.
    If 'words' is set (as list or set), then the vocabulary is intersected with this list as well.
    Indices are re-organized from 0..N in order of descending frequency (=sum of counts from both m1 and m2).
    These indices correspond to the new syn0 and syn0norm objects in both gensim models:
        -- so that Row 0 of m1.syn0 will be for the same word as Row 0 of m2.syn0
        -- you can find the index of any word on the .index2word list: model.index2word.index(word) => 2
    The .vocab dictionary is also updated for each model, preserving the count but updating the index.
    """

    # Get the vocab for each model
    vocab_m1 = set(m1.vocab.keys())
    vocab_m2 = set(m2.vocab.keys())

    # Find the common vocabulary
    common_vocab = vocab_m1&vocab_m2
    if words: common_vocab&=set(words)

    # If no alignment necessary because vocab is identical...
    if not vocab_m1-common_vocab and not vocab_m2-common_vocab:
        return (m1,m2)

    # Otherwise sort by frequency (summed for both)
    common_vocab = list(common_vocab)
    common_vocab.sort(key=lambda w: m1.vocab[w].count + m2.vocab[w].count,reverse=True)

    # Then for each model...
    for m in [m1,m2]:
        # Replace old syn0norm array with new one (with common vocab)
        indices = [m.vocab[w].index for w in common_vocab]
        old_arr = m.syn0norm
        new_arr = np.array([old_arr[index] for index in indices])
        m.syn0norm = m.syn0 = new_arr

        # Replace old vocab dictionary with new one (with common vocab)
        # and old index2word with new one
        m.index2word = common_vocab
        old_vocab = m.vocab
        new_vocab = {}
        for new_index,word in enumerate(common_vocab):
            old_vocab_obj=old_vocab[word]
            new_vocab[word] = gensim.models.word2vec.Vocab(index=new_index, count=old_vocab_obj.count)
        m.vocab = new_vocab

    return (m1,m2)
```

In order to explore this, let's get some data that follows a time trend. We'll
look at conference proceedings from the American Society for Clinical
Oncologists.

```python
asco = "../data/ASCO_abstracts.csv"

ascoDF = pandas.read_table(asco, sep=',', doublequote=True, escapechar=None, quotechar='"', quoting=0, skipinitialspace=False, lineterminator=None, header='infer', index_col=None, names=None, prefix=None, skiprows=None, skipfooter=None, skip_footer=0, na_values=None, true_values=None, false_values=None, delimiter=None, converters=None, dtype=None, usecols=None, engine=None, delim_whitespace=False, as_recarray=False, na_filter=True, compact_ints=False, use_unsigned=False, low_memory=True, buffer_lines=None, warn_bad_lines=True, error_bad_lines=True, keep_default_na=True, thousands=None, comment=None, decimal='.', parse_dates=False, keep_date_col=False, dayfirst=False, date_parser=None, memory_map=False, float_precision=None, nrows=None, iterator=False, chunksize=None, verbose=False, encoding="ISO-8859-1", squeeze=False, mangle_dupe_cols=True, tupleize_cols=False, infer_datetime_format=False, skip_blank_lines=True)
asco_year_id = {1995: "28", 1996:"29", 1997:"30", 1998:"31", 1999:"17",2001:"10",2002:"16",2003:"23",2004:"26",2005:"34",2006:"40",2007:"47",2008:"55",2009:"65",2010:"74",2011:"102"}
asco95DF = ascoDF.loc[ascoDF['MeetingID'] == asco_year_id[1995]].copy()
asco96DF = ascoDF.loc[ascoDF['MeetingID'] == asco_year_id[1996]].copy()
asco97DF = ascoDF.loc[ascoDF['MeetingID'] == asco_year_id[1997]].copy()
asco98DF = ascoDF.loc[ascoDF['MeetingID'] == asco_year_id[1998]].copy()
asco99DF = ascoDF.loc[ascoDF['MeetingID'] == asco_year_id[1999]].copy()
asco01DF = ascoDF.loc[ascoDF['MeetingID'] == asco_year_id[2001]].copy()
asco02DF = ascoDF.loc[ascoDF['MeetingID'] == asco_year_id[2002]].copy()
asco03DF = ascoDF.loc[ascoDF['MeetingID'] == asco_year_id[2003]].copy()
asco04DF = ascoDF.loc[ascoDF['MeetingID'] == asco_year_id[2004]].copy()
asco05DF = ascoDF.loc[ascoDF['MeetingID'] == asco_year_id[2005]].copy()
asco06DF = ascoDF.loc[ascoDF['MeetingID'] == asco_year_id[2006]].copy()
asco07DF = ascoDF.loc[ascoDF['MeetingID'] == asco_year_id[2007]].copy()
asco08DF = ascoDF.loc[ascoDF['MeetingID'] == asco_year_id[2008]].copy()
asco09DF = ascoDF.loc[ascoDF['MeetingID'] == asco_year_id[2009]].copy()
asco10DF = ascoDF.loc[ascoDF['MeetingID'] == asco_year_id[2010]].copy()
```

```python
#Do for all others...
asco96DF['tokenized_sents'] = asco96DF['Body'].apply(lambda x: [nltk.word_tokenize(s) for s in nltk.sent_tokenize(x)])
asco96DF['normalized_sents'] = asco96DF['tokenized_sents'].apply(lambda x: [normlizeTokens(s, stopwordLst = stop_words_nltk, stemmer = None) for s in x])
```

```python
#Do for all others...
#asco95W2V = gensim.models.word2vec.Word2Vec(asco95DF['normalized_sents'].sum())
asco96W2V = gensim.models.word2vec.Word2Vec(asco96DF['normalized_sents'].sum())
```

Do a matrix for one word, e.g., "breast" that looks for changes over the whole
period
Then look for the word that changes the most over the whole period (sum of
dimension changes)

# Projection

We can also project word vectors to an arbitray semantic dimension. To
demonstrate this possibility, let's first load a model trained with New York
Times news articles.

```python
nytimes_model = gensim.models.KeyedVectors.load_word2vec_format('../data/nytimes_cbow.reduced.txt')
```

First we can visualize with dimension reduction

```python
#words to create dimensions
tnytTargetWords = ['man','him','he', 'woman', 'her', 'she', 'black','blacks','African', 'white', 'whites', 'Caucasian', 'rich', 'richer', 'richest', 'expensive', 'wealthy', 'poor', 'poorer', 'poorest', 'cheap', 'inexpensive']
#words we will be mapping
tnytTargetWords += ["doctor","lawyer","plumber","scientist","hairdresser", "nanny","carpenter","entrepreneur","musician","writer", "banker","poet","nurse", "steak", "bacon", "croissant", "cheesecake", "salad", "cheeseburger", "vegetables", "beer", "wine", "pastry", "basketball", "baseball", "boxing", "softball", "volleyball", "tennis", "golf", "hockey", "soccer"]


wordsSubMatrix = []
for word in tnytTargetWords:
    wordsSubMatrix.append(nytimes_model[word])
wordsSubMatrix = np.array(wordsSubMatrix)
wordsSubMatrix
```

```python
pcaWordsNYT = sklearn.decomposition.PCA(n_components = 50).fit(wordsSubMatrix)
reducedPCA_dataNYT = pcaWordsNYT.transform(wordsSubMatrix)
#T-SNE is theoretically better, but you should experiment
tsneWordsNYT = sklearn.manifold.TSNE(n_components = 2).fit_transform(reducedPCA_dataNYT)
```

```python
fig = plt.figure(figsize = (10,6))
ax = fig.add_subplot(111)
ax.set_frame_on(False)
plt.scatter(tsneWordsNYT[:, 0], tsneWordsNYT[:, 1], alpha = 0) #Making the points invisible
for i, word in enumerate(tnytTargetWords):
    ax.annotate(word, (tsneWordsNYT[:, 0][i],tsneWordsNYT[:, 1][i]), size =  20 * (len(tnytTargetWords) - i) / len(tnytTargetWords))
plt.xticks(())
plt.yticks(())
plt.show()
```

Define some convenient functions for getting dimensions.

```python
def normalize(vector):
    normalized_vector = vector / np.linalg.norm(vector)
    return normalized_vector

def dimension(model, positives, negatives):
    diff = sum([normalize(model[x]) for x in positives]) - sum([normalize(model[y]) for y in negatives])
    return diff
```

Let's calculate three dimensions: gender, race, and class.

```python
Gender = dimension(nytimes_model, ['man','him','he'], ['woman', 'her', 'she'])
Race = dimension(nytimes_model, ['black','blacks','African'], ['white', 'whites', 'Caucasian'])
Class = dimension(nytimes_model, ['rich', 'richer', 'richest', 'expensive', 'wealthy'], ['poor', 'poorer', 'poorest', 'cheap', 'inexpensive'])
```

Here we have some words.

```python
Occupations = ["doctor","lawyer","plumber","scientist","hairdresser", "nanny","carpenter","entrepreneur","musician","writer", "banker","poet","nurse"]

Foods = ["steak", "bacon", "croissant", "cheesecake", "salad", "cheeseburger", "vegetables", "beer", "wine", "pastry"]

Sports  = ["basketball", "baseball", "boxing", "softball", "volleyball", "tennis", "golf", "hockey", "soccer"]
```

Define a function to project words in a word list to each of the three
dimensions.

```python
def makeDF(model, word_list):
    g = []
    r = []
    c = []
    for word in word_list:
        g.append(sklearn.metrics.pairwise.cosine_similarity(nytimes_model[word].reshape(1,-1), Gender.reshape(1,-1))[0][0])
        r.append(sklearn.metrics.pairwise.cosine_similarity(nytimes_model[word].reshape(1,-1), Race.reshape(1,-1))[0][0])
        c.append(sklearn.metrics.pairwise.cosine_similarity(nytimes_model[word].reshape(1,-1), Class.reshape(1,-1))[0][0])
    df = pandas.DataFrame({'gender': g, 'race': r, 'class': c}, index = word_list)
    return df
```

Get the projections.

```python
OCCdf = makeDF(nytimes_model, Occupations) 
Fooddf = makeDF(nytimes_model, Foods)
Sportsdf = makeDF(nytimes_model, Sports)
```

Define some useful functions for plotting.

```python
def Coloring(Series):
    x = Series.values
    y = x-x.min()
    z = y/y.max()
    c = list(plt.cm.rainbow(z))
    return c

def PlotDimension(ax,df, dim):
    ax.set_frame_on(False)
    ax.set_title(dim, fontsize = 20)
    colors = Coloring(df[dim])
    for i, word in enumerate(df.index):
        ax.annotate(word, (0, df[dim][i]), color = colors[i], alpha = 0.6, fontsize = 12)
    MaxY = df[dim].max()
    MinY = df[dim].min()
    plt.ylim(MinY,MaxY)
    plt.yticks(())
    plt.xticks(())
```

Plot the occupational words in each of the three dimensions.

```python
fig = plt.figure(figsize = (12,4))
ax1 = fig.add_subplot(131)
PlotDimension(ax1, OCCdf, 'gender')
ax2 = fig.add_subplot(132)
PlotDimension(ax2, OCCdf, 'race')
ax3 = fig.add_subplot(133)
PlotDimension(ax3, OCCdf, 'class')
plt.show()
```

Foods:

```python
fig = plt.figure(figsize = (12,4))
ax1 = fig.add_subplot(131)
PlotDimension(ax1, Fooddf, 'gender')
ax2 = fig.add_subplot(132)
PlotDimension(ax2, Fooddf, 'race')
ax3 = fig.add_subplot(133)
PlotDimension(ax3, Fooddf, 'class')
plt.show()
```

Sports:

```python
fig = plt.figure(figsize = (12,4))
ax1 = fig.add_subplot(131)
PlotDimension(ax1, Sportsdf, 'gender')
ax2 = fig.add_subplot(132)
PlotDimension(ax2, Sportsdf, 'race')
ax3 = fig.add_subplot(133)
PlotDimension(ax3, Sportsdf, 'class')
plt.show()
```
