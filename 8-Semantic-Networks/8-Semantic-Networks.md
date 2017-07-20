# Week 8 - Semantic Networks

This week, we explore the representation and analysis of semantic networks. A
word or document network is an unsupervized representation of text akin to a
clustering or an embedding, but semantic networks can also be defined using
semantic or syntactic information derived from methods we have used earlier in
the quarter. For example, we can define links between words as a function of
their co-presence within a document, chapter, paragraph, sentence, noun phrase
or continuous bag of words. We can also define links as a function of words that
rely on one another within a directed dependency parse, or links between
extracted Subjects, Verbs and Objects, or nouns and the adjectives that modify
them (or verbs and the adverbs that modify *them*). Rendering words linked as a
network or discrete topology allows us to take advantage of the wide range of
metrics and models developed for network analysis. These include measurement of
network centrality, density and modularity, "block modeling" structurally
equivalent relationships, andsophisticated graphical renderings of networks or
network partitions that allow us to visually interrogate their structure and
complexity.

For this notebook we will use the following packages:

```python
#All these packages need to be installed from pip

import nltk #For POS tagging
import sklearn #For generating some matrices
import pandas #For DataFrames
import numpy as np #For arrays
import matplotlib.pyplot as plt #For plotting
import seaborn #MAkes the plots look nice
import IPython.display #For displaying images

#This 'magic' command makes the plots work better
#in the notebook, don't use it outside of a notebook.
#Also you can ignore the warning
%matplotlib inline

#This has some C components and installs are OS specific
#See http://igraph.org/python/ for details
import igraph as ig #For the networks


import pickle #if you want to save layouts
```

# Note about `igraph`

Getting `igraph` to make plots in jupyter can be tricky, if you are having
issues try these steps:

+ install `cairocffi`, remove `pycairo` if present
+ install `igraph` version [`0.7.1`](https://github.com/igraph/python-
igraph/tree/release-0.7) from the github repo, **Do not** use the pip version,
there is a [known issue](https://github.com/igraph/python-igraph/issues/89) with
it

If the resulting plots have garbled labels, save them as svg files `target =
'filename.svg'` and the labels will be fine.


We will primarily be dealing with graphs in this notebook, so lets first go over
how to use them.

To start with lets create an undirected graph:

```python
g = ig.Graph()
g
```

We can add vertices. These are all numbered following Python convention,
starting at 0 and so this will add vertices `0`, `1` and `2`.

```python
g.add_vertices(3)
```

Now we have 3 vertices:

```python
g.vcount()
```

Or if we want to get more information about the graph:

```python
print(g.summary())
```

We can give vertices names, or even give the graph a name:

```python
g.vs['name'] = ['a', 'b', 'c'] #vs stands for VertexSeq(uence)
```

Now we can call upon the vertices by name:

```python
g.vs.find('a')
```

Still pretty boring though...

Lets add a couple of edges. Notice that we can use names or ids:

```python
g.add_edges([(0,1), (1,2), ('a', 'c')])
print(g.summary())
```

Note how the summary has changed.

We can also give the edges properties. Instead of names we can give them
weights:

```python
print("Before weighted", g.is_weighted())
g.es['a', 'b']['weight'] = 4
g.es['a', 'c']['weight'] = 10
print("After weighted", g.is_weighted())
```

Let's visualize it now:

```python
ig.plot(g)
```

Very exciting :-).

There are a great many things to do with the graph once we have created it, some
of which we will explore today.

First lets load our data, the Grimmer corpus:

```python
senReleasesDF = pandas.read_csv('data/senReleasesTraining.csv', index_col = 0)
senReleasesDF[:5]
```

We will be extracting sentences, as well as tokenizing and stemming. (You should
be able to do this in your sleep now):

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

For now we will not be dropping any stop words:

```python
senReleasesDF['tokenized_sents'] = senReleasesDF['text'].apply(lambda x: [nltk.word_tokenize(s) for s in nltk.sent_tokenize(x)])
senReleasesDF['normalized_sents'] = senReleasesDF['tokenized_sents'].apply(lambda x: [normlizeTokens(s, stopwordLst = None, stemmer = snowball) for s in x])

senReleasesDF[:5]
```

Let's start by looking at words co-occurring in the same sentences:

```python
def wordCooccurrence(sentences, makeMatrix = False):
    words = set()
    for sent in sentences:
        words |= set(sent)
    wordLst = list(words)
    wordIndices = {w: i for i, w in enumerate(wordLst)}
    wordCoCounts = {}
    #consider a sparse matrix if memory becomes an issue
    coOcMat = np.zeros((len(wordIndices), len(wordIndices)))
    for sent in sentences:
        for i, word1 in enumerate(sent):
            word1Index = wordIndices[word1]
            for word2 in sent[i + 1:]:
                coOcMat[word1Index][wordIndices[word2]] += 1
    if makeMatrix:
        return coOcMat, wordLst
    else:
        coOcMat = coOcMat.T + coOcMat
        edges = list(zip(*np.where(coOcMat)))
        weights = coOcMat[np.where(coOcMat)]
        g = ig.Graph( n = len(wordLst),
            edges = edges,
            vertex_attrs = {'name' : wordLst, 'label' : wordLst},
            edge_attrs = {'weight' : weights}
                    )
        return g
```

Build a graph based on word cooccurence in the first 100 press releases.

```python
g = wordCooccurrence(senReleasesDF['normalized_sents'][:100].sum())
```

Total number of vertices:

```python
g.vcount()
```

Total number of edges:

```python
g.ecount()
```

A part of the adjacency matrix of cleaned word by press releases:

```python
g.get_adjacency()[:10, :10]
```

We can save the graph and read it later:

```python
g.write_gml('data/Obama_words.gml')
```

Or, we can build graphs starting with a two-mode network. Let's again use the
document-word frequency matrix that we used in week 3.

```python
def tokenize(text):
    tokenlist = nltk.word_tokenize(text)
    normalized = normlizeTokens(tokenlist, stemmer = snowball)
    return normalized
```

```python
senVectorizer = sklearn.feature_extraction.text.CountVectorizer(stop_words=stop_words_nltk, tokenizer = tokenize)
senVects = senVectorizer.fit_transform(senReleasesDF['text'][:100])
```

```python
senVects.shape
```

We need to turn the matrix into a list as our input.

```python
INP = senVects.todense().tolist()
```

```python
g = ig.Graph.Incidence(INP, multiple=True)
```

We can check whether it is a bipartite network.

```python
g.is_bipartite()
```

Give colors and shapes to documents and words:

```python
g.vs['color']  = ['red']*senVects.shape[0] + ['blue']*senVects.shape[1]
g.vs['shape']  = ['square']*senVects.shape[0] + ['circle']*senVects.shape[1]
```

A very popular layout algorithm for visualizing graphs is the Fruchterman-
Reingold Algorithm, which uses a physical metaphor for lay-out. Nodes repel one
another, and edges draw connected elements together like springs. The algorithm
attempts to minimize the energy in such a system. For a large graph, however,
the algorithm is computational demanding. The commented code gives you a layout
and you can save it as a pickle. Let's then load a layout.

```python
#These take a long time
#layout = g.layout_fruchterman_reingold()
#pickle.dump(layout, open( "layout.pkl", "wb" ) )
#layout = pickle.load(open('layout.pkl','rb'))
```

Then, let's plot the bipartite network:

```python
ig.plot(g, layout = None, vertex_size = 5)
```

A faster algorithm for large networks is the Large Graph Layout algorithm. If we
want even faster computation and tunable visualizations, check out
[Pajek](http://mrvar.fdv.uni-lj.si/pajek/) or [gephi](https://gephi.org/).

```python
layout = g.layout_lgl()
ig.plot(g, layout = layout, vertex_size = 5)
```

A two-mode network can be easily transformed into two one-mode network, enabling
words to be connected to other words via the number of documents that share
them, or documents to be connected to other documents via the words they share.

```python
gDoc, gWord = g.bipartite_projection()
```

Let's first take a look at the document-to-document network:

```python
gDoc.summary()
```

Let's do a visualization. It is not surprising that almost every document is
connected to every one else. We can use edge weight to distinguish distance
(modeled as attraction) between the nodes.

```python
layout = gDoc.layout_fruchterman_reingold(weights = gDoc.es['weight'])
ig.plot(gDoc, layout = layout, vertex_shape = 'circle')
```

Another way to visualize the graph is to binarize the adjacency matrix with some
cutoff values for edge weight. Let's use the medianw weight as our cutoff
threshhold.

```python
median = np.median(gDoc.es['weight'])
medEdges = gDoc.es.select(lambda x: x['weight'] > median)
g1_d = gDoc.subgraph_edges(medEdges)
```

```python
g1_d.summary()
```

```python
layout = g1_d.layout_fruchterman_reingold()
ig.plot(g1_d, layout = layout, vertex_shape = 'circle')
```

Now let's turn it around and look at the word-to-word network (via documents).
Instead of working directly with `gWord`, let's start with matrix multiplication
which gives us more freedom to define edge weight. First, let's reduce the
number of words to a manageable size.

```python
senVectorizer = sklearn.feature_extraction.text.CountVectorizer(stop_words=stop_words_nltk, tokenizer = tokenize, max_features = 200)
senVects = senVectorizer.fit_transform(senReleasesDF['text'][:100])
wordsLst = senVectorizer.get_feature_names()
senVects.shape
```

Let's define the weight of an edge to the be the number of document coocurrences
of two words divided by the square root of the product of their marginal
frequencies.

```python
A = senVects.todense() #Get the doc-word frequency matrix.
M = np.sum(A, axis = 0) #The marginal frequencies are simplies the column sums.
W = A.T.dot(A)/np.sqrt(M.T.dot(M)) #Get out weight matrix.
np.fill_diagonal(W,0) #Set the diagonal to zero.
```

```python
G = ig.Graph.Weighted_Adjacency(W.tolist(), mode=ig.ADJ_UNDIRECTED)
G.vs['name'] = wordsLst #Names are what you reference
G.vs['label'] = wordsLst #Label are what is displayed
```

Now, let's visualize this:

```python
layout = G.layout_fruchterman_reingold()
```

```python
ig.plot(G, layout = layout, vertex_size = 0, edge_width = 0.1)
```

Even better, we can input the weight matrix with some cutoff value. We use 4 for
now as it is useful for demonstration, but you can explore:

```python
cutoff = 4
G_b = G.subgraph_edges(G.es.select(lambda x: x['weight'] > cutoff))
```

```python
G_b = G.subgraph_edges(G.es.select(weight_gt = cutoff))
```

```python
layout = G_b.layout_fruchterman_reingold()
```

```python
ig.plot(G_b, layout = layout, vertex_size = 0, edge_width = 0.2)
```

We can continue to trim globally or locally, to investigate the structure of
words around a core word of interest.

#Can we do an example of this...visualize just the neighborhood of a
word...e.g., the word and all words most closely connected and their close
connections (friend of a friend)?

## <span style="color:red">*Your Turn*</span>

<span style="color:red">Construct cells immediately below this that render some
reasonable networks to meaningfully characterize the structure of words and
documents (or subdocuments like chapters or paragraphs) from your corpus. Also
create some smaller, word neighborhood graphs. What are useful filters and
thresholds and what semantic structures do they reveal that given insight into
the social world and social game inscribed in your corpus?

Alternatively, there are some more informative networks statistics.

Lets begin with measures of centrality. The concept of centrality is that some
nodes (words or documents) are more *central* to the network than others. There
are many distinct and opposing operationalizations of this concept, however. One
important concept is *betweenness* centrality, which distinguishes nodes that
require the most shortest pathways between all other nodes in the network.
Semantically, words with a high betweenness centrality may link distinctive
domains, rather than being "central" to any one.

```python
G_b.vs[np.argmax(G_b.betweenness())]
```

We can color and size the nodes by betweenness centrality:

```python
pal = ig.GradientPalette("red", "blue", G_b.vcount())
G_b.vs['color'] = [pal.get(int(v)) for v in np.argsort(G_b.betweenness())]
ig.plot(G_b, layout = layout, edge_width = 0.2)
```

The distrubution of betweenness centrality is:

```python
plt.hist(G_b.betweenness())
plt.show()
```

Or if we set a max of 500

```python
plt.hist([min(b, 500) for b in G_b.betweenness()])
plt.show()
```

This is an exponential distrubution.

Another way to visualize the graph involes the use of label sizes to represent
betweenness centrality and edge widths to represent edge weight:

```python
edge_width = (np.array(G_b.es['weight'])*0.1).tolist()
size = (5 + np.sqrt(np.array(G_b.betweenness()))).tolist()
ig.plot(G_b, layout = layout, vertex_size = 0, vertex_label_size = size, edge_width = edge_width)
```

Here it appears that "health"/"drug", "torture"/"Iraq", and "loan"/"lend" are
key concepts that connect others in the broader network. This is interesting in
that they seem to be a domain-specific rather than linking words like "require"
and "govern".

What are the top ten words in terms of betweenness?

```python
sorted(zip(G_b.vs['name'], G_b.betweenness()), key = lambda x: x[1], reverse = True)[:10]
```

What are words further down (the lowest all have centralities of 0):

```python
sorted(zip(G_b.vs['name'], G_b.betweenness()), key = lambda x: x[1], reverse = True)[140:150]
```

Alternatively, we can look at degree centrality, which is simply the number of
connections possessed by each node.

```python
G_b.vs[np.argmax(G_b.degree())]
```

```python
pal = ig.GradientPalette("red", "blue", max(G_b.degree()) + 1)
G_b.vs['color'] = [pal.get(v) for v in G_b.degree()]
ig.plot(G_b, layout = layout, edge_width = 0.2)
```

The top 10 words by degree are:

```python
sorted(zip(G_b.vs['name'], G_b.degree()), key = lambda x: x[1], reverse = True)[:10]
```

And the bottom 10:

```python
sorted(zip(G_b.vs['name'], G_b.degree()), key = lambda x: x[1], reverse = False)[:10]
```

The distrubtioon of degree looks like

```python
plt.hist(G_b.degree())
plt.show()
```

We can also look at closeness centrality, or the average Euclidean or path
distance between a node and all others in the network. A node with the highest
closeness centrality is most likely to send a signal with the most coverage to
the rest of the network.

```python
G_b.vs[np.argmax(G_b.closeness())]
```

```python
pal = ig.GradientPalette("red", "blue", G_b.vcount())
G_b.vs['color'] = [pal.get(int(v)) for v in np.argsort(G_b.closeness())]
ig.plot(G_b, layout = layout, edge_width = 0.2)
```

Top and bottom:

```python
sorted(zip(G_b.vs['name'], G_b.closeness()), key = lambda x: x[1], reverse = True)[:10]
```

```python
sorted(zip(G_b.vs['name'], G_b.closeness()), key = lambda x: x[1], reverse = False)[:10]
```

Or eignvector centrality, an approach that weights degree by the centrality of
those to whom one is tied (and the degree to whom they are tied, etc.) In short,
its an $n$th order degree measure.

```python
G_b.vs['color'] = [pal.get(int(v)) for v in np.argsort(G_b.eigenvector_centrality())]
ig.plot(G_b, layout = layout, edge_width = 0.2)
```

Top and bottom:

```python
sorted(zip(G_b.vs['name'], G_b.eigenvector_centrality()), key = lambda x: x[1], reverse = True)[:10]
```

```python
sorted(zip(G_b.vs['name'], G_b.eigenvector_centrality()), key = lambda x: x[1], reverse = False)[:10]
```

## <span style="color:red">*Your Turn*</span>

<span style="color:red">Construct cells immediately below this that calculate
different kinds of centrality for distinct words or documents in a network
composed from your corpus of interest. Which type of words tend to be most and
least central? Can you identify how different centrality measures distinguish
different kind of words in your corpus? What do these patterns suggest about the
semantic content and structure of your documents?

We can also look at global statistics, like the density of a network, defined as
the number of edges existing divided by the total number of edges possible:

```python
G_b.density()
```

We can also calculate the average degree per node:

```python
np.mean(G_b.degree())
```

The diameter calculates the average distance between any two nodes in the
network:

```python
G_b.diameter()
```

Moreover, we can find cliques, or completely connected sets of nodes:

```python
G_b.largest_cliques()
```

```python
print(', '.join((G_b.vs[i]['name'] for i in G_b.largest_cliques()[0])))
```

Now lets look at a subgraph of the network, those nodes that are within 2 edges
'war'.

```python
warNeighbors = G_b.neighbors('war')
warNeighborsPlus1 = set(warNeighbors)
for n in warNeighbors:
    warNeighborsPlus1 |= set(G_b.neighbors(n))
```

```python
G_war = G_b.subgraph(G_b.vs.select(warNeighborsPlus1))
G_war.vcount()
```

```python
G_war.vs['color'] = 'red'
G_war.vs.find('war')['color'] = 'yellow'
ig.plot(G_war, target = 'data/war_plot.png')
```

There is a bug in iGraph that causes some plots to display incorrectly, but the
svgs they produce are unaffected. So we can just display the svgs with Ipython.

```python
IPython.display.Image('data/war_plot.png')
```

# POS based networks

Now lets look at links between specific parts of speech within a network.

For this we will be using the `nltk` POS facilities instead of the Stanford
ones. These are much faster, but also somewhat less accurate. (You get what you
*pay* for in computational power).

Lets look at nouns co-occurring in sentences using the top 10 (by score) reddit
posts:

```python
redditDF = pandas.read_csv('data/reddit.csv', index_col = 0)
```

```python
redditTopScores = redditDF.sort_values('score')[-100:]
redditTopScores['sentences'] = redditTopScores['text'].apply(lambda x: [nltk.word_tokenize(s) for s in nltk.sent_tokenize(x)])
redditTopScores.index = range(len(redditTopScores) - 1, -1,-1) #Reindex to make things nice in the future
redditTopScores
```

Now we'll normalize the tokens through stemming:

```python
redditTopScores['normalized_sents'] = redditTopScores['sentences'].apply(lambda x: [normlizeTokens(s, stopwordLst = None, stemmer = snowball) for s in x])
```

```python
redditTopScores
```

```python
def posCooccurrence(sentences, *posType, makeMatrix = False):
    pal = ig.RainbowPalette(n = len(posType))
    palMap = {p : pal.get(i) for i, p in enumerate(posType)}
    words = set()
    reducedSents = []
    #Only using the first kind of POS for each word
    wordsMap = {}
    for sent in sentences:
        s = [(w, t) for w, t in nltk.pos_tag(sent) if t in posType]
        for w, t in s:
            if w not in wordsMap:
                wordsMap[w] = t
        reducedSent = [w for w, t in s]
        words |= set(reducedSent)
        reducedSents.append(reducedSent)
    wordLst = list(words)
    wordIndices = {w: i for i, w in enumerate(wordLst)}
    wordCoCounts = {}
    #consider a sparse matrix if memory becomes an issue
    coOcMat = np.zeros((len(wordIndices), len(wordIndices)))
    for sent in reducedSents:
        for i, word1 in enumerate(sent):
            word1Index = wordIndices[word1]
            for word2 in sent[i + 1:]:
                coOcMat[word1Index][wordIndices[word2]] += 1
    if makeMatrix:
        return coOcMat, wordLst
    else:
        coOcMat = coOcMat.T + coOcMat
        edges = list(zip(*np.where(coOcMat)))
        weights = coOcMat[np.where(coOcMat)]
        kinds = [wordsMap[w] for w in wordLst]
        colours = [palMap[k] for k in kinds]
        g = ig.Graph( n = len(wordLst),
            edges = edges,
            vertex_attrs = {'name' : wordLst, 
                            'label' : wordLst, 
                            'kind' : kinds,
                            'color' : colours,
                           },
            edge_attrs = {'weight' : weights})
        return g
```

```python
gNN = posCooccurrence(redditTopScores['normalized_sents'].sum(), 'NN')
```

```python
gNN.vcount()
```

This is a bit to large to effectively visilize, so let's remove the verices
whose degree is less than or equal to 100:

```python
gNN_d = gNN.subgraph(gNN.vs.select(lambda x: x.degree() > 100))
```

```python
gNN_d.vcount()
```

```python
ig.plot(gNN_d, target = 'data/gNN_d.png')
```

```python
IPython.display.Image('data/gNN_d.png')
```

This is still a hairball because we retained the most highly connected
veritices. Let's trim a few edges as well.

```python
gNN_d.ecount()
```

```python
gNN_d = gNN_d.subgraph_edges(gNN_d.es.select(lambda x: x['weight'] > 10))
```

```python
gNN_d.ecount()
```

```python
#I know what the graph is like and the default layout is fine
ig.plot(gNN_d, target ='data/gNN_d2.png')
```

```python
IPython.display.Image('data/gNN_d2.png')
```

That is an interesting pattern, everyone is talking about themselves
("I...this", "I...that"). <span style="color:red">**How do you interpret all
those self loops?**

```python
for e in gNN_d.es.select(lambda x: x.is_loop()):
    print(gNN_d.vs[e.source])
```

What if we want to look at noun-verb pairs instead?

```python
gNV = posCooccurrence(redditTopScores['normalized_sents'].sum(), 'NN', 'VB')
```

`gNV` has co-occurrences between nouns and nouns as well as between verbs and
verbs. Let's remove these and make it purely about noun and verb combinations:

```python
gNV.ecount()
```

```python
gNV_p = gNV.subgraph_edges(gNV.es.select(lambda x: gNV.vs[x.source]['kind'] != gNV.vs[x.target]['kind']))
```

```python
gNV_p.ecount()
```

Dropping low weight edges and low degree vertices again gives us:

```python
gNV_p = gNV_p.subgraph(gNV_p.vs.select(lambda x: x.degree() > 50))
gNV_p = gNV_p.subgraph_edges(gNV_p.es.select(lambda x: x['weight'] > 5))
```

```python
gNV_p.vcount()
```

```python
layoutNV = gNV_p.layout_fruchterman_reingold()
ig.plot(gNV_p, layout = layoutNV, target = 'data/gNV_p.png')
```

```python
IPython.display.Image('data/gNV_p.png')
```

Looking at this plot it looks like `I` and `be` are the centers of two
communities. Lets check this by partitioning the networks, a graphical method of
clustering. Our first approach will use the information theoretic infomap
algorithm:

```python
print(gNV_p.community_infomap(edge_weights = gNV_p.es['weight']))
```

Let's focus on one of these communities and create an "ego network" surrounding
a single (important) word:

```python
gNV_I = gNV.subgraph(gNV.vs.find('i').neighbors())
#Still need to filter it a bit to display
gNV_I.vcount()
```

```python
gNV_I = gNV_I.subgraph(gNV_I.vs.select(lambda x: x.degree() > 10))
gNV_I = gNV_I.subgraph_edges(gNV_I.es.select(lambda x: x['weight'] > 15))
```

```python
layoutNV_I = gNV_I.layout_fruchterman_reingold(repulserad = .001)
ig.plot(gNV_I, layout = layoutNV_I, target = 'data/gNV_I.png')
```

```python
IPython.display.Image('data/gNV_I.png')
```

Instead of just those connect to a vertex we can find all those connected to it
within 2 hops, lets look at 'stori' for this

```python
gNV.vs.find('stori')
```

```python
storyNeighbors = gNV.neighbors('stori')
storyNeighborsPlus1 = set(storyNeighbors)
for n in storyNeighbors:
    storyNeighborsPlus1 |= set(gNV.neighbors(n))
```

```python
gNV_story = gNV.subgraph(gNV.vs.select(storyNeighborsPlus1))
gNV_story.vcount()
```

A large network, but we can compute some statistics:

```python
sorted(zip(gNV_story.vs['name'], gNV_story.degree()), key = lambda x: x[1], reverse = True)[:10]
```

Or by eignvector centrality:

```python
sorted(zip(gNV_story.vs['name'], gNV_story.eigenvector_centrality()), key = lambda x: x[1], reverse = True)[:10]
```

Notice that 'stori' isn't even in the top 10:

Lets filter it a bit than plot

```python
gNV_story = gNV_story.subgraph(gNV_story.vs.select(lambda x: x.degree() > 5 or x['name'] == 'stori'))
gNV_story = gNV_story.subgraph_edges(gNV_story.es.select(lambda x: x['weight'] > 10))
gNV_story.vs.find('stori')['color'] = 'yellow'
```

```python
layoutNV_story = gNV_story.layout_fruchterman_reingold(repulserad = .001)
ig.plot(gNV_story, layout = layoutNV_story, target = 'data/gNV_story.png')
```

```python
IPython.display.Image('data/gNV_story.png')
```

I is still in the middle

## <span style="color:red">*Your Turn*</span>

<span style="color:red">Construct cells immediately below this that construct at
least two different networks comprising different combinations of word types,
linked by different syntactic structures, that illuminate your corpus and the
dynamics you are interested to explore. Graph these networks or subnetworks
within them. What are relationships that are meaningful?
