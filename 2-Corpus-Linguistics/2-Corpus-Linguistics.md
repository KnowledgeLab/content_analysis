# Week 2 - Corpus Linguistics

This week, we take text corpora that we have developed, spidered, scraped, and
encoded, and we find and count words and simple phrases and properties of those
counts (e.g., word frequency distributions). Initially, we model how to search
corpora for keywords or phrases. Next, we examine the distributions of terms and
phrases across a corpus, and the corelation between different words and phrase
counts. Then we calculate word frequenceis, conditional frequences (the
frequency of word *shock* conditional on the presence of word *awe*), and
statistically significant collocations of length *n*. Then we distinguish
*important* words and phrase within the corpus, and image them with Wordls!
Next, we perform the Kullbach Leibler Divergence between the word frequence
distributions from two corpora. Finally, we coarsely disambiguate words based of
part-of-speech (POS) tagging, and normalize them through stemming and
lemmatization.

For this notebook we will be using the following packages

```python
#All these packages need to be installed from pip
import requests #for http requests
import nltk #the Natural Language Toolkit
import pandas #gives us DataFrames
import matplotlib.pyplot as plt #For graphics
import wordcloud #Makes word clouds
import numpy as np #For KL divergence
import scipy #For KL divergence
import seaborn as sns #makes our plots look nicer
import sklearn.manifold #For a manifold plot
from nltk.corpus import stopwords #For stopwords

#This 'magic' command makes the plots work better
#in the notebook, don't use it outside of a notebook
%matplotlib inline

import json #For API responses
import urllib.parse #For joining urls
```

# Retrieving exemplary corpora

To get started we will need some examples. Let's start by downloading one of the
corpuses from `nltk`, a Natural Language Toolkit developed by computational
linguists at the University of Pennsylvania. Lets take a look at how that works.

First we can get a list of works available from the Gutenburg corpus, with the
[corpus module](http://www.nltk.org/api/nltk.corpus.html).

```python
#You should have them already downloaded in the data directory
nltk.data.path.append('../data/nltk_data')
#If you haven't downloaded the copura then you will have to run `nltk.download()`
print(nltk.corpus.gutenberg.fileids())
print(len(nltk.corpus.gutenberg.fileids()))
```

We can also look at the individual works

```python
nltk.corpus.gutenberg.raw('shakespeare-macbeth.txt')[:1000]
```

All the listed works have been nicely marked up and classified for us so we can
do much better than just looking at raw text.

```python
print(nltk.corpus.gutenberg.words('shakespeare-macbeth.txt'))
print(nltk.corpus.gutenberg.sents('shakespeare-macbeth.txt'))
```

# Counting words

If we want to do some analysis we can start by simply counting the number of
times each word occurs within the corpus.

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

Notice how `wordCounter()` is not a very complicated function. That is because
the hard parts have already been done by `nltk`. If we were using unprocessed
text we would have to tokenize and determine what to do with the non-word
characters.

nltk also offers a built-in way for getting a frequency distribution from a list
of words:

```python
words = [word.lower() for word in nltk.corpus.gutenberg.words('shakespeare-macbeth.txt')]
freq = nltk.FreqDist(words)
print (freq['this'])
```

Lets sort and plot our counts to investigate the shape of our word frequency
distribution.

First we need to sort the words by count.

```python
#Doing this in place (changing the original DataFrame) as we don't need the unsorted DataFrame
countedWords.sort_values('count', ascending=False, inplace=True)
countedWords[:10]
```

Punctuation and very common words (articles 'a' and 'the'; prepositions 'of' and
'to') make up the most common values, but this isn't very interesting and can
actually get in the way of our analysis. We may remove these 'function words' by
removing according to a stopword list, setting some frequency threshold, or
using a weighting scheme (like tf.idf) to decrease their influence (all modeled
later in this tutorial).

```python
#Creating a figure and axis lets us do things like change the scaling or add a title
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(range(len(countedWords)), countedWords['count'])
plt.show()
```

This shows that likelihood of a word occurring is inversely proportional to its
rank. This effect is called [Zipf's
Law](https://en.wikipedia.org/wiki/Zipf%27s_law), and suggests that the most
frequent word will occur approximately twice as often as the second most
frequent word, three times as often as the third most frequent word, etc. Zipf's
law is most easily observed by plotting the data on a log-log graph, with the
axes being log (rank order) and log (frequency) resulting in a linear
relationship.

```python
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(range(len(countedWords)), countedWords['count'])
ax.set_yscale('log')
ax.set_xscale('log')
plt.show()
```

(The same relationship occurs in many other rankings, such as population ranks
of cities, corporation sizes, income rankings, etc.) It was imagined by Zipf to
be driven by a principle of 'least effort' where speakers did not work any
harder than necessary to communicate a given idea, but the basis for this
relationship is still not well understood and conforms at least as well to a
process of [preferential
attachment](https://en.wikipedia.org/wiki/Preferential_attachment) whereby
people disproportionately attend to popular words.

There are many other properties of words we can examine. First lets look
surrounding words with the concordance. To do this we need to load the text into
a `ConcordanceIndex`

```python
macbethIndex = nltk.text.ConcordanceIndex(nltk.corpus.gutenberg.words('shakespeare-macbeth.txt'))
```

Then we can retrieve all the words that cooccur with a word in a given word
window. Let's first look at `'macbeth'`.

```python
macbethIndex.print_concordance('macbeth')
```

Weird?! `'macbeth'` doesn't occur anywhere in the the text. What happened?

`ConcordanceIndex` is case sensitive, lets try looking for `'Macbeth'`

```python
macbethIndex.print_concordance('Macbeth')
```

That's better. What about something a lot less frequent?

```python
print(countedWords[countedWords['word'] == 'donalbaine'])
macbethIndex.print_concordance('Donalbaine')
```

# Getting and working with raw text

First, we will retrieve a collection of press releases from a GitHub API that
archived them, based on a number of analyses by Justin Grimmer, a political
scientist whose work we will read next week, and who will be joining the
University of Chicago next July!

GitHub API requests are made to `'https://api.github.com/'` and responses are in
JSON, similar to Tumblr's API.

We will get the information on [github.com/lintool/GrimmerSenatePressReleases](h
ttps://github.com/lintool/GrimmerSenatePressReleases) as it contains a nice set
documents.

```python
r = requests.get('https://api.github.com/repos/lintool/GrimmerSenatePressReleases')
senateReleasesData = json.loads(r.text)
print(senateReleasesData.keys())
print(senateReleasesData['description'])
```

What we are interested in here is the `'contents_url'`

```python
print(senateReleasesData['contents_url'])
```

We can use this to get any or all of the files from the repo.

```python
r = requests.get('https://api.github.com/repos/lintool/GrimmerSenatePressReleases/contents/raw/Whitehouse')
whitehouseLinks = json.loads(r.text)
whitehouseLinks[0]
```

Now we have a list of information about Whitehouse press releases. Let's look at
one of them.

```python
r = requests.get(whitehouseLinks[0]['download_url'])
whitehouseRelease = r.text
print(whitehouseRelease[:1000])
len(whitehouseRelease)
```

Now we have a string of correctly encoded text, similar to those we constructed
from last week's assignment. For analysis of its words, next we need to tokenize
it, or to split it into a sequence of tokens or word instances.

```python
whTokens = nltk.word_tokenize(whitehouseRelease)
whTokens[:10]
```

`whTokens` is a list of 'words' constructed from nltk's `word_tokenize` method
built on the [Penn Treebank
tokenizer](http://www.nltk.org/api/nltk.tokenize.html#module-
nltk.tokenize.treebank). While it's better than `.split(' ')`, a general python
string method that splits on whitespace, it is not perfect. There are many
different ways to tokenize a string. `word_tokenize` is unfortunately unaware of
sentences and is essentially a complicated regular expression (regex) run across
the string.

If we want to find sentences we can use something like `nltk.sent_tokenize()`,
which implements the [Punkt Sentence tokenizer](http://www.nltk.org/api/nltk.tok
enize.html#nltk.tokenize.punkt.PunktSentenceTokenizer), a machine learning based
algorithm that works well for many European languages.

We could also use the [Stanford
tokenizer](http://www.nltk.org/api/nltk.tokenize.html#module-
nltk.tokenize.stanford) or construct our own regex with
[`RegexpTokenizer()`](http://www.nltk.org/api/nltk.tokenize.html#module-
nltk.tokenize.regexp). Picking the correct tokenizer is important as the tokens
form the base of our analysis.

For our purposes in this example, the Penn Treebank tokenizer is fine.

To use the list of tokens in `nltk`, and take advantage of functions like
`concordance`, shown above, we can convert it into a `Text`.

```python
whText = nltk.Text(whTokens)

whitehouseIndex = nltk.text.ConcordanceIndex(whText) 
whitehouseIndex.print_concordance('Whitehouse')
```

*Note*, The `Text` class is for doing rapid, exploratory analysis. It provides
an easy interface to many of the operations we want to do, but it does not allow
us much control over the particular operations it glosses. When you are doing a
more complete analysis, you should be using the module specifically designed for
that task instead of the shortcut method `Text` provides, e.g. use
[`collocations` Module](http://www.nltk.org/api/nltk.html#module-
nltk.collocations) instead of `.collocations()`.

Now that we have gotten this loaded, let's glance at few features we will delve
into more deeply later.

For example, we can find words that statistically tend to occur together and
typically have a composite, idiomatic meaning irreducible to the semantics of
its component words. We will do this later with more control over exactly how
these are identified.

```python
whText.collocations()
```

Or we can pick a word (or words) and find what words tend to occur around it.

```python
whText.common_contexts(['stem'])
```

We can also just count the number of times the word occurs.

```python
whText.count('stem')
```


We can also plot each time a set of words occurs.

```python
sns.reset_orig() #Seaborn messes with this plot, disabling it
whText.dispersion_plot(['stem', 'cell', 'federal' ,'Lila', 'Barber', 'Whitehouse'])
sns.set() #Re-enabling seaborn
```

If we want to do an analysis of all the Whitehouse press releases we will first
need to obtain them. By looking at the API we can see the the URL we want is [ht
tps://api.github.com/repos/lintool/GrimmerSenatePressReleases/contents/raw/White
house](https://api.github.com/repos/lintool/GrimmerSenatePressReleases/contents/
raw/Whitehouse), so we can create a function to scrape the individual files.

If you want to know more about downloading from APIs, refer back to the 1st
notebook.

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

whReleases = getGithubFiles('https://api.github.com/repos/lintool/GrimmerSenatePressReleases/contents/raw/Whitehouse', maxFiles = 10)
whReleases[:5]
```

If you are having trouble downloading the data uncomment this next cell

```python
#whReleases = pandas.read_csv('../data/whReleases.csv')
```

Now we have all the texts in a DataFrame we can look at a few things.

First let's tokenize the texts with the same tokenizer as we used before. We
will just save the tokens as a list for now; no need to convert to `Text`s.

```python
whReleases['tokenized_text'] = whReleases['text'].apply(lambda x: nltk.word_tokenize(x))
```

Now lets see how long each of the press releases is

```python
whReleases['word_counts'] = whReleases['tokenized_text'].apply(lambda x: len(x))
whReleases['word_counts']
```

# Filtering and normalizing text

As we want to start comparing the different releases we may choose to do a bit
of filtering and normalizing that will allow us to focus on what we most care
about. We can first make all of the words lower case, then drop the non-word
tokens. Next, we can remove some 'stop words', stem the remaining words to
remove suffixes, prefixes and (in some languages) infixes, or lemmatize tokens
by intelligently grouping inflected or variant forms of the same word (e.g.,
with a stemmer and a dictionary).

To begin this process, we will first define a function to work over the
tokenized lists, then another to add normalized tokens to a new column.

Nltk has a built-in list of stopwords. They are already imported in the import
section. Let's first take a look at what they are.

```python
print(', '.join(stopwords.words('english')))
```

We can remove these stopwords from the analysis by fiat, but we could also take
a more principled approach by looking at the frequency distribution of words and
selecting a specific cut-off associated with the preservation of 'meaningful
words' identified upon inspection. Alternatively, we could automatically set a
cut-off by rule, such as removal of all words more frequent then the most
frequent verb, or the most frequent noun (not pronoun), or some term of central
interest.

Lets generate our own stop list now.

```python
countsDict = {}
for word in whReleases['tokenized_text'].sum():
    if word in countsDict:
        countsDict[word] += 1
    else:
        countsDict[word] = 1
word_counts = sorted(countsDict.items(), key = lambda x : x[1], reverse = True)
word_counts[:20]
```

We can look at this and pick the cutoff, usually it is at the first noun. So we
will cut all words before `'Whitehouse'`.

```python
#The stop list is then all words that occur before the first noun
stop_words_freq = []
for word, count in word_counts:
    if word == 'Whitehouse':
        break
    else:
        stop_words_freq.append(word)
stop_words_freq
wordnet = nltk.stem.WordNetLemmatizer()
wordnet.lemmatize('are')
```

Now we can create our function to normalize the tokens

```python
stop_words_nltk = stopwords.words('english')
#stop_words = ["the","it","she","he", "a"] #Uncomment this line if you want to use your own list of stopwords.

#The stemmers and lemmers need to be initialized before bing run
porter = nltk.stem.porter.PorterStemmer()
snowball = nltk.stem.snowball.SnowballStemmer('english')
wordnet = nltk.stem.WordNetLemmatizer()

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

whReleases['normalized_tokens'] = whReleases['tokenized_text'].apply(lambda x: normlizeTokens(x, stopwordLst = stop_words_nltk, stemmer = porter))

whReleases['normalized_tokens_count'] = whReleases['normalized_tokens'].apply(lambda x: len(x))

whReleases
```

The stemmer we use here is called the [Porter
Stemmer](http://www.nltk.org/api/nltk.stem.html#module-nltk.stem.porter), there
are many others, including another good one by the same person (Martin Porter)
called the [Snowball Stemmer](http://www.nltk.org/api/nltk.stem.html#module-
nltk.stem.snowball) or 'another stemmer' called the [Lancaster
Stemmer](http://www.nltk.org/_modules/nltk/stem/lancaster.html). We can also
normalize our words by lemmatizing them, or matching word variants or inflected
forms to a common morpheme in a dictionary like
[WordNet](https://wordnet.princeton.edu) through intelligent stemming rules.
(The WordNet function `morphy` returns the given word if it cannot be matched to
a word in the dictionary).

Now that it is cleaned we start analyzing the dataset. We can start by finding
frequency distributions for the dataset. Lets start looking at all the press
releases together. The [`ConditionalFreqDist`](http://www.nltk.org/api/nltk.html
#nltk.probability.ConditionalProbDist) class reads in an iterable of tuples, the
first element is the condition and the second the focal word. For starters, we
will use word lengths as the conditions, but tags or clusters will provide more
useful results.

```python
#.sum() adds together the lists from each row into a single list
whcfdist = nltk.ConditionalFreqDist(((len(w), w) for w in whReleases['normalized_tokens'].sum()))

#print the number of words
print(whcfdist.N())
```

From this we can lookup the distributions of different word lengths

```python
whcfdist[3].plot()
```

See that the most frequent 3-character word is "thi". But what is "thi"? It is
actually "this" stemmed by the Porter Stemmer.

```python
porter = nltk.stem.porter.PorterStemmer()
print (porter.stem('this'))
```

Let's try with the Snowball Stemer. See that "this" is corretly stemmed as a
4-character word.

```python
print (snowball.stem('this'))

whReleases['normalized_tokens'] = whReleases['tokenized_text'].apply(lambda x: normlizeTokens(x, stopwordLst = stop_words_nltk, stemmer = snowball))
whReleases['normalized_tokens_count'] = whReleases['normalized_tokens'].apply(lambda x: len(x))
whcfdist = nltk.ConditionalFreqDist(((len(w), w) for w in whReleases['normalized_tokens'].sum()))
whcfdist[3].plot()
```

We can also create a Conditional Probability Distribution or [`ConditionalProbDi
st`](http://www.nltk.org/api/nltk.html#nltk.probability.ConditionalProbDist)
from the `ConditionalFreqDist`. To do this, however, we need a model for the
probability distribution. A simple model is
[`ELEProbDist`](http://www.nltk.org/api/nltk.html#nltk.probability.ELEProbDist)
which gives the expected likelihood estimate for the probability distribution of
the experiment used to generate the observed frequency distribution.

```python
whcpdist = nltk.ConditionalProbDist(whcfdist, nltk.ELEProbDist)

#print the most common 2 letter word
print(whcpdist[2].max())

#And its probability
print(whcpdist[2].prob(whcpdist[2].max()))
```

Length is one feature of a word, but there are many more important features we
care about. Another critical feature is a word's role in the sentence, or its
part of speech (POS). Here, we will be classifying words according to their part
of speech (POS), using the
[`nltk.pos_tag()`](http://www.nltk.org/api/nltk.tag.html#nltk.tag.pos_tag). The
tags used here are those from the [Brown Corpus
tagset](http://www.scs.leeds.ac.uk/amalgam/tagsets/brown.html).

```python
whReleases['normalized_tokens_POS'] = [nltk.pos_tag(t) for t in whReleases['normalized_tokens']]
```

This gives us a new column with the part of speech as a short initialism and the
word in a tuple, exactly how the `nltk.ConditionalFreqDist()` function wants
them. We can now construct another conditional frequency distribution.

```python
whcfdist_WordtoPOS = nltk.ConditionalFreqDist(whReleases['normalized_tokens_POS'].sum())
list(whcfdist_WordtoPOS.items())[:10]
```

This gives the frequency of each word as each part of speech...which can be
uninformative and boring.

```python
whcfdist_WordtoPOS['administr'].plot()
```

What we want is the converse; the frequency of each part of speech for each
word.

```python
whcfdist_POStoWord = nltk.ConditionalFreqDist((p, w) for w, p in whReleases['normalized_tokens_POS'].sum())
```

We can now identify and collect all of the superlative adjectives

```python
whcfdist_POStoWord['JJS']
```

Or look at the most common nouns

```python
whcfdist_POStoWord['NN'].most_common(5)
```

Or plot the base form verbs against their number of occurrences

```python
whcfdist_POStoWord['VB'].plot()
```

We can then do a similar analysis of the word probabilities

```python
whcpdist_POStoWord = nltk.ConditionalProbDist(whcfdist_POStoWord, nltk.ELEProbDist)

#print the most common nouns
print(whcpdist_POStoWord['NN'].max())

#And its probability
print(whcpdist_POStoWord['NN'].prob(whcpdist_POStoWord['NN'].max()))
```

We can event create a...wait for it...*WORD CLOUD* or Wordl to gaze at and draw
mystical, approximate inferences about important nouns and verbs in our corpus

```python
wc = wordcloud.WordCloud(background_color="white", max_words=500, width= 1000, height = 1000, mode ='RGBA', scale=.5).generate(' '.join(whReleases['normalized_tokens'].sum()))
plt.imshow(wc)
plt.axis("off")
plt.savefig("whitehouse_word_cloud.pdf", format = 'pdf')
```

# Collocations

As illustrated above, we might want to find significant bigrams (two-word
phrases), trigrams (three-word phrases), n-grams (*n*-word phrases) or skip-
grams (noncontinguous 'phrases' with skip-length *n*). We will begin with the [`
nltk.collocations.BigramCollocationFinder`](http://www.nltk.org/api/nltk.html?hi
ghlight=bigramcollocationfinder#nltk.collocations.BigramCollocationFinder)
class, which can be given raw lists of strings with the `from_words()` method.
By default it only looks at continuous bigrams but there is an option
(`window_size`) to allow skip-grams.

```python
whBigrams = nltk.collocations.BigramCollocationFinder.from_words(whReleases['normalized_tokens'].sum())
print("There are {} bigrams in the finder".format(whBigrams.N))
```

To compare the bigrams we need to tell nltk what our score function is.
Initially, we will look at the raw counts.

```python
def bigramScoring(count, wordsTuple, total):
    return count

print(whBigrams.nbest(bigramScoring, 10))
```

One note about how `BigramCollocationFinder` works. It doesn't use the strings
internally.

```python
birgramScores = []

def bigramPrinting(count, wordsTuple, total):
    global birgramScores
    birgramScores.append("The first word is:  {}, The second word is: {}".format(*wordsTuple))
    #Returns None so all the tuples are considered to have the same rank

whBigrams.nbest(bigramPrinting, 10)
print('\n'.join(birgramScores[:10]))
```

The words are each given numeric IDs and there is a dictionary that maps the IDs
to the words they represent. This is a common performance optimization.

Two words can appear together by chance. Recall from  Manning and Schütze's
textbook that a t-value can be computed for each bigram to see how significant
the association is. You may also want to try computing the $\chi^2$, likelihood
ratio, and pointwise mutual information statistics.

```python
bigram_measures = nltk.collocations.BigramAssocMeasures()
whBigrams.score_ngrams(bigram_measures.student_t)[:40]
# other options include student_t, chi_sq, likelihood_ratio, pmi
```

There are a few other available measures

```python
[s for s in dir(bigram_measures) if s[0] != '_']
```

We can also look at trigrams

```python
trigram_measures = nltk.collocations.TrigramAssocMeasures()
whTrigrams = nltk.collocations.TrigramCollocationFinder.from_words(whReleases['normalized_tokens'].sum())
whTrigrams.score_ngrams(trigram_measures.student_t)[:20]
```

Our ngrams

```python
Ngrams = nltk.ngrams(whReleases['normalized_tokens'].sum(), 4)
Counts = {}
for ngram in list(Ngrams):
    if ngram in Counts.keys():
        Counts[ngram] += 1
    else:
        Counts[ngram] = 1
Filtered = {}
for key in Counts.keys():
    if Counts[key] < 2:
        pass
    else:
        Filtered[key] = Counts[key]
print(Filtered)
```

# KL Divergence

If we want to compare across the different corpus one of the places to start is
Kullback-Leibler divergence, which computes the relative entropy between two
distributions.

Recall that given two discrete probability distributions $P$ and $Q$, the
Kullback-Leibler divergence from $Q$ to $P$ is defined as:

$D_{\mathrm{KL}}(P\|Q) = \sum_i P(i) \, \log\frac{P(i)}{Q(i)}$.

The [scipy.stats.entropy()](https://docs.scipy.org/doc/scipy/reference/generated
/scipy.stats.entropy.html) function does the calculation for you, which takes in
two arrays of probabilities and computes the KL divergence. Note that the KL
divergence is in general not commutative, i.e. $D_{\mathrm{KL}}(P\|Q) \neq
D_{\mathrm{KL}}(Q\|P)$ .

Also note that the KL divernce is the sum of elementwise divergences. Scipy
provides [scipy.special.kl_div()](https://docs.scipy.org/doc/scipy/reference/gen
erated/scipy.special.kl_div.html#scipy-special-kl-div) which calculates
elementwise divergences for you.

To do this we will need to create the arrays, lets compare the Whitehouse
releases with the Kennedy releases. First we have to download them and load them
into a DataFrame.

```python
kenReleases = getGithubFiles('https://api.github.com/repos/lintool/GrimmerSenatePressReleases/contents/raw/Kennedy', maxFiles = 10)
kenReleases[:5]
```

Then we can tokenize, stem and remove stop words, like we did for the Whitehouse
press releases

```python
kenReleases['tokenized_text'] = kenReleases['text'].apply(lambda x: nltk.word_tokenize(x))
kenReleases['normalized_tokens'] = kenReleases['tokenized_text'].apply(lambda x: normlizeTokens(x, stopwordLst = stop_words_nltk, stemmer = snowball))
```

Now we need to compare the two collection of words, remove those not found in
both, and assign the remaining ones indices.

```python
whWords = set(whReleases['normalized_tokens'].sum())
kenWords = set(kenReleases['normalized_tokens'].sum())

#Change & to | if you want to keep all words
overlapWords = whWords & kenWords

overlapWordsDict = {word: index for index, word in enumerate(overlapWords)}
overlapWordsDict['student']
```

Now we can count the occurrences of each word in the corpora and create our
arrays. Note that we don't have to use numpy arrays as we do here. We could just
use a list, but the arrays are faster in numpy so we encourage you to get in the
habit of using them.

```python
def makeProbsArray(dfColumn, overlapDict):
    words = dfColumn.sum()
    countList = [0] * len(overlapDict)
    for word in words:
        try:
            countList[overlapDict[word]] += 1
        except KeyError:
            #The word is not common so we skip it
            pass
    countArray = np.array(countList)
    return countArray / countArray.sum()

whProbArray = makeProbsArray(whReleases['normalized_tokens'], overlapWordsDict)
kenProbArray = makeProbsArray(kenReleases['normalized_tokens'], overlapWordsDict)
kenProbArray.sum()
#There is a little bit of a floating point math error
#but it's too small to see with print and too small matter here
```

We can now compute the KL divergence. Pay attention to the asymmetry. Use [the
Jensen–Shannon
divergence](https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence),
which is the average KL divergence between each distribution and the average of
both distributions (i.e., the midpoint), if you want symmetry.

```python
wh_kenDivergence = scipy.stats.entropy(whProbArray, kenProbArray)
print (wh_kenDivergence)
ken_whDivergence = scipy.stats.entropy(kenProbArray, whProbArray)
print (ken_whDivergence)
```

Then, we can do the elementwise calculation and see which words best distinguish
the two corpora.

```python
wh_kenDivergence_ew = scipy.special.kl_div(whProbArray, kenProbArray)
kl_df = pandas.DataFrame(list(overlapWordsDict.keys()), columns = ['word'], index = list(overlapWordsDict.values()))
kl_df = kl_df.sort_index()
kl_df['elementwise divergence'] = wh_kenDivergence_ew
kl_df[:10]
```

```python
kl_df.sort_values(by='elementwise divergence', ascending=False)[:10]
```

Now we will apply what we learned today to the Guternberg texts in nltk and see
if we can detect patterns between them.

First, let's transform every text into normalized tokens. Note that in this
first step, no stopword is removed.

```python
fileids = nltk.corpus.gutenberg.fileids()
corpora = []
for fileid in fileids:
    words = nltk.corpus.gutenberg.words(fileid)
    normalized_tokens = normlizeTokens(words, stopwordLst = [], stemmer = snowball)
    corpora.append(normalized_tokens)
```

Then, let's separate the normalized tokens into stopwords and non-stopwords.

```python
corpora_s = []
corpora_nons = []
for corpus in corpora:
    s = []
    nons = []
    for word in corpus:
        if word in stop_words_nltk:
            s.append(word)
        else:
            nons.append(word)
    corpora_s.append(s)
    corpora_nons.append(nons)
```

Define some covenient funtions for calculating KL divergences.

```python
def Divergence(X, Y):
    P = X.copy()
    Q = Y.copy()
    P.columns = ['P']
    Q.columns = ['Q']
    df = Q.join(P).fillna(0)
    p = df.iloc[:,1]
    q = df.iloc[:,0]
    D_kl = scipy.stats.entropy(p, q)
    return D_kl

def kl_divergence(corpus1, corpus2):
    freqP = nltk.FreqDist(corpus1)
    P = pandas.DataFrame(list(freqP.values()), columns = ['frequency'], index = list(freqP.keys()))
    freqQ = nltk.FreqDist(corpus2)
    Q = pandas.DataFrame(list(freqQ.values()), columns = ['frequency'], index = list(freqQ.keys()))
    return Divergence(P, Q)
```

Calculate the KL divergence for each pair of corpora, turn the results into a
matrix, and visualize the matrix as a heatmap. Recall that
$D_{\mathrm{KL}}(P\|Q)$ measures the amount of information loss when $Q$ is used
to approximate $P$. Here, the rows are the $P$s used for calculating KL
divergences, and the columns are the $Q$s. So, each cell measures the amount of
information loss when the word distribution of the column text is used to
approximate the word distribution of the row text.

```python
L = []
for p in corpora:
    l = []
    for q in corpora:
        l.append(kl_divergence(p,q))
    L.append(l)
M = np.array(L)
fig = plt.figure()
div = pandas.DataFrame(M, columns = fileids, index = fileids)
ax = sns.heatmap(div)
plt.show()
```

See that works by the same author have the lowest within-group KL divergences.

To reveal more patterns, let's do a multidimensional scaling of the matrix.

```python
mds = sklearn.manifold.MDS()
pos = mds.fit(M).embedding_
x = pos[:,0]
y = pos[:,1]
fig, ax = plt.subplots(figsize = (6,6))
plt.plot(x, y, ' ')
for i, txt in enumerate(fileids):
    ax.annotate(txt, (x[i],y[i]))
```

Do you see any patterns in the image shown above? Does it make sense?

We may just want to focus on the distrbution of stopwords or non-stopwords.
Let's do the analysis again first for stopwords and then for non-stopwords.

```python
L = []
for p in corpora_s:
    l = []
    for q in corpora_s:
        l.append(kl_divergence(p,q))
    L.append(l)
M = np.array(L)
fig = plt.figure()
div = pandas.DataFrame(M, columns = fileids, index = fileids)
ax = sns.heatmap(div)
plt.show()
```

```python
L = []
for p in corpora_nons:
    l = []
    for q in corpora_nons:
        l.append(kl_divergence(p,q))
    L.append(l)
M = np.array(L)
fig = plt.figure()
div = pandas.DataFrame(M, columns = fileids, index = fileids)
ax = sns.heatmap(div)
plt.show()
```

Which analysis distinguishes the authors better?
