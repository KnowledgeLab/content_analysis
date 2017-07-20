# Week 7 - Information Extraction


This week, we move from arbitrary textual classification to the use of
computation and linguistic models to parse precise claims from documents. Rather
than focusing on simply the *ideas* in a corpus, here we focus on understanding
and extracting its precise *claims*. This process involves a sequential pipeline
of classifying and structuring tokens from text, each of which generates
potentially useful data for the content analyst. Steps in this process, which we
examine in this notebook, include: 1) tagging words by their part of speech
(POS) to reveal the linguistic role they play in the sentence (e.g., Verb, Noun,
Adjective, etc.); 2) tagging words as named entities (NER) such as places or
organizations; 3) structuring or "parsing" sentences into nested phrases that
are local to, describe or depend on one another; and 4) extracting informational
claims from those phrases, like the Subject-Verb-Object (SVO) triples we extract
here. While much of this can be done directly in the python package NLTK that we
introduced in week 2, here we use NLTK bindings to the Stanford NLP group's open
software, written in Java. Try typing a sentence into the online version
[here]('http://nlp.stanford.edu:8080/corenlp/') to get a sense of its potential.
It is superior in performance to NLTK's implementations, but takes time to run,
and so for these exercises we will parse and extract information for a very
small text corpus. Of course, for final projects that draw on these tools, we
encourage you to install the software on your own machines or shared servers at
the university (RCC, SSRC) in order to perform these operations on much more
text.

For this notebook we will be using the following packages:

```python
#All these packages need to be installed from pip
#For NLP
import nltk
from nltk.tag import StanfordNERTagger
from nltk.tag import StanfordPOSTagger
from nltk.parse import stanford
from nltk.tokenize import word_tokenize
from nltk.tree import Tree
from nltk.draw.tree import TreeView
from nltk.tokenize import sent_tokenize

import numpy as np #For arrays
import pandas #Gives us DataFrames
import matplotlib.pyplot as plt #For graphics
import seaborn #Makes the graphics look nicer

#Displays the graphs
import graphviz #You also need to install the command line graphviz

#These are from the standard library
import os.path
import zipfile
import subprocess
import io
import tempfile

%matplotlib inline
```

If you want to use the [Stanford NLP group](http://nlp.stanford.edu/) programs
with nltk on your own machine (you do *not* need to do this for this
assignment), it will require a little bit of setup. We are basing these
instructions on those provided by nltk,
[here](https://github.com/nltk/nltk/wiki/Installing-Third-Party-
Software#stanford-tagger-ner-tokenizer-and-parser), but with small changes to
work with our notebooks. We also note that lower performance versions of many of
the techniques demonstrated here are available natively within nltk (see the
updated [nltk book](http://www.nltk.org/book/)).

1. Install [Java 1.8+](http://www.oracle.com/technetwork/java/javase/downloads/j
dk8-downloads-2133151.html)
    + Make sure your `JAVAPATH` is setup if you're on windows
2. Download the following zip files from the Stanford NLP group, where DATE is
the release date of the files, this will be the value of `stanfordVersion`
    + [`stanford-corenlp-
full-2016-10-31.zip`](https://stanfordnlp.github.io/CoreNLP/)
    + [`stanford-postagger-full-
DATE.zip`](http://nlp.stanford.edu/software/tagger.html#Download)
    + [`stanford-ner-DATE.zip`](http://nlp.stanford.edu/software/CRF-
NER.html#Download)
    + [`stanford-parser-full-DATE.zip`](http://nlp.stanford.edu/software/lex-
parser.html#Download)
3. Unzip the files and place the resulting directories in the same location,
this will become `stanfordDir`
4. Lookup the version number used by the parser `stanford-parser-VERSION-
models.jar` and set to to be `parserVersion`

```python
#This is the date at the end of each of the zip files, e.g.
#the date in stanford-ner-2016-10-31.zip
stanfordVersion = '2016-10-31'

#This is the version numbers of the parser models, these
#are files in `stanford-parser-full-2016-10-31.zip`, e.g.
#stanford-parser-3.7.0-models.jar
parserVersion = '3.7.0'

#This is where the zip files were unzipped.Make sure to
#unzip into directories named after the zip files
#Don't just put all the files in `stanford-NLP`
stanfordDir = '/mnt/efs/resources/shared/stanford-NLP'

#Parser model, there are a few for english and a couple of other languages as well
modelName = 'englishPCFG.ser.gz'
```

We now will initialize all the tools

Setting up [NER tagger](http://www.nltk.org/api/nltk.tag.html?highlight=stanford
postagger#nltk.tag.stanford.StanfordNERTagger)

```python
nerClassifierPath = os.path.join(stanfordDir,'stanford-ner-{}'.format(stanfordVersion), 'classifiers/english.all.3class.distsim.crf.ser.gz')

nerJarPath = os.path.join(stanfordDir,'stanford-ner-{}'.format(stanfordVersion), 'stanford-ner.jar')

nerTagger = StanfordNERTagger(nerClassifierPath, nerJarPath)
```

Setting up [POS Tagger](http://www.nltk.org/api/nltk.tag.html?highlight=stanford
postagger#nltk.tag.stanford.StanfordPOSTagger)

```python
postClassifierPath = os.path.join(stanfordDir, 'stanford-postagger-full-{}'.format(stanfordVersion), 'models/english-bidirectional-distsim.tagger')

postJarPath = os.path.join(stanfordDir,'stanford-postagger-full-{}'.format(stanfordVersion), 'stanford-postagger.jar')

postTagger = StanfordPOSTagger(postClassifierPath, postJarPath)
```

Setting up [Parser](http://www.nltk.org/api/nltk.parse.html?highlight=stanfordpa
rser#module-nltk.parse.stanford)

```python
parserJarPath = os.path.join(stanfordDir, 'stanford-parser-full-{}'.format(stanfordVersion), 'stanford-parser.jar')

parserModelsPath = os.path.join(stanfordDir, 'stanford-parser-full-{}'.format(stanfordVersion), 'stanford-parser-{}-models.jar'.format(parserVersion))

modelPath = os.path.join(stanfordDir, 'stanford-parser-full-{}'.format(stanfordVersion), modelName)

#The model files are stored in the jar, we need to extract them for nltk to use
if not os.path.isfile(modelPath):
    with zipfile.ZipFile(parserModelsPath) as zf:
        with open(modelPath, 'wb') as f:
            f.write(zf.read('edu/stanford/nlp/models/lexparser/{}'.format(modelName)))

parser = stanford.StanfordParser(parserJarPath, parserModelsPath, modelPath)

depParser = stanford.StanfordDependencyParser(parserJarPath, parserModelsPath)
```

Open Information Extraction is a module packaged within the Stanford Core NLP
package, but it is not yet supported by `nltk`. As a result, we will be defining
our own function that runs the Stanford Core NLP java code right here. For other
projects, it is often useful to use Java or other programs (in C, C++) within a
python workflow, and this is an example. `openIE()` takes in a string or list of
strings and then produces as output all the subject, verb, object (SVO) triples
Stanford Corenlp can find, as a DataFrame.

```python
#Watch out, this will very rarely raise an error since it trusts stanford-corenlp 
def openIE(target):
    if isinstance(target, list):
        target = '\n'.join(target)
    #setup the java targets
    coreDir = '{}/stanford-corenlp-full-{}'.format(stanfordDir, stanfordVersion)
    cp = '{0}/stanford-corenlp-{1}.jar:{0}/stanford-corenlp-{1}-models.jar:CoreNLP-to-HTML.xsl:slf4j-api.jar:slf4j-simple.jar'.format(coreDir, parserVersion)
    with tempfile.NamedTemporaryFile(mode = 'w', delete = False) as f:
        #Core nlp requires a files, so we will make a temp one to pass to it
        #This file should be deleted by the OS soon after it has been used
        f.write(target)
        f.seek(0)
        print("Starting OpenIE run")
        #If you know what these options do then you should mess with them on your own machine and not the shared server
        sp = subprocess.run(['java', '-mx2g', '-cp', cp, 'edu.stanford.nlp.naturalli.OpenIE', '-threads', '1', f.name], stdout = subprocess.PIPE, stderr = subprocess.PIPE)
        #Live stderr is non-trivial so this is the best we can do
        print(sp.stderr.decode('utf-8'))
        retSting = sp.stdout.decode('utf-8')
    #Making the DataFrame, again having to pass a fake file, yay POSIX I guess
    with io.StringIO(retSting) as f:
        df = pandas.read_csv(f, delimiter = '\t', names =['certainty', 'subject', 'verb', 'object'])
    return df
```

First, we will illustrate these tools on some *very* short examples:

```python
text = ['I saw the elephant in my pajamas.', 'The quick brown fox jumped over the lazy dog.', 'While in France, Christine Lagarde discussed short-term stimulus efforts in a recent interview with the Wall Street Journal.', 'Trayvon Benjamin Martin was an African American from Miami Gardens, Florida, who, at 17 years old, was fatally shot by George Zimmerman, a neighborhood watch volunteer, in Sanford, Florida.', 'Buffalo buffalo Buffalo buffalo buffalo buffalo Buffalo buffalo']
tokenized_text = [word_tokenize(t) for t in text]
print('\n'.join(text))
```

# Part-of-Speech (POS) tagging

In POS tagging, we classify each word by its semantic role in a sentence. The
Stanford POS tagger uses the [Penn Treebank tag set]('http://repository.upenn.ed
u/cgi/viewcontent.cgi?article=1603&context=cis_reports') to POS tag words from
input sentences. As discussed in the second assignment, this is a relatively
precise tagset, which allows more informative tags, and also more opportunities
to err :-).

|#. |Tag |Description |
|---|----|------------|
|1.     |CC     |Coordinating conjunction
|2.     |CD     |Cardinal number
|3.     |DT     |Determiner
|4.     |EX     |Existential there
|5.     |FW     |Foreign word
|6.     |IN     |Preposition or subordinating conjunction
|7.     |JJ     |Adjective
|8.     |JJR|   Adjective, comparative
|9.     |JJS|   Adjective, superlative
|10.|   LS      |List item marker
|11.|   MD      |Modal
|12.|   NN      |Noun, singular or mass
|13.|   NNS     |Noun, plural
|14.|   NNP     |Proper noun, singular
|15.|   NNPS|   Proper noun, plural
|16.|   PDT     |Predeterminer
|17.|   POS     |Possessive ending
|18.|   PRP     |Personal pronoun
|19.|   PRP\$|  Possessive pronoun
|20.|   RB      |Adverb
|21.|   RBR     |Adverb, comparative
|22.|   RBS     |Adverb, superlative
|23.|   RP      |Particle
|24.|   SYM     |Symbol
|25.|   TO      |to
|26.|   UH      |Interjection
|27.|   VB      |Verb, base form
|28.|   VBD     |Verb, past tense
|29.|   VBG     |Verb, gerund or present participle
|30.|   VBN     |Verb, past participle
|31.|   VBP     |Verb, non-3rd person singular present
|32.|   VBZ     |Verb, 3rd person singular present
|33.|   WDT     |Wh-determiner
|34.|   WP      |Wh-pronoun
|35.|   WP$     |Possessive wh-pronoun
|36.|   WRB     |Wh-adverb

```python
pos_sents = postTagger.tag_sents(tokenized_text)
print(pos_sents)
```

This looks quite good. Now we will try POS tagging with a somewhat larger
corpus. We consider a few of the top posts from the reddit data we used last
week.

```python
redditDF = pandas.read_csv('data/reddit.csv')
```

Grabbing the 10 highest scoring posts and tokenizing the sentences. Once again,
notice that we aren't going to do any kind of stemming this week (although
*semantic* normalization may be performed where we translate synonyms into the
same focal word).

```python
redditTopScores = redditDF.sort_values('score')[-10:]
redditTopScores['sentences'] = redditTopScores['text'].apply(lambda x: [nltk.word_tokenize(s) for s in nltk.sent_tokenize(x)])
redditTopScores.index = range(len(redditTopScores) - 1, -1,-1) #Reindex to make things nice in the future
redditTopScores
```

```python
redditTopScores['POS_sents'] = redditTopScores['sentences'].apply(lambda x: postTagger.tag_sents(x))
```

```python
redditTopScores['POS_sents']
```

And count the number of `NN` (nouns)

```python
countTarget = 'NN'
targetCounts = {}
for entry in redditTopScores['POS_sents']:
    for sentence in entry:
        for ent, kind in sentence:
            if kind != countTarget:
                continue
            elif ent in targetCounts:
                targetCounts[ent] += 1
            else:
                targetCounts[ent] = 1
sortedTargets = sorted(targetCounts.items(), key = lambda x: x[1], reverse = True)
sortedTargets[:20]
```

What about the number of top verbs (`VB`)?

```python
countTarget = 'VB'
targetCounts = {}
for entry in redditTopScores['POS_sents']:
    for sentence in entry:
        for ent, kind in sentence:
            if kind != countTarget:
                continue
            elif ent in targetCounts:
                targetCounts[ent] += 1
            else:
                targetCounts[ent] = 1
sortedTargets = sorted(targetCounts.items(), key = lambda x: x[1], reverse = True)
sortedTargets[:20]
```

What about the adjectives that modify the word, "computer"?

```python
NTarget = 'JJ'
Word = 'computer'
NResults = set()
for entry in redditTopScores['POS_sents']:
    for sentence in entry:
        for (ent1, kind1),(ent2,kind2) in zip(sentence[:-1], sentence[1:]):
            if (kind1,ent2.lower())==(NTarget,Word):
                NResults.add(ent1)
            else:
                continue

print(NResults)     
```

## Evaluating POS tagger

We can check the POS tagger by running it on a manually tagged corpus and
identifying a reasonable error metric.

```python
treeBank = nltk.corpus.treebank
treeBank.tagged_sents()[0]
```

```python
treeBank.sents()[0]
```

```python
stanfordTags = postTagger.tag_sents(treeBank.sents()[:30])
```

And compare the two

```python
NumDiffs = 0
for sentIndex in range(len(stanfordTags)):
    for wordIndex in range(len(stanfordTags[sentIndex])):
        if stanfordTags[sentIndex][wordIndex][1] != treeBank.tagged_sents()[sentIndex][wordIndex][1]:
            if treeBank.tagged_sents()[sentIndex][wordIndex][1] != '-NONE-':
                print("Word: {}  \tStanford: {}\tTreebank: {}".format(stanfordTags[sentIndex][wordIndex][0], stanfordTags[sentIndex][wordIndex][1], treeBank.tagged_sents()[sentIndex][wordIndex][1]))
                NumDiffs += 1
total = sum([len(s) for s in stanfordTags])
print("The Precision is {:.3f}%".format((total-NumDiffs)/total * 100))
```

So we can see that the stanford POS tagger is quite good. Nevertheless, for a 20
word sentence, we only have a 66% chance ($1-.96^{20}$) of tagging (and later
parsing) it correctly.

## <span style="color:red">*Your turn*</span>

<span style="color:red">In the cells immediately following, perform POS tagging
on a meaningful (but modest) subset of a corpus associated with your final
project. Examine the list of words associated with at least three different
parts of speech. Consider conditional associations (e.g., adjectives associated
with nouns or adverbs with verbs of interest). What do these distributions
suggest about your corpus?

# Named-Entity Recognition

Named Entity Recognition (NER) is also a classification task, which identifies
named objects. Included with Stanford NER are a 4 class model trained on the
CoNLL 2003 eng.train, a 7 class model trained on the MUC 6 and MUC 7 training
data sets, and a 3 class model trained on both data sets plus some additional
data (including ACE 2002 and limited data in-house) on the intersection of those
class sets.

**3 class**:    Location, Person, Organization

**4 class**:    Location, Person, Organization, Misc

**7 class**:    Location, Person, Organization, Money, Percent, Date, Time

These models each use distributional similarity features, which provide some
performance gain at the cost of increasing their size and runtime. Also
available are the same models missing those features.

(We note that the training data for the 3 class model does not include any
material from the CoNLL eng.testa or eng.testb data sets, nor any of the MUC 6
or 7 test or devtest datasets, nor Alan Ritter's Twitter NER data, so all of
these would be valid tests of its performance.)

First, we tag our first set of exemplary sentences:

```python
classified_sents = nerTagger.tag_sents(tokenized_text)
print(classified_sents)
```

We can also run NER over our entire corpus:

```python
redditTopScores['classified_sents'] = redditTopScores['sentences'].apply(lambda x: nerTagger.tag_sents(x))
```

```python
redditTopScores['classified_sents']
```

Find the most common entities (which are, of course, boring):

```python
entityCounts = {}
for entry in redditTopScores['classified_sents']:
    for sentence in entry:
        for ent, kind in sentence:
            if ent in entityCounts:
                entityCounts[ent] += 1
            else:
                entityCounts[ent] = 1
sortedEntities = sorted(entityCounts.items(), key = lambda x: x[1], reverse = True)
sortedEntities[:10]
```

Or those occurring only twice:

```python
[x[0] for x in sortedEntities if x[1] == 2]
```

We could also list the most common "non-objects". (We note that we're not
graphing these because there are so few here.)

```python
nonObjCounts = {}
for entry in redditTopScores['classified_sents']:
    for sentence in entry:
        for ent, kind in sentence:
            if kind == 'O':
                continue
            elif ent in nonObjCounts:
                nonObjCounts[ent] += 1
            else:
                nonObjCounts[ent] = 1
sortedNonObj = sorted(nonObjCounts.items(), key = lambda x: x[1], reverse = True)
sortedNonObj[:10]
```

What about the Organizations?

```python
OrgCounts = {}
for entry in redditTopScores['classified_sents']:
    for sentence in entry:
        for ent, kind in sentence:
            if kind != 'ORGANIZATION':
                continue
            elif ent in OrgCounts:
                OrgCounts[ent] += 1
            else:
                OrgCounts[ent] = 1
sortedOrgs = sorted(OrgCounts.items(), key = lambda x: x[1], reverse = True)
sortedOrgs[:10]
```

These, of course, have much smaller counts.

## <span style="color:red">*Your turn*</span>

<span style="color:red">In the cells immediately following, perform NER on a
(modest) subset of your corpus of interest. List all of the different kinds of
entities tagged? What does their distribution suggest about the focus of your
corpus? For a subset of your corpus, tally at least one type of named entity and
calculate the Precision, Recall and F-score for the NER classification just
performed.

# Parsing

Here we will introduce the Stanford Parser by feeding it tokenized text from our
initial example sentences. The parser is a dependency parser, but this initial
program outputs a simple, self-explanatory phrase-structure representation.

```python
parses = list(parser.parse_sents(tokenized_text)) #Converting the iterator to a list so we can call by index. They are still 
fourthSentParseTree = list(parses[3]) #iterators so be careful about re-running code, without re-running this block
print(fourthSentParseTree)
```

Trees are a common data structure and there are a large number of things to do
with them. What we are intetered in is the relationship between different types
of speech

```python
def treeRelation(parsetree, relationType, *targets):
    if isinstance(parsetree, list):
        parsetree = parsetree[0]
    if set(targets) & set(parsetree.leaves()) != set(targets):
        return []
    else:
        retList = []
        for subT in parsetree.subtrees():
            if subT.label() == relationType:
                if set(targets) & set(subT.leaves()) == set(targets):
                    retList.append([(subT.label(), ' '.join(subT.leaves()))])
    return retList
```

```python
def treeSubRelation(parsetree, relationTypeScope, relationTypeTarget, *targets):
    if isinstance(parsetree, list):
        parsetree = parsetree[0]
    if set(targets) & set(parsetree.leaves()) != set(targets):
        return []
    else:
        retSet = set()
        for subT in parsetree.subtrees():
            if set(targets) & set(subT.leaves()) == set(targets):
                if subT.label() == relationTypeScope:
                    for subsub in subT.subtrees():
                        if subsub.label()==relationTypeTarget:
                            retSet.add(' '.join(subsub.leaves()))
    return retSet
```

```python
treeRelation(fourthSentParseTree, 'NP', 'Florida', 'who')
```

Notice that Florida occurs twice in two different nested noun phrases in the
sentence.

We can also find all of the verbs within the noun phrase defined by one or more
target words:

```python
treeSubRelation(fourthSentParseTree, 'NP', 'VBN', 'Florida', 'who')
```

Or if we want to to look at the whole tree

```python
fourthSentParseTree[0].pretty_print()
```

Or another sentence

```python
list(parses[1])[0].pretty_print()
```

## Dependency parsing and graph representations

Dependency parsing was developed to robustly capture linguistic dependencies
from text. The complex tags associated with these parses are detailed
[here]('http://universaldependencies.org/u/overview/syntax.html'). When parsing
with the dependency parser, we will work directly from the untokenized text.
Note that no *processing* takes place before parsing sentences--we do not remove
so-called stop words or anything that plays a syntactic role in the sentence,
although anaphora resolution and related normalization may be performed before
or after parsing to enhance the value of information extraction.

```python
depParses = list(depParser.raw_parse_sents(text)) #Converting the iterator to a list so we can call by index. They are still 
secondSentDepParseTree = list(depParses[1])[0] #iterators so be careful about re-running code, without re-running this block
print(secondSentDepParseTree)
```

This is a graph and we can convert it to a dot file and use that to visulize it.
Try traversing the tree and extracting elements that are nearby one another.

```python
secondSentGraph = graphviz.Source(secondSentDepParseTree.to_dot())
secondSentGraph
```

Or another sentence

```python
graphviz.Source(list(depParses[3])[0].to_dot())
```

We can also do a dependency parse on the reddit sentences:

```python
topPostDepParse = list(depParser.parse_sents(redditTopScores['sentences'][0]))
```

This takes a few seconds, but now lets look at the parse tree from one of the
processed sentences.

The sentence is:

```python
targetSentence = 7
print(' '.join(redditTopScores['sentences'][0][targetSentence]))
```

Which leads to a very rich dependancy tree:

```python
graphviz.Source(list(topPostDepParse[targetSentence])[0].to_dot())
```

## <span style="color:red">*Your turn*</span>

<span style="color:red">In the cells immediately following, parse a (modest)
subset of your corpus of interest. How deep are the phrase structure and
dependency parse trees nested? How does parse depth relate to perceived sentence
complexity? What are five things you can extract from these parses for
subsequent analysis? (e.g., nouns collocated in a noun phrase; adjectives that
modify a noun; etc.) Capture these sets of things for a focal set of words
(e.g., "Bush", "Obama", "Trump"). What do they reveal about the roles that these
entities are perceive to play in the social world inscribed by your texts?

# Information extraction

Information extraction approaches typically (as here, with Stanford's Open IE
engine) ride atop the dependency parse of a sentence. They are a pre-coded
example of the type analyzed in the prior.

```python
ieDF = openIE(text)
```

`openIE()` prints everything stanford core produces and we can see from looking
at it that initializing the dependency parser takes most of the time, so calling
the function will always take at least 12 seconds.

```python
ieDF
```

No buffalos (because there were no verbs), but the rest is somewhat promising.
Note, however, that it abandoned the key theme of the sentence about the tragic
Trayvon Martin death ("fatally shot"), likely because it was buried so deeply
within the complex phrase structure. This is obviously a challenge.

## <span style="color:red">*Your thoughts*</span>

<span style="color:red">How would you extract relevant information about the
Trayvon Martin sentence directly from the dependency parse (above)? Code an
example here. (For instance, what compound nouns show up with what verb phrases
within the sentence?)

And we can also look for subject, object, target triples in one of the reddit
stories.

```python
ieDF = openIE(redditTopScores['text'][0])
```

```python
ieDF
```

That's almost 200 triples in only:

```python
len(redditTopScores['sentences'][0])
```

sentences and

```python
sum([len(s) for s in redditTopScores['sentences'][0]])
```

words.

Lets find at the most common subject in this story.

```python
ieDF['subject'].value_counts()
```

I is followed by various male pronouns and compound nouns (e.g., "old man"). 'I'
occures most often with the following verbs:

```python
ieDF[ieDF['subject'] == 'I']['verb'].value_counts()
```

and the following objects

```python
ieDF[ieDF['subject'] == 'I']['object'].value_counts()
```

## <span style="color:red">*Your Turn*</span>

<span style="color:red">In the cells immediately following, perform open
information extraction on a modest subset of texts relevant to your final
project. Analyze the relative attachment of several subjects relative to verbs
and objects and visa versa? Describe how you would select among these statements
to create a database of high-value statements for your project and do it.

```python

```
