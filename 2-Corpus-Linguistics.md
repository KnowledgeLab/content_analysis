
# Layout
# update !!!!
+ Searching text for keywords
+ Distribution of terms
+ Correlation
+ Word frequencies
+ Conditional frequencies
+ Statistically significant collocations
+ Distinguishing or Important words and phrases (Wordls!)
    + tf-idf
        + next week
+ POS-tagged words and phrases
+ Lemmatized words and phrases
    + stemmers
+ Dictionary-based annotations.

+ divergences
    + kl
    + Needs to be added

+ Sources
    + US senate press releases
        + e.g. [http://www.reid.senate.gov/press_releases](http://www.reid.senate.gov/press_releases)
    + Tumblr
    + Literature

+ More headers
+ More topic based less narrative

%%javascript
$.getScript('https://kmahelona.github.io/ipython_notebook_goodies/ipython_notebook_toc.js')



# Week 2 - Corpus Linguistics

Intro stuff

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
import seaborn as sns; sns.set()
from nltk.corpus import stopwords #For stopwords

#This 'magic' command makes the plots work better
#in the notebook, don't use it outside of a notebook
%matplotlib inline

import json #For API responses
import urllib.parse #For joining urls
```

# Getting our corpuses

To get started we will need some targets, let's start by downloading one of the corpuses from `nltk`. Lets take a look at how that works.

First we can get a list of works available from the Gutenburg corpus, with the [corpus module](http://www.nltk.org/api/nltk.corpus.html).


```python
print(nltk.corpus.gutenberg.fileids())
print(len(nltk.corpus.gutenberg.fileids()))
```

    ['austen-emma.txt', 'austen-persuasion.txt', 'austen-sense.txt', 'bible-kjv.txt', 'blake-poems.txt', 'bryant-stories.txt', 'burgess-busterbrown.txt', 'carroll-alice.txt', 'chesterton-ball.txt', 'chesterton-brown.txt', 'chesterton-thursday.txt', 'edgeworth-parents.txt', 'melville-moby_dick.txt', 'milton-paradise.txt', 'shakespeare-caesar.txt', 'shakespeare-hamlet.txt', 'shakespeare-macbeth.txt', 'whitman-leaves.txt']
    18
    

We can also look at the individual works


```python
nltk.corpus.gutenberg.raw('shakespeare-macbeth.txt')[:1000]
```




    "[The Tragedie of Macbeth by William Shakespeare 1603]\n\n\nActus Primus. Scoena Prima.\n\nThunder and Lightning. Enter three Witches.\n\n  1. When shall we three meet againe?\nIn Thunder, Lightning, or in Raine?\n  2. When the Hurley-burley's done,\nWhen the Battaile's lost, and wonne\n\n   3. That will be ere the set of Sunne\n\n   1. Where the place?\n  2. Vpon the Heath\n\n   3. There to meet with Macbeth\n\n   1. I come, Gray-Malkin\n\n   All. Padock calls anon: faire is foule, and foule is faire,\nHouer through the fogge and filthie ayre.\n\nExeunt.\n\n\nScena Secunda.\n\nAlarum within. Enter King Malcome, Donalbaine, Lenox, with\nattendants,\nmeeting a bleeding Captaine.\n\n  King. What bloody man is that? he can report,\nAs seemeth by his plight, of the Reuolt\nThe newest state\n\n   Mal. This is the Serieant,\nWho like a good and hardie Souldier fought\n'Gainst my Captiuitie: Haile braue friend;\nSay to the King, the knowledge of the Broyle,\nAs thou didst leaue it\n\n   Cap. Doubtfull it stood,\nAs two spent Swimmers, t"



All the listed works have been nicely marked up and classified for us so we can do much better than just looking at raw text.


```python
print(nltk.corpus.gutenberg.words('shakespeare-macbeth.txt'))
print(nltk.corpus.gutenberg.sents('shakespeare-macbeth.txt'))
```

    ['[', 'The', 'Tragedie', 'of', 'Macbeth', 'by', ...]
    [['[', 'The', 'Tragedie', 'of', 'Macbeth', 'by', 'William', 'Shakespeare', '1603', ']'], ['Actus', 'Primus', '.'], ...]
    

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




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>word</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>fellowes</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6</td>
      <td>gentle</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>lands</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>aleppo</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>station</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>wayle</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2</td>
      <td>dispaire</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2</td>
      <td>reade</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>pluckt</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>intermission</td>
    </tr>
  </tbody>
</table>
</div>



Notice how `wordCounter()` is not a very complicated function. That is because the hard parts have already been done by `nltk`. If we were using unprocessed text we would have to tokenize and determine what to do with the non-word characters.

nltk also offers a built-in way for getting a frequency distribution from a list of words:


```python
words = [word.lower() for word in nltk.corpus.gutenberg.words('shakespeare-macbeth.txt')]
freq = nltk.FreqDist(words)
print (freq['this'])
```

    104
    

Lets plot our counts and see what it looks like.

First we need to sort the words by count.


```python
#Doing this in place as we don't need the unsorted DataFrame
countedWords.sort_values('count', ascending=False, inplace=True)
countedWords[:10]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>word</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1851</th>
      <td>1962</td>
      <td>,</td>
    </tr>
    <tr>
      <th>1505</th>
      <td>1235</td>
      <td>.</td>
    </tr>
    <tr>
      <th>2766</th>
      <td>650</td>
      <td>the</td>
    </tr>
    <tr>
      <th>570</th>
      <td>637</td>
      <td>'</td>
    </tr>
    <tr>
      <th>2048</th>
      <td>546</td>
      <td>and</td>
    </tr>
    <tr>
      <th>723</th>
      <td>477</td>
      <td>:</td>
    </tr>
    <tr>
      <th>1476</th>
      <td>384</td>
      <td>to</td>
    </tr>
    <tr>
      <th>750</th>
      <td>348</td>
      <td>i</td>
    </tr>
    <tr>
      <th>3211</th>
      <td>338</td>
      <td>of</td>
    </tr>
    <tr>
      <th>2217</th>
      <td>241</td>
      <td>a</td>
    </tr>
  </tbody>
</table>
</div>



The punctuation and very common words (like 'a' and 'the') makes up all the top most common values, this isn't very interesting and can actually get in the way of analysis. We will be removing these later on.


```python
plt.plot(range(len(countedWords)), countedWords['count'])
plt.show()
```


![png](2-Corpus-Linguistics_files/2-Corpus-Linguistics_15_0.png)


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

    No matches
    

weird, `'macbeth'` doesn't occur anywhere in the the text. What happened?

`ConcordanceIndex` is case sensitive, lets try looking for `'Macbeth'`


```python
macbethIndex.print_concordance('Macbeth')
```

    Displaying 25 of 61 matches:
                                      Macbeth by William Shakespeare 1603 ] Act
     the Heath 3 . There to meet with Macbeth 1 . I come , Gray - Malkin All . 
    but all ' s too weake : For braue Macbeth ( well hee deserues that Name ) D
    smay ' d not this our Captaines , Macbeth and Banquoh ? Cap . Yes , as Spar
    , And with his former Title greet Macbeth Rosse . Ile see it done King . Wh
     King . What he hath lost , Noble Macbeth hath wonne . Exeunt . Scena Terti
    ithin . 3 . A Drumme , a Drumme : Macbeth doth come All . The weyward Siste
    , the Charme ' s wound vp . Enter Macbeth and Banquo . Macb . So foule and 
    an : what are you ? 1 . All haile Macbeth , haile to thee Thane of Glamis 2
    hee Thane of Glamis 2 . All haile Macbeth , haile to thee Thane of Cawdor 3
    hee Thane of Cawdor 3 . All haile Macbeth , that shalt be King hereafter Ba
    . Hayle 3 . Hayle 1 . Lesser than Macbeth , and greater 2 . Not so happy , 
    hough thou be none : So all haile Macbeth , and Banquo 1 . Banquo , and Mac
    eth , and Banquo 1 . Banquo , and Macbeth , all haile Macb . Stay you imper
    he King hath happily receiu ' d , Macbeth , The newes of thy successe : and
    gh the roughest Day Banq . Worthy Macbeth , wee stay vpon your leysure Macb
    I built An absolute Trust . Enter Macbeth , Banquo , Rosse , and Angus . O 
    ke , To cry , hold , hold . Enter Macbeth . Great Glamys , worthy Cawdor , 
    ruice ouer the Stage . Then enter Macbeth Macb . If it were done , when ' t
    re giues way to in repose . Enter Macbeth , and a Seruant with a Torch . Gi
    hether they liue , or dye . Enter Macbeth . Macb . Who ' s there ? what hoa
    ard a voyce cry , Sleep no more : Macbeth does murther Sleepe , the innocen
    ore Cawdor Shall sleepe no more : Macbeth shall sleepe no more Lady . Who w
     made a Shift to cast him . Enter Macbeth . Macd . Is thy Master stirring ?
    selues : awake , awake , Exeunt . Macbeth and Lenox . Ring the Alarum Bell 
    

That's a lot better

what about something a lot less frequent


```python
print(countedWords[countedWords['word'] == 'donalbaine'])
macbethIndex.print_concordance('Donalbaine')
```

          count        word
    1220      7  donalbaine
    Displaying 7 of 7 matches:
    m within . Enter King Malcome , Donalbaine , Lenox , with attendants , mee
    Enter King , Lenox , Malcolme , Donalbaine , and Attendants . King . Is ex
    rches . Enter King , Malcolme , Donalbaine , Banquo , Lenox , Macduff , Ro
     ' th ' second Chamber ? Lady . Donalbaine Mac . This is a sorry sight Lad
    er , and Treason , Banquo , and Donalbaine : Malcolme awake , Shake off th
    to brag of . Enter Malcolme and Donalbaine . Donal . What is amisse ? Macb
    were subborned , Malcolme , and Donalbaine the Kings two Sonnes Are stolne
    

# Getting press releases

First we need to understand the GitHub API

requests are made to `'https://api.github.com/'` and responses are in JSON, similar to Tumblr's API.

We will get the information on [github.com/lintool/GrimmerSenatePressReleases](https://github.com/lintool/GrimmerSenatePressReleases) as it contains a nice set documents.


```python
r = requests.get('https://api.github.com/repos/lintool/GrimmerSenatePressReleases')
senateReleasesData = json.loads(r.text)
print(senateReleasesData.keys())
print(senateReleasesData['description'])
```

    dict_keys(['hooks_url', 'has_pages', 'pushed_at', 'html_url', 'keys_url', 'git_refs_url', 'issue_comment_url', 'default_branch', 'pulls_url', 'comments_url', 'git_commits_url', 'stargazers_url', 'labels_url', 'name', 'git_tags_url', 'homepage', 'private', 'deployments_url', 'branches_url', 'trees_url', 'network_count', 'watchers', 'subscribers_count', 'downloads_url', 'size', 'stargazers_count', 'has_wiki', 'blobs_url', 'language', 'assignees_url', 'open_issues', 'milestones_url', 'merges_url', 'issues_url', 'forks', 'archive_url', 'full_name', 'open_issues_count', 'ssh_url', 'watchers_count', 'url', 'contributors_url', 'subscription_url', 'mirror_url', 'events_url', 'teams_url', 'contents_url', 'notifications_url', 'forks_count', 'git_url', 'description', 'statuses_url', 'svn_url', 'collaborators_url', 'has_issues', 'clone_url', 'forks_url', 'tags_url', 'owner', 'issue_events_url', 'languages_url', 'commits_url', 'compare_url', 'subscribers_url', 'has_downloads', 'created_at', 'updated_at', 'fork', 'releases_url', 'id'])
    Grimmer's Senate Press Releases
    

What we are interested in is the `'contents_url'`


```python
print(senateReleasesData['contents_url'])
```

    https://api.github.com/repos/lintool/GrimmerSenatePressReleases/contents/{+path}
    

We can use this to get any, or all of the files from the repo


```python
r= requests.get('https://api.github.com/repos/lintool/GrimmerSenatePressReleases/contents/raw/Whitehouse')
whitehouseLinks = json.loads(r.text)
whitehouseLinks[0]
```




    {'_links': {'git': 'https://api.github.com/repos/lintool/GrimmerSenatePressReleases/git/blobs/f524289ee563dca58690c8d36c23dce5dbd9962a',
      'html': 'https://github.com/lintool/GrimmerSenatePressReleases/blob/master/raw/Whitehouse/10Apr2007Whitehouse123.txt',
      'self': 'https://api.github.com/repos/lintool/GrimmerSenatePressReleases/contents/raw/Whitehouse/10Apr2007Whitehouse123.txt?ref=master'},
     'download_url': 'https://raw.githubusercontent.com/lintool/GrimmerSenatePressReleases/master/raw/Whitehouse/10Apr2007Whitehouse123.txt',
     'git_url': 'https://api.github.com/repos/lintool/GrimmerSenatePressReleases/git/blobs/f524289ee563dca58690c8d36c23dce5dbd9962a',
     'html_url': 'https://github.com/lintool/GrimmerSenatePressReleases/blob/master/raw/Whitehouse/10Apr2007Whitehouse123.txt',
     'name': '10Apr2007Whitehouse123.txt',
     'path': 'raw/Whitehouse/10Apr2007Whitehouse123.txt',
     'sha': 'f524289ee563dca58690c8d36c23dce5dbd9962a',
     'size': 2206,
     'type': 'file',
     'url': 'https://api.github.com/repos/lintool/GrimmerSenatePressReleases/contents/raw/Whitehouse/10Apr2007Whitehouse123.txt?ref=master'}



Now we have a list of information about Whitehouse press releases. Lets look at one of them.


```python
r = requests.get(whitehouseLinks[0]['download_url'])
whitehouseRelease = r.text
print(whitehouseRelease[:1000])
len(whitehouseRelease)
```

    SEN. WHITEHOUSE SHARES WESTERLY GIRL'S STORY IN PUSH FOR STEM CELL RESEARCH
      Sharing the story of Lila Barber, a 12 year old girl from Westerly, Sen. Sheldon Whitehouse (D-R.I.) on Tuesday, April 10, 2007, illustrated the hope stem cell research can offer in a speech on the Senate floor in favor of legislation to expand federal funding for stem cell research.  
       Whitehouse met Lila two weeks ago. She was diagnosed two years ago with osteosarcoma, a cancerous bone condition, and last year underwent cadaver bone transplant surgery. The procedure saved her leg and is helping her remain cancer-free, but the transplanted tissue will not grow with her and likely will break down over time. Stem cell research, Whitehouse explained, could vastly improve the care of patients like Lila by allowing surgeons to enhance transplants with a patient's own stem cells, which could replace the lost bone and cartilage, or grow entirely new replacement bones and joints. 
       "Stem cell research gives hope
    




    2206



Now we have a blob of text we first need to tokenize it.


```python
whTokens = nltk.word_tokenize(whitehouseRelease)
whTokens[:10]
```




    ['SEN.',
     'WHITEHOUSE',
     'SHARES',
     'WESTERLY',
     'GIRL',
     "'S",
     'STORY',
     'IN',
     'PUSH',
     'FOR']



`whTokens` is a list of 'words', it's better than `.split(' ')`,  but it is not perfect. There are many different ways to tokenize a string and the one we used here is called the [Penn Treebank tokenizer](http://www.nltk.org/api/nltk.tokenize.html#module-nltk.tokenize.treebank). This tokenizer isn't aware of sentences and is a basically a complicated regex that's run over the string.

If we want to find sentences we can use something like `nltk.sent_tokenize()` which implements the [Punkt Sentence tokenizer](http://www.nltk.org/api/nltk.tokenize.html#nltk.tokenize.punkt.PunktSentenceTokenizer), a machine learning based algorithm that works well for many European languages.

We could also use the [Stanford tokenizer](http://www.nltk.org/api/nltk.tokenize.html#module-nltk.tokenize.stanford) or use our own regex with [`RegexpTokenizer()`](http://www.nltk.org/api/nltk.tokenize.html#module-nltk.tokenize.regexp). Picking the correct tokenizer is important as the tokens form the base of our analysis.

For now though the Penn Treebank tokenizer is fine.

To use the list of tokens in `nltk` we can convert it into a `Text`.


```python
whText = nltk.Text(whTokens)
```

*Note*, The `Text` class is for doing exploratory and fast analysis. It provides an easy interface to many of the operations we want to do, but it does not allow us much control. When you are doing a full analysis you should be using the module for that task instead of the method `Text` provides, e.g. use  [`collocations` Module](http://www.nltk.org/api/nltk.html#module-nltk.collocations) instead of `.collocations()`.

Now that we got this loaded lets, look at few things

We can find words that tend to occur together


```python
whText.collocations()
```

    Rhode Island; stem cells; cell research; Cell Enhancement; Enhancement
    Act; President Bush; Stem Cell; stem cell; Stem cell
    

Or we can pick a word (or words) and find what words tend to occur around it


```python
whText.common_contexts(['stem'])
```

    hope_cell ``_cell of_cell on_cells ._cell for_cell the_cell
    embryonic_cells own_cells
    

We can also just count the number of times the word occurs


```python
whText.count('stem')
```




    7



Or plot each time it occurs


```python
whText.dispersion_plot(['stem', 'cell', 'federal' ,'Lila', 'Barber', 'Whitehouse'])
```


![png](2-Corpus-Linguistics_files/2-Corpus-Linguistics_43_0.png)


If we want to do an analysis of all the Whitehouse press releases we will first need to obtain them. By looking at the API we can see the the URL we want is [https://api.github.com/repos/lintool/GrimmerSenatePressReleases/contents/raw/Whitehouse](https://api.github.com/repos/lintool/GrimmerSenatePressReleases/contents/raw/Whitehouse), so we can create a function to scrape the individual files.

If you want to know more about downloading from APIs look at the 1st notebook


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




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>download_url</th>
      <th>html_url</th>
      <th>name</th>
      <th>path</th>
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>https://raw.githubusercontent.com/lintool/Grim...</td>
      <td>https://github.com/lintool/GrimmerSenatePressR...</td>
      <td>10Apr2007Whitehouse123.txt</td>
      <td>raw/Whitehouse/10Apr2007Whitehouse123.txt</td>
      <td>SEN. WHITEHOUSE SHARES WESTERLY GIRL'S STORY I...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>https://raw.githubusercontent.com/lintool/Grim...</td>
      <td>https://github.com/lintool/GrimmerSenatePressR...</td>
      <td>10Apr2008Whitehouse2.txt</td>
      <td>raw/Whitehouse/10Apr2008Whitehouse2.txt</td>
      <td>SEN. WHITEHOUSE SAYS PRESIDENT BUSH MUST BEGIN...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>https://raw.githubusercontent.com/lintool/Grim...</td>
      <td>https://github.com/lintool/GrimmerSenatePressR...</td>
      <td>10Apr2008Whitehouse3.txt</td>
      <td>raw/Whitehouse/10Apr2008Whitehouse3.txt</td>
      <td>EPA MUST REVIEW LEGAL PROCESS TO ROOT OUT POLI...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>https://raw.githubusercontent.com/lintool/Grim...</td>
      <td>https://github.com/lintool/GrimmerSenatePressR...</td>
      <td>10Aug2007Whitehouse78.txt</td>
      <td>raw/Whitehouse/10Aug2007Whitehouse78.txt</td>
      <td>R.I. SENATORS PRAISE SEN. DENIAL OF LNG FACILI...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>https://raw.githubusercontent.com/lintool/Grim...</td>
      <td>https://github.com/lintool/GrimmerSenatePressR...</td>
      <td>10Jan2008Whitehouse35.txt</td>
      <td>raw/Whitehouse/10Jan2008Whitehouse35.txt</td>
      <td>SEN. WHITEHOUSE COMMENTS ON ONE-YEAR ANNIVERSA...</td>
    </tr>
  </tbody>
</table>
</div>



Now we have all the texts in a DataFrame we can look at a few things.

First let's tokenize the texts with the same tokenizer as we used before, we will just save the tokens as a list for now, no need to convert to `Text`s.


```python
whReleases['tokenized_text'] = whReleases['text'].apply(lambda x: nltk.word_tokenize(x))
```

Now lets see how long each of the press releases is


```python
whReleases['word_counts'] = whReleases['tokenized_text'].apply(lambda x: len(x))
whReleases['word_counts']
```




    0    397
    1    344
    2    553
    3    216
    4    257
    5    380
    6    270
    7    521
    8    484
    9    482
    Name: word_counts, dtype: int64



As we want to start comparing the different releases we need to do a bit of normalizing. We first will make all the words lower case, drop the non-word tokens, then we can stem them and finally remove some stop words.

To do this we will define a function to work over the tokenized lists, then use another apply to add the normalized tokens to a new column.

Nltk has a built-in list of stopwords. They are already imported in the import section. Let's first take a look at what they are.


```python
stopwords.words('english')
```




    ['i',
     'me',
     'my',
     'myself',
     'we',
     'our',
     'ours',
     'ourselves',
     'you',
     'your',
     'yours',
     'yourself',
     'yourselves',
     'he',
     'him',
     'his',
     'himself',
     'she',
     'her',
     'hers',
     'herself',
     'it',
     'its',
     'itself',
     'they',
     'them',
     'their',
     'theirs',
     'themselves',
     'what',
     'which',
     'who',
     'whom',
     'this',
     'that',
     'these',
     'those',
     'am',
     'is',
     'are',
     'was',
     'were',
     'be',
     'been',
     'being',
     'have',
     'has',
     'had',
     'having',
     'do',
     'does',
     'did',
     'doing',
     'a',
     'an',
     'the',
     'and',
     'but',
     'if',
     'or',
     'because',
     'as',
     'until',
     'while',
     'of',
     'at',
     'by',
     'for',
     'with',
     'about',
     'against',
     'between',
     'into',
     'through',
     'during',
     'before',
     'after',
     'above',
     'below',
     'to',
     'from',
     'up',
     'down',
     'in',
     'out',
     'on',
     'off',
     'over',
     'under',
     'again',
     'further',
     'then',
     'once',
     'here',
     'there',
     'when',
     'where',
     'why',
     'how',
     'all',
     'any',
     'both',
     'each',
     'few',
     'more',
     'most',
     'other',
     'some',
     'such',
     'no',
     'nor',
     'not',
     'only',
     'own',
     'same',
     'so',
     'than',
     'too',
     'very',
     's',
     't',
     'can',
     'will',
     'just',
     'don',
     'should',
     'now',
     'd',
     'll',
     'm',
     'o',
     're',
     've',
     'y',
     'ain',
     'aren',
     'couldn',
     'didn',
     'doesn',
     'hadn',
     'hasn',
     'haven',
     'isn',
     'ma',
     'mightn',
     'mustn',
     'needn',
     'shan',
     'shouldn',
     'wasn',
     'weren',
     'won',
     'wouldn']




```python
stop_words = stopwords.words('english')
#stop_words = ["the","it","she","he", "a"] #Uncomment this line if you want to use your own list of stopwords.

#The stemmer needs to be initialized before bing run
porter = nltk.stem.porter.PorterStemmer()
snowball = nltk.stem.snowball.SnowballStemmer('english')

def normlizeTokens(tokenLst, stopwordLst = stop_words, stemmer = porter):
    #We can use a generator here as we just need to iterate over it

    #Lowering the case and removing non-words
    workingIter = (w.lower() for w in tokenLst if w.isalpha())

    #Now we can use it
    workingIter = (stemmer.stem(w) for w in workingIter)

    #We will return a list with the stopwords removed
    return [w for w in workingIter if w not in stopwordLst]

whReleases['normalized_tokens'] = whReleases['tokenized_text'].apply(lambda x: normlizeTokens(x))

whReleases['normalized_tokens_count'] = whReleases['normalized_tokens'].apply(lambda x: len(x))

whReleases
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>download_url</th>
      <th>html_url</th>
      <th>name</th>
      <th>path</th>
      <th>text</th>
      <th>tokenized_text</th>
      <th>word_counts</th>
      <th>normalized_tokens</th>
      <th>normalized_tokens_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>https://raw.githubusercontent.com/lintool/Grim...</td>
      <td>https://github.com/lintool/GrimmerSenatePressR...</td>
      <td>10Apr2007Whitehouse123.txt</td>
      <td>raw/Whitehouse/10Apr2007Whitehouse123.txt</td>
      <td>SEN. WHITEHOUSE SHARES WESTERLY GIRL'S STORY I...</td>
      <td>[SEN., WHITEHOUSE, SHARES, WESTERLY, GIRL, 'S,...</td>
      <td>397</td>
      <td>[whitehous, share, westerli, girl, stori, push...</td>
      <td>231</td>
    </tr>
    <tr>
      <th>1</th>
      <td>https://raw.githubusercontent.com/lintool/Grim...</td>
      <td>https://github.com/lintool/GrimmerSenatePressR...</td>
      <td>10Apr2008Whitehouse2.txt</td>
      <td>raw/Whitehouse/10Apr2008Whitehouse2.txt</td>
      <td>SEN. WHITEHOUSE SAYS PRESIDENT BUSH MUST BEGIN...</td>
      <td>[SEN., WHITEHOUSE, SAYS, PRESIDENT, BUSH, MUST...</td>
      <td>344</td>
      <td>[whitehous, say, presid, bush, must, begin, br...</td>
      <td>171</td>
    </tr>
    <tr>
      <th>2</th>
      <td>https://raw.githubusercontent.com/lintool/Grim...</td>
      <td>https://github.com/lintool/GrimmerSenatePressR...</td>
      <td>10Apr2008Whitehouse3.txt</td>
      <td>raw/Whitehouse/10Apr2008Whitehouse3.txt</td>
      <td>EPA MUST REVIEW LEGAL PROCESS TO ROOT OUT POLI...</td>
      <td>[EPA, MUST, REVIEW, LEGAL, PROCESS, TO, ROOT, ...</td>
      <td>553</td>
      <td>[epa, must, review, legal, process, root, poli...</td>
      <td>305</td>
    </tr>
    <tr>
      <th>3</th>
      <td>https://raw.githubusercontent.com/lintool/Grim...</td>
      <td>https://github.com/lintool/GrimmerSenatePressR...</td>
      <td>10Aug2007Whitehouse78.txt</td>
      <td>raw/Whitehouse/10Aug2007Whitehouse78.txt</td>
      <td>R.I. SENATORS PRAISE SEN. DENIAL OF LNG FACILI...</td>
      <td>[R.I, ., SENATORS, PRAISE, SEN, ., DENIAL, OF,...</td>
      <td>216</td>
      <td>[senat, prais, sen, denial, lng, facil, permit...</td>
      <td>115</td>
    </tr>
    <tr>
      <th>4</th>
      <td>https://raw.githubusercontent.com/lintool/Grim...</td>
      <td>https://github.com/lintool/GrimmerSenatePressR...</td>
      <td>10Jan2008Whitehouse35.txt</td>
      <td>raw/Whitehouse/10Jan2008Whitehouse35.txt</td>
      <td>SEN. WHITEHOUSE COMMENTS ON ONE-YEAR ANNIVERSA...</td>
      <td>[SEN., WHITEHOUSE, COMMENTS, ON, ONE-YEAR, ANN...</td>
      <td>257</td>
      <td>[whitehous, comment, anniversari, presid, bush...</td>
      <td>132</td>
    </tr>
    <tr>
      <th>5</th>
      <td>https://raw.githubusercontent.com/lintool/Grim...</td>
      <td>https://github.com/lintool/GrimmerSenatePressR...</td>
      <td>10Mar2008Whitehouse8.txt</td>
      <td>raw/Whitehouse/10Mar2008Whitehouse8.txt</td>
      <td>SENS. REED, WHITEHOUSE WELCOME RHODE ISLAND ST...</td>
      <td>[SENS, ., REED, ,, WHITEHOUSE, WELCOME, RHODE,...</td>
      <td>380</td>
      <td>[sen, reed, whitehous, welcom, rhode, island, ...</td>
      <td>195</td>
    </tr>
    <tr>
      <th>6</th>
      <td>https://raw.githubusercontent.com/lintool/Grim...</td>
      <td>https://github.com/lintool/GrimmerSenatePressR...</td>
      <td>10Sep2007Whitehouse72.txt</td>
      <td>raw/Whitehouse/10Sep2007Whitehouse72.txt</td>
      <td>REP. WHITEHOUSE ISSUES STATEMENT ON GEN. PETRA...</td>
      <td>[REP., WHITEHOUSE, ISSUES, STATEMENT, ON, GEN....</td>
      <td>270</td>
      <td>[whitehous, issu, statement, petraeu, iraq, re...</td>
      <td>118</td>
    </tr>
    <tr>
      <th>7</th>
      <td>https://raw.githubusercontent.com/lintool/Grim...</td>
      <td>https://github.com/lintool/GrimmerSenatePressR...</td>
      <td>11Apr2007Whitehouse122.txt</td>
      <td>raw/Whitehouse/11Apr2007Whitehouse122.txt</td>
      <td>SEN. WHITEHOUSE URGES BUSH FOR NEW DIRECTION I...</td>
      <td>[SEN., WHITEHOUSE, URGES, BUSH, FOR, NEW, DIRE...</td>
      <td>521</td>
      <td>[whitehous, urg, bush, new, direct, iraq, shel...</td>
      <td>257</td>
    </tr>
    <tr>
      <th>8</th>
      <td>https://raw.githubusercontent.com/lintool/Grim...</td>
      <td>https://github.com/lintool/GrimmerSenatePressR...</td>
      <td>11Jan2007Whitehouse161.txt</td>
      <td>raw/Whitehouse/11Jan2007Whitehouse161.txt</td>
      <td>SENS. REED, WHITEHOUSE URGE PORTUGAL TO RECONS...</td>
      <td>[SENS, ., REED, ,, WHITEHOUSE, URGE, PORTUGAL,...</td>
      <td>484</td>
      <td>[sen, reed, whitehous, urg, portug, reconsid, ...</td>
      <td>254</td>
    </tr>
    <tr>
      <th>9</th>
      <td>https://raw.githubusercontent.com/lintool/Grim...</td>
      <td>https://github.com/lintool/GrimmerSenatePressR...</td>
      <td>11Mar2008Whitehouse7.txt</td>
      <td>raw/Whitehouse/11Mar2008Whitehouse7.txt</td>
      <td>WHITEHOUSE UNVEILS 'BUSH DEBT': $7.7 TRILLION ...</td>
      <td>[WHITEHOUSE, UNVEILS, 'BUSH, DEBT, ', :, $, 7....</td>
      <td>482</td>
      <td>[whitehous, unveil, debt, trillion, foregon, s...</td>
      <td>260</td>
    </tr>
  </tbody>
</table>
</div>



The stemmer we use here is called the [Porter Stemmer](http://www.nltk.org/api/nltk.stem.html#module-nltk.stem.porter), there are many others, including another good one by the same person (Martin Porter) called the [Snowball Stemmer](http://www.nltk.org/api/nltk.stem.html#module-nltk.stem.snowball).

Now that it is cleaned we start analyzing the dataset. We can start by finding frequency disruptions for the dataset. Lets start looking at all the press releases together. The[`ConditionalFreqDist`](http://www.nltk.org/api/nltk.html#nltk.probability.ConditionalProbDist) class reads in a iterable of tuples, the first element is the condition and the second the word, for now we will use word lengths as the conditions, but tags or clusters would provide more useful results.


```python
#.sum() adds together the lists from each row into a single list
whcfdist = nltk.ConditionalFreqDist(((len(w), w) for w in whReleases['normalized_tokens'].sum()))

#print the number of words
print(whcfdist.N())
```

    2038
    

From this we can lookup the distributions of different word lengths


```python
whcfdist[3].plot()
```


![png](2-Corpus-Linguistics_files/2-Corpus-Linguistics_56_0.png)


See that the most frequent 3-character word is "thi". But what is "thi"? It is actually "this" stemmed by the Porter Stemmer. 


```python
porter = nltk.stem.porter.PorterStemmer()
print (porter.stem('this'))
```

    thi
    

Let's try with the Snowball Stemer. See that "this" is corretly stemmed as a 4-character word. 


```python
print (snowball.stem('this'))

whReleases['normalized_tokens'] = whReleases['tokenized_text'].apply(lambda x: normlizeTokens(x, stemmer = snowball))
whReleases['normalized_tokens_count'] = whReleases['normalized_tokens'].apply(lambda x: len(x))
whcfdist = nltk.ConditionalFreqDist(((len(w), w) for w in whReleases['normalized_tokens'].sum()))
whcfdist[3].plot()
```

    this
    


![png](2-Corpus-Linguistics_files/2-Corpus-Linguistics_60_1.png)


We can also create a [`ConditionalProbDist`](http://www.nltk.org/api/nltk.html#nltk.probability.ConditionalProbDist) from the `ConditionalFreqDist`, to do this though we need a model for the probability distribution. A simple model is [`ELEProbDist`](http://www.nltk.org/api/nltk.html#nltk.probability.ELEProbDist) which gives the expected likelihood estimate.


```python
whcpdist = nltk.ConditionalProbDist(whcfdist, nltk.ELEProbDist)

#print the most common 2 letter word
print(whcpdist[2].max())

#And its probability
print(whcpdist[2].prob(whcpdist[2].max()))
```

    us
    0.6470588235294118
    

Word lengths are a good start but there are many more Important features we care about. To start with we will be classifying words with their part of speech (POS), using the [`nltk.pos_tag()`](http://www.nltk.org/api/nltk.tag.html#nltk.tag.pos_tag).


```python
whReleases['normalized_tokens_POS'] = [nltk.pos_tag(t) for t in whReleases['normalized_tokens']]
```

This gives us a new column with the part of speech as a short initialism and the word in a tuple, exactly how the `nltk.ConditionalFreqDist()` function wants them. We can now make another conditional frequency distribution.


```python
whcfdist_WordtoPOS = nltk.ConditionalFreqDist(whReleases['normalized_tokens_POS'].sum())
list(whcfdist_WordtoPOS.items())[:10]
```




    [('wonderland', FreqDist({'NN': 1})),
     ('expect', FreqDist({'VBP': 1})),
     ('strict', FreqDist({'JJ': 1})),
     ('contribut', FreqDist({'NN': 1})),
     ('tutor', FreqDist({'NN': 1})),
     ('question', FreqDist({'NN': 5})),
     ('reason', FreqDist({'NN': 1})),
     ('dredg', FreqDist({'JJ': 1, 'NN': 2})),
     ('gave', FreqDist({'VBD': 1})),
     ('reiter', FreqDist({'NN': 1}))]



This gives the frequency of each word being each part of speech, which is usually quite boring.


```python
whcfdist_WordtoPOS['administr'].plot()
```


![png](2-Corpus-Linguistics_files/2-Corpus-Linguistics_68_0.png)


What we want is the the other direction, the frequency of each part of speech for each word.


```python
whcfdist_POStoWord = nltk.ConditionalFreqDist((p, w) for w, p in whReleases['normalized_tokens_POS'].sum())
```

We can now get all of the superlative adjectives


```python
whcfdist_POStoWord['JJS']
```




    FreqDist({'best': 1, 'strongest': 2})



Or look at the most common nouns


```python
whcfdist_POStoWord['NN'].most_common(5)
```




    [('bush', 24), ('presid', 23), ('iraq', 22), ('rhode', 18), ('island', 18)]



Or plot the base form verbs against their number of occurrences


```python
whcfdist_POStoWord['VB'].plot()
```


![png](2-Corpus-Linguistics_files/2-Corpus-Linguistics_76_0.png)


We can then do a similar analysis of the probabilities


```python
whcpdist_POStoWord = nltk.ConditionalProbDist(whcfdist_POStoWord, nltk.ELEProbDist)

#print the most common nouns
print(whcpdist_POStoWord['NN'].max())

#And its probability
print(whcpdist_POStoWord['NN'].prob(whcpdist_POStoWord['NN'].max()))
```

    bush
    0.01786365293474298
    


```python
wc = wordcloud.WordCloud(background_color="white", max_words=500, width= 1000, height = 1000, mode ='RGBA', scale=.5).generate(' '.join(whReleases['normalized_tokens'].sum()))
plt.imshow(wc)
plt.axis("off")
plt.savefig("whitehouse_word_cloud.pdf", format = 'pdf')
```


![png](2-Corpus-Linguistics_files/2-Corpus-Linguistics_79_0.png)


# Collocations

We also might want to find significant bigrams and trigrams. To do this we will use the [`nltk.collocations.BigramCollocationFinder`](http://www.nltk.org/api/nltk.html?highlight=bigramcollocationfinder#nltk.collocations.BigramCollocationFinder) class, which can be given raw lists of strings with the `from_words()` method. By default it only looks at continuous bigrams but there is an option (`window_size`) to allow skip-gram.


```python
whBigrams = nltk.collocations.BigramCollocationFinder.from_words(whReleases['normalized_tokens'].sum())
print("There are {} bigrams in the finder".format(whBigrams.N))
```

    There are 1999 bigrams in the finder
    

There are a lot of bigrams, but most of them only occur once, we should first filter our set to remove some of the least common.


```python
#This modifies the finder inplace
whBigrams.apply_freq_filter(2)
print("There are {} bigrams in the finder".format(whBigrams.N))
```

    There are 1999 bigrams in the finder
    

To compare the bigrams we need to tell nltk what our score function is, for now we will just look at the raw counts.


```python
def bigramScoring(count, wordsTuple, total):
    return count

print(whBigrams.nbest(bigramScoring, 10))
```

    [('rhode', 'island'), ('presid', 'bush'), ('sheldon', 'whitehous'), ('stem', 'cell'), ('whitehous', 'said'), ('bush', 'administr'), ('american', 'peopl'), ('bring', 'troop'), ('senat', 'sheldon'), ('troop', 'home')]
    

One note about how `BigramCollocationFinder` works, it doesn't use the strings internally.


```python
def bigramPrinting(count, wordsTuple, total):
    print("The first word is:  {}\nThe second word is: {}".format(*wordsTuple))
    #Returns None so all the tuples are considered to have the same rank

print(whBigrams.nbest(bigramPrinting, 10))
```

    The first word is:  5
    The second word is: 5
    The first word is:  3
    The second word is: 20
    The first word is:  2
    The second word is: 2
    The first word is:  16
    The second word is: 40
    The first word is:  28
    The second word is: 4
    The first word is:  32
    The second word is: 12
    The first word is:  4
    The second word is: 6
    The first word is:  3
    The second word is: 6
    The first word is:  7
    The second word is: 3
    The first word is:  40
    The second word is: 13
    The first word is:  16
    The second word is: 15
    The first word is:  30
    The second word is: 6
    The first word is:  32
    The second word is: 9
    The first word is:  2
    The second word is: 22
    The first word is:  2
    The second word is: 6
    The first word is:  5
    The second word is: 4
    The first word is:  16
    The second word is: 16
    The first word is:  30
    The second word is: 30
    The first word is:  8
    The second word is: 13
    The first word is:  20
    The second word is: 13
    The first word is:  13
    The second word is: 3
    The first word is:  20
    The second word is: 20
    The first word is:  30
    The second word is: 2
    The first word is:  8
    The second word is: 40
    The first word is:  28
    The second word is: 12
    The first word is:  2
    The second word is: 5
    The first word is:  32
    The second word is: 15
    The first word is:  6
    The second word is: 16
    The first word is:  5
    The second word is: 4
    The first word is:  22
    The second word is: 3
    The first word is:  15
    The second word is: 40
    The first word is:  3
    The second word is: 6
    The first word is:  3
    The second word is: 5
    The first word is:  8
    The second word is: 12
    The first word is:  15
    The second word is: 5
    The first word is:  14
    The second word is: 6
    The first word is:  4
    The second word is: 8
    The first word is:  2
    The second word is: 6
    The first word is:  12
    The second word is: 4
    The first word is:  3
    The second word is: 2
    The first word is:  3
    The second word is: 8
    The first word is:  40
    The second word is: 3
    The first word is:  40
    The second word is: 2
    The first word is:  16
    The second word is: 2
    The first word is:  5
    The second word is: 9
    The first word is:  14
    The second word is: 3
    The first word is:  4
    The second word is: 8
    The first word is:  15
    The second word is: 30
    The first word is:  3
    The second word is: 2
    The first word is:  5
    The second word is: 16
    The first word is:  4
    The second word is: 3
    The first word is:  28
    The second word is: 3
    The first word is:  13
    The second word is: 6
    The first word is:  6
    The second word is: 6
    The first word is:  2
    The second word is: 4
    The first word is:  13
    The second word is: 32
    The first word is:  16
    The second word is: 4
    The first word is:  6
    The second word is: 8
    The first word is:  2
    The second word is: 2
    The first word is:  8
    The second word is: 12
    The first word is:  6
    The second word is: 4
    The first word is:  6
    The second word is: 2
    The first word is:  22
    The second word is: 9
    The first word is:  3
    The second word is: 32
    The first word is:  28
    The second word is: 6
    The first word is:  8
    The second word is: 3
    The first word is:  5
    The second word is: 4
    The first word is:  8
    The second word is: 3
    The first word is:  14
    The second word is: 6
    The first word is:  16
    The second word is: 7
    The first word is:  6
    The second word is: 24
    The first word is:  16
    The second word is: 30
    The first word is:  2
    The second word is: 20
    The first word is:  2
    The second word is: 6
    The first word is:  9
    The second word is: 40
    The first word is:  12
    The second word is: 12
    The first word is:  32
    The second word is: 40
    The first word is:  24
    The second word is: 13
    The first word is:  6
    The second word is: 6
    The first word is:  40
    The second word is: 2
    The first word is:  40
    The second word is: 16
    The first word is:  16
    The second word is: 2
    The first word is:  9
    The second word is: 2
    The first word is:  13
    The second word is: 28
    The first word is:  2
    The second word is: 2
    The first word is:  22
    The second word is: 3
    The first word is:  24
    The second word is: 5
    The first word is:  32
    The second word is: 3
    The first word is:  14
    The second word is: 2
    The first word is:  12
    The second word is: 14
    The first word is:  3
    The second word is: 24
    The first word is:  3
    The second word is: 5
    The first word is:  3
    The second word is: 2
    The first word is:  6
    The second word is: 5
    The first word is:  8
    The second word is: 22
    The first word is:  3
    The second word is: 20
    The first word is:  9
    The second word is: 24
    The first word is:  3
    The second word is: 3
    The first word is:  4
    The second word is: 8
    The first word is:  2
    The second word is: 2
    The first word is:  4
    The second word is: 30
    The first word is:  2
    The second word is: 3
    The first word is:  40
    The second word is: 11
    The first word is:  12
    The second word is: 40
    The first word is:  6
    The second word is: 13
    The first word is:  3
    The second word is: 8
    The first word is:  28
    The second word is: 4
    The first word is:  30
    The second word is: 2
    The first word is:  5
    The second word is: 6
    The first word is:  13
    The second word is: 28
    The first word is:  6
    The second word is: 32
    The first word is:  14
    The second word is: 12
    The first word is:  2
    The second word is: 2
    The first word is:  12
    The second word is: 9
    The first word is:  7
    The second word is: 5
    The first word is:  3
    The second word is: 5
    The first word is:  2
    The second word is: 14
    The first word is:  2
    The second word is: 5
    The first word is:  15
    The second word is: 2
    The first word is:  5
    The second word is: 13
    The first word is:  28
    The second word is: 2
    The first word is:  7
    The second word is: 5
    The first word is:  30
    The second word is: 13
    The first word is:  5
    The second word is: 6
    The first word is:  5
    The second word is: 3
    The first word is:  40
    The second word is: 2
    The first word is:  28
    The second word is: 2
    The first word is:  16
    The second word is: 8
    The first word is:  3
    The second word is: 3
    The first word is:  22
    The second word is: 24
    The first word is:  24
    The second word is: 32
    The first word is:  4
    The second word is: 32
    []
    

The words are each given numeric IDs and there is a dictionary that maps the IDs to the words they represent, this is a common performance optimization.

Two words can appear together by chance. Recall from  Manning and Schütze's textbook that a t-value can be computed for each bigram to see how significant the association is. You may also want to try computing the chi square and likelihood ratio statisitcs. 


```python
bigram_measures = nltk.collocations.BigramAssocMeasures()
whBigrams.score_ngrams(bigram_measures.student_t)[:40]
```




    [(('rhode', 'island'), 4.427392223583876),
     (('presid', 'bush'), 4.136521610402945),
     (('stem', 'cell'), 3.443306607943331),
     (('sheldon', 'whitehous'), 3.3947849244896755),
     (('whitehous', 'said'), 2.893279973319993),
     (('bush', 'administr'), 2.5720113688566872),
     (('unit', 'state'), 2.433560094631012),
     (('american', 'peopl'), 2.409052943627678),
     (('bring', 'troop'), 2.405376870977178),
     (('troop', 'home'), 2.3857711501745116),
     (('senat', 'sheldon'), 2.3808697199738447),
     (('cell', 'research'), 2.2119063625353097),
     (('jack', 'reed'), 1.9919959979989994),
     (('come', 'home'), 1.9837418709354677),
     (('consul', 'provid'), 1.9579789894947475),
     (('said', 'today'), 1.9399699849924963),
     (('state', 'senat'), 1.9089544772386193),
     (('whitehous', 'also'), 1.8899449724862432),
     (('american', 'troop'), 1.8679339669834918),
     (('honor', 'societi'), 1.7294514316695746),
     (('youth', 'program'), 1.7277185144033727),
     (('potenti', 'close'), 1.7262744166815374),
     (('budget', 'resolut'), 1.7251191385040694),
     (('epw', 'committe'), 1.7251191385040694),
     (('chang', 'cours'), 1.7248303189597025),
     (('year', 'ago'), 1.7135663567293897),
     (('new', 'direct'), 1.7077899658420497),
     (('senat', 'youth'), 1.7077899658420497),
     (('reed', 'sheldon'), 1.7043241313096456),
     (('provid', 'consul'), 1.6835291241152222),
     (('senat', 'budget'), 1.6835291241152222),
     (('direct', 'iraq'), 1.6765974550504141),
     (('member', 'senat'), 1.6269204934192907),
     (('home', 'iraq'), 1.611901877112207),
     (('whitehous', 'member'), 1.581864644498039),
     (('said', 'whitehous'), 1.5472062991739997),
     (('district', 'columbia'), 1.4127986413502103),
     (('humpti', 'dumpti'), 1.4127986413502103),
     (('jose', 'socrat'), 1.4127986413502103),
     (('prime', 'minist'), 1.4127986413502103)]



Exercise: In Manning and Schütze's textbook, there is a section (section 5.3.2) on how to use the t-test to find words whose co-occurance patterns that best distinguish two words. Can you implement that? For instance, can you tell what words come after "America" a lot but not so often after "Iraq"? 

# KL Divergence

If we want to compare across the different corpus one of the places to start is Kullback-Leibler divergence, which computes the relative entropy between two distributions. 

Recall that given two discrete probability distributions $P$ and $Q$, the Kullback-Leibler divergence from $Q$ to $P$ is defined as:

$D_{\mathrm{KL}}(P\|Q) = \sum_i P(i) \, \log\frac{P(i)}{Q(i)}$.

The [scipy.stats.entropy()](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.entropy.html) function does the calculation for you, which takes in two arrays of probabilities and computes the KL divergence. Note that the KL divergence is in general not commutative, i.e. $D_{\mathrm{KL}}(P\|Q) \neq D_{\mathrm{KL}}(Q\|P)$ .

Also note that the KL divernce is the sum of elementwise divergences. Scipy provides [scipy.special.kl_div()](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.kl_div.html#scipy-special-kl-div) which calculates elementwise divergences for you.

To do this we will need to create the arrays, lets compare the Whitehouse releases with the Kennedy releases. First we have to download them and load them into a DataFrame.


```python
kenReleases = getGithubFiles('https://api.github.com/repos/lintool/GrimmerSenatePressReleases/contents/raw/Kennedy', maxFiles = 10)
kenReleases[:5]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>download_url</th>
      <th>html_url</th>
      <th>name</th>
      <th>path</th>
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>https://raw.githubusercontent.com/lintool/Grim...</td>
      <td>https://github.com/lintool/GrimmerSenatePressR...</td>
      <td>01Apr2005Kennedy14.txt</td>
      <td>raw/Kennedy/01Apr2005Kennedy14.txt</td>
      <td>FOR IMMEDIATE RELEASE   FOR IMMEDIATE...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>https://raw.githubusercontent.com/lintool/Grim...</td>
      <td>https://github.com/lintool/GrimmerSenatePressR...</td>
      <td>01Aug2005Kennedy12.txt</td>
      <td>raw/Kennedy/01Aug2005Kennedy12.txt</td>
      <td>FOR IMMEDIATE RELEASE   FOR IMMEDIATE...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>https://raw.githubusercontent.com/lintool/Grim...</td>
      <td>https://github.com/lintool/GrimmerSenatePressR...</td>
      <td>01Aug2006Kennedy10.txt</td>
      <td>raw/Kennedy/01Aug2006Kennedy10.txt</td>
      <td>FOR IMMEDIATE RELEASE  FOR IMMEDIATE ...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>https://raw.githubusercontent.com/lintool/Grim...</td>
      <td>https://github.com/lintool/GrimmerSenatePressR...</td>
      <td>01Aug2006Kennedy11.txt</td>
      <td>raw/Kennedy/01Aug2006Kennedy11.txt</td>
      <td>FOR IMMEDIATE RELEASE  FOR IMMEDIATE ...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>https://raw.githubusercontent.com/lintool/Grim...</td>
      <td>https://github.com/lintool/GrimmerSenatePressR...</td>
      <td>01Aug2006Kennedy12.txt</td>
      <td>raw/Kennedy/01Aug2006Kennedy12.txt</td>
      <td>FOR IMMEDIATE RELEASE  FOR IMMEDIATE ...</td>
    </tr>
  </tbody>
</table>
</div>



Then we can tokenize, stem and remove stop words, like we did for the Whitehouse releases


```python
kenReleases['tokenized_text'] = kenReleases['text'].apply(lambda x: nltk.word_tokenize(x))
kenReleases['normalized_tokens'] = kenReleases['tokenized_text'].apply(lambda x: normlizeTokens(x, stemmer = snowball))
```

Now we need to compare the two collection of words and remove those not found in both and assign the remaining ones indices.


```python
whWords = set(whReleases['normalized_tokens'].sum())
kenWords = set(kenReleases['normalized_tokens'].sum())

#Change & to | if you want to keep all words
overlapWords = whWords & kenWords

overlapWordsDict = {word: index for index, word in enumerate(overlapWords)}
overlapWordsDict['student']
```




    103



Now can count the occurrences of each these words in the corpora and create our arrays. Note, we don't have to use numpy arrays, we could just use a list, but the arrays are faster so we should get in the habit of using them.


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




    1.0



We can now compute the KL divergence. Pay attention to the asymmetry. Use [the Jensen–Shannon divergence](https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence) if you want symmetry.


```python
wh_kenDivergence = scipy.stats.entropy(whProbArray, kenProbArray)
print (wh_kenDivergence)
ken_whDivergence = scipy.stats.entropy(kenProbArray, whProbArray)
print (ken_whDivergence)
```

    0.617641967748
    0.587708565719
    

Then, we can do the elementwise calculation and see which words best distinguish the two corpora.


```python
wh_kenDivergence_ew = scipy.special.kl_div(whProbArray, kenProbArray)
kl_df = pandas.DataFrame(list(overlapWordsDict.keys()), columns = ['word'], index = list(overlapWordsDict.values()))
kl_df = kl_df.sort_index()
kl_df['elementwise divergence'] = wh_kenDivergence_ew
kl_df[:10]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>word</th>
      <th>elementwise divergence</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>scientif</td>
      <td>0.000305</td>
    </tr>
    <tr>
      <th>1</th>
      <td>expect</td>
      <td>0.000063</td>
    </tr>
    <tr>
      <th>2</th>
      <td>forward</td>
      <td>0.000127</td>
    </tr>
    <tr>
      <th>3</th>
      <td>chang</td>
      <td>0.001450</td>
    </tr>
    <tr>
      <th>4</th>
      <td>school</td>
      <td>0.006421</td>
    </tr>
    <tr>
      <th>5</th>
      <td>billion</td>
      <td>0.005358</td>
    </tr>
    <tr>
      <th>6</th>
      <td>nation</td>
      <td>0.003586</td>
    </tr>
    <tr>
      <th>7</th>
      <td>learn</td>
      <td>0.000063</td>
    </tr>
    <tr>
      <th>8</th>
      <td>afford</td>
      <td>0.000063</td>
    </tr>
    <tr>
      <th>9</th>
      <td>april</td>
      <td>0.002023</td>
    </tr>
  </tbody>
</table>
</div>




```python
kl_df.sort_values(by='elementwise divergence', ascending=False)[:10]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>word</th>
      <th>elementwise divergence</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>210</th>
      <td>iraq</td>
      <td>0.086929</td>
    </tr>
    <tr>
      <th>178</th>
      <td>bush</td>
      <td>0.042354</td>
    </tr>
    <tr>
      <th>94</th>
      <td>stem</td>
      <td>0.021884</td>
    </tr>
    <tr>
      <th>251</th>
      <td>law</td>
      <td>0.020547</td>
    </tr>
    <tr>
      <th>123</th>
      <td>protect</td>
      <td>0.019358</td>
    </tr>
    <tr>
      <th>331</th>
      <td>american</td>
      <td>0.018154</td>
    </tr>
    <tr>
      <th>238</th>
      <td>depart</td>
      <td>0.017650</td>
    </tr>
    <tr>
      <th>334</th>
      <td>bring</td>
      <td>0.014128</td>
    </tr>
    <tr>
      <th>113</th>
      <td>member</td>
      <td>0.012429</td>
    </tr>
    <tr>
      <th>149</th>
      <td>war</td>
      <td>0.011739</td>
    </tr>
  </tbody>
</table>
</div>



# Let's Do a Fun Example

Let's apply what we learned today to the guterberg texts in nltk and see if we can detect any patterns among them. 

First, let's transform every text into normalized tokens. Note that in this first step, no stopword is used. 


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
        if word in stop_words:
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
    p = df.ix[:,1]
    q = df.ix[:,0]
    D_kl = scipy.stats.entropy(p, q)
    return D_kl

def kl_divergence(corpus1, corpus2):
    freqP = nltk.FreqDist(corpus1)
    P = pandas.DataFrame(list(freqP.values()), columns = ['frequency'], index = list(freqP.keys()))
    freqQ = nltk.FreqDist(corpus2)
    Q = pandas.DataFrame(list(freqQ.values()), columns = ['frequency'], index = list(freqQ.keys()))
    return Divergence(P, Q)
```

Calculate the KL divergence for each pair of corpora, turn the results into a matrix, and visualize the matrix as a heatmap.


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


![png](2-Corpus-Linguistics_files/2-Corpus-Linguistics_112_0.png)


See that works by the same author have the lowest within-group KL divergeces. 

To reveal more patterns, let's do a multidimensional scaling on the matrix.


```python
from sklearn import manifold
mds = manifold.MDS()
pos = mds.fit(M).embedding_
x = pos[:,0]
y = pos[:,1]
fig, ax = plt.subplots(figsize = (6,6))
plt.plot(x, y, ' ')
for i, txt in enumerate(fileids):
    ax.annotate(txt, (x[i],y[i]))
```

    C:\Users\Shilin\Anaconda2\envs\py3\lib\site-packages\sklearn\manifold\mds.py:396: UserWarning: The MDS API has changed. ``fit`` now constructs an dissimilarity matrix from data. To use a custom dissimilarity matrix, set ``dissimilarity='precomputed'``.
      warnings.warn("The MDS API has changed. ``fit`` now constructs an"
    


![png](2-Corpus-Linguistics_files/2-Corpus-Linguistics_114_1.png)


Do you see any patterns in the image shown above? Does it make sense?

We may just want to focus on the distrbutions of stopwords or non-stopwords. Let's do the analysis again first for stopwords and then for non-stopwords.


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


![png](2-Corpus-Linguistics_files/2-Corpus-Linguistics_116_0.png)



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


![png](2-Corpus-Linguistics_files/2-Corpus-Linguistics_117_0.png)


Which analysis distinguishes the authors better?
