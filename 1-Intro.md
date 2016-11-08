# Layout

+ opening
    + contents
    + files used
    + packages used
+ Scraping
    + Images
        + OCR
    + raw text
    + html
    + PDFs
    + word doc etc.
+ spidering
    + wikipedia
    + APIs
        + REST
        + tumblr
            + Newly registered consumers are rate limited to 250 requests per hour, and 5,000 requests per day. If your application requires more requests for either of these periods, please use the 'Request rate limit removal' link on an app above.
            + Reid McIlroy-Young OAuth
                + Consumer Key: TgqpubaBeckUPRHWUCTHIe2DzGYyZ0hXYFenh2tiyZMGv874h8
                + Secret Key:  GTXHKip2c8TJyMz9A2iRhrV1cx03FSaSaznXGoVvCW2Fx5lyCv
+ reading files
    + encodings
    + unicode
+ filtering
+ data structures
    + pandas


# Week 1 - Intro

Intro stuff ...

For this notebook we will be using the following packages

```python
#All these packages need to be installed from pip
import requests #for http requests
import bs4 #called 'BeautifulSoup', an html parser
import pandas #gives us DataFrames
import docx #reading MS doc files, install as `python-docx`

#Stuff for pdfs
import pdfminer.pdfinterp
import pdfminer.converter
import pdfminer.layout
import pdfminer.pdfpage

#These come with Python
import re #for regexs
import urllib.parse #For joining urls
import io #for making http requests look like files
import json #For Tumblr api responses
```

We will also be working on the following files/urls

```python
wikipedia_base_url = 'https://en.wikipedia.org'
wikipedia_content_analysis = 'https://en.wikipedia.org/wiki/Content_analysis'
content_analysis_save = 'wikipedia_content_analysis.html'
example_text_file = 'sometextfile.txt'
information_extraction_pdf = 'https://web.stanford.edu/~jurafsky/slp3/20.pdf'
```

# Scraping

Before we can start analyzing content we need to obtain it. Sometimes it will be provided to us before hand, but often we will need to download it. As a starting example we will attempt to download the wikipedia page on content analysis. The page is located at [https://en.wikipedia.org/wiki/Content_analysis](https://en.wikipedia.org/wiki/Content_analysis) so lets start with that.

We can do this by making an HTTP GET request to that url, a GET request is simply a request to the server to provide the contents given by some url. The other request we will be using in this class is called a POST request and requests the server to take some content we provide. While the Python standard library does have the ability do make GET requests we will be using the [_requests_](http://docs.python-requests.org/en/master/) package as it is _'the only Non-GMO HTTP library for Python'_, also it provides a nicer interface.

```python
#wikipedia_content_analysis = 'https://en.wikipedia.org/wiki/Content_analysis'
requests.get('https://en.wikipedia.org/wiki/Content_analysis')
```

`'Response [200]'` means the server responded with what we asked for. If you get another number (e.g. 404) it likely means there was some kind of error, these codes are called HTTP response codes and a list of them can be found [here](https://en.wikipedia.org/wiki/List_of_HTTP_status_codes). The response object contains all the data the server sent including the website's contents and the HTTP header. We are interested in the contents which we can access with the `.text` attribute.

```python
wikiContentRequest = requests.get('https://en.wikipedia.org/wiki/Content_analysis')
print(wikiContentRequest.text[:1000])
```

This is not what we were looking for, because it is the start of the HTML that makes up the website. This is HTML and is meant to be read by computers. Luckily we have a computer to parse it for us. To do the parsing we will use [_Beautiful Soup_](https://www.crummy.com/software/BeautifulSoup/) which is a better parser than the one in the standard library.

```python
wikiContentSoup = bs4.BeautifulSoup(wikiContentRequest.text, 'html.parser')
print(wikiContentSoup.text[:200])
```

This is better but there's a bunch of random whitespace and we have way more than just the text of the article. This is because what we requested is the whole webpage, not just the text for the article.

We need to extract only the text we care about, in order to do this we will need to inspect the html. One way to do this is to simply go to the website with a browser and use its inspection or view source tool, but if there is javascript or other dynamic loading occurring on the page it is very likely that what Python receives is not what you will see. So we will need to view what Python receives. To do this we can save the html `requests` obtained.

```python
#content_analysis_save = 'wikipedia_content_analysis.html'

with open(content_analysis_save, mode='w', encoding='utf-8') as f:
    f.write(wikiContentRequest.text)
```

Now lets open the file (`wikipedia_content_analysis.html`) we just created with a web browser. It should look sort of like the original but missing all the images and formatting.

As there is very little standardization on structuring webpages figuring out how best to extract what you want is an art. Looking at this page it looks like all the main textual content is within `<p>`(paragraph) tags inside the `<body>` tag.

```python
contentPTags = wikiContentSoup.body.findAll('p')
for pTag in contentPTags[:3]:
    print(pTag.text)
```

We now have all the text from the page, split up by paragraph. If we wanted to get the section headers or references as well it would require a bit more work, but is doable.

There is one more thing we might want to do before sending this text to be processed, remove the references indicators (`[2]`, `[3]` , etc). To do this we can use a short regular expression (regex).

```python
contentParagraphs = []
for pTag in contentPTags:
    #strings starting with r are raw so their \'s are not modifier characters
    #If we didn't start with r the string would be: '\\[\\d+\\]'
    contentParagraphs.append(re.sub(r'\[\d+\]', '', pTag.text))

#convert to a DataFrame
contentParagraphsDF = pandas.DataFrame({'paragraph-text' : contentParagraphs})
print(contentParagraphsDF)
```

Now we have a `DataFrame` of all the relevant text from the page ready to be processed

If you are not familiar with regex, it is a way of specifying searches of text. A regex engine takes in the search pattern, in the above case `'\[\d+\]'` and some string, the paragraph texts. Then it reads the input string one character at a time checking if it matches the search. For example the regex `'\d'` matches number characters.

```python
findNumber = r'\d'
regexResults = re.search(findNumber, 'not a number, not a number, numbers 2134567890, not a number')
regexResults
```

In Python the regex package (`re`) usually returns `Match` objects (you can have multiple pattern hits in a a single `Match`), to get the string that matched our pattern we can use the `.group()` method, and as we want the first one will will ask for the 0'th group.

```python
print(regexResults.group(0))
```

That gives us the first number, if we wanted the whole block of numbers we can add a wildcard `'+'` which requests 1 or more instances of the proceeding character.

```python
findNumbers = r'\d+'
regexResults = re.search(findNumbers, 'not a number, not a number, numbers 2134567890, not a number')
print(regexResults.group(0))
```

Now we have the whole block of numbers, there are a huge number of special characters in regex, for the full description of Python's implementation look at the [re docs](https://docs.python.org/3/library/re.html) there is also a short [tutorial](https://docs.python.org/3/howto/regex.html#regex-howto).

# Spidering

What if we want to to get a bunch of different pages from wikipedia. We would need to get the url of each of the pages we want, usually we will want pages that are linked to by other pages, so we will need to parse pages and find the links. Right now we will be getting all the links in the body of the content analysis page.

To do this we will need to find all the `<a>` (anchor) tags with `href`s (hyperlink references) inside of `<p>` tags. `href` can have many [different](http://stackoverflow.com/questions/4855168/what-is-href-and-why-is-it-used) [forms](https://en.wikipedia.org/wiki/Hyperlink#Hyperlinks_in_HTML) so dealing with them can be tricky generally though you be extracting from it absolute or relative links. An absolute link is one you can follow with out any modification, while a relative link has a base url that you are then appending. Wikipedia uses relative urls for its internal links so below is an example of dealing with them.

```python
#wikipedia_base_url = 'https://en.wikipedia.org'

otherPAgeURLS = []
#We also want to know where the links come from so we also will get:
#the paragraph number
#the word the link is in
for paragraphNum, pTag in enumerate(contentPTags):
    #we only want hrefs that link to wiki pages
    tagLinks = pTag.findAll('a', href=re.compile('/wiki/'), class_=False)
    for aTag in tagLinks:
        #We need to extract the url from the <a> tag
        relurl = aTag.get('href')
        linkText = aTag.text
        #wikipedia_base_url is the base we can use the urllib joining function to merge them
        #Giving a nice structured tupe like this means we can use tuple expansion later
        otherPAgeURLS.append((
            urllib.parse.urljoin(wikipedia_base_url, relurl),
            paragraphNum,
            linkText,
        ))
print(otherPAgeURLS[:10])
```

We will be adding these new texts to our DataFrame `contentParagraphsDF` so we will need to add 2 more columns to keep track of paragraph numbers and sources.

```python
contentParagraphsDF['source'] = [wikipedia_content_analysis] * len(contentParagraphsDF['paragraph-text'])
contentParagraphsDF['paragraph-number'] = range(len(contentParagraphsDF['paragraph-text']))
contentParagraphsDF['source-paragraph-number'] = [None] * len(contentParagraphsDF['paragraph-text'])
contentParagraphsDF['source-paragraph-text'] = [None] * len(contentParagraphsDF['paragraph-text'])

contentParagraphsDF
```

Then we can define a function to parse each linked page and add its text to our DataFrame.

```python
def getTextFromWikiPage(targetURL, sourceParNum, sourceText):
    #Make a dict to store data before adding it to the DataFrame
    parsDict = {'source' : [], 'paragraph-number' : [], 'paragraph-text' : [], 'source-paragraph-number' : [],  'source-paragraph-text' : []}
    #Now we get the page
    r = requests.get(targetURL)
    soup = bs4.BeautifulSoup(r.text, 'html.parser')
    #enumerating gives use the paragraph number
    for parNum, pTag in enumerate(soup.body.findAll('p')):
        #same regex as before
        parsDict['paragraph-text'].append(re.sub(r'\[\d+\]', '', pTag.text))
        parsDict['paragraph-number'].append(parNum)
        parsDict['source'].append(targetURL)
        parsDict['source-paragraph-number'].append(sourceParNum)
        parsDict['source-paragraph-text'].append(sourceText)
    return pandas.DataFrame(parsDict)
```

And run it on our list of link tags

```python
for urlTuple in otherPAgeURLS[:3]:
    #ignore_index means the indices will not be reset after each append
    contentParagraphsDF = contentParagraphsDF.append(getTextFromWikiPage(*urlTuple),ignore_index=True)
contentParagraphsDF
```
## Tumblr API

Generally website owners do not like you scraping their sites. Scraping if done badly can act like a DOS attack so you should be careful about how often you make calls to a site. Some site though want automated tools to access their data, so they create [application programming interface (APIs)](https://en.wikipedia.org/wiki/Application_programming_interface). An API specifies a procedure for an application (or script) to access their data. Often this is though a [representational state transfer (REST)](https://en.wikipedia.org/wiki/Representational_state_transfer) web service, which just means if you make correctly formatted HTTP requests they will return nicely formatted data.

A nice example for us to study is [Tumblr](https://www.tumblr.com), they have a [simple RESTful API](https://www.tumblr.com/docs/en/api/v1) that allows you to read posts without any complicated html parsing.

We can get the first 20 posts from a blog by making an http GET request to `'http://{blog}.tumblr.com/api/read/json'`, were `{blog}` is the name of the target blog. Lets try and get the posts from [http://lolcats-lol-cat.tumblr.com/](http://lolcats-lol-cat.tumblr.com/) (Note the blog says at the top 'One hour one pic lolcats', but the canonical name that Tumblr uses is in the URL 'lolcats-lol-cat').

```python
tumblrAPItarget = 'http://{}.tumblr.com/api/read/json'

r = requests.get(tumblrAPItarget.format('lolcats-lol-cat'))

print(r.text[:1000])
```

This might not look very good, but it has a lot fewer angle braces than html, which is nice. What we have is [JSON](https://en.wikipedia.org/wiki/JSON) a 'human readable' text based data transmission format based on javascript. Luckily, we can convert it to a python `dict`.

```python
#We need to load only the stuff between the curly braces
d = json.loads(r.text[len('var tumblr_api_read = '):-2])
print(d.keys())
print(len(d['posts']))
```

If we read the [API specification](https://www.tumblr.com/docs/en/api/v1), we will see there are a lot of things we can get if, we add things to our GET request. First we can get posts by their id number, lets get post `146020177084`.

```python
r = requests.get(tumblrAPItarget.format('lolcats-lol-cat'), params = {'id' : 146020177084})
d = json.loads(r.text[len('var tumblr_api_read = '):-2])
d['posts'][0].keys()
d['posts'][0]['photo-url-1280']

with open('lolcat.gif', 'wb') as f:
    gifRequest = requests.get(d['posts'][0]['photo-url-1280'], stream = True)
    f.write(gifRequest.content)
```

<img src='lolcat.gif'>

Such beauty, now we could get the text from all the posts as well as some metadata, like the post date, caption or the tags. But, we could instead get the links to all the images.

```python
#Putting a max incase the blog has millions of images
#The given max will be rounded up to the nearest multiple of 50
def tumblrImageScrape(blogName, maxImages = 200):
    #Restating this here so the function isn't dependent on any external variables
    tumblrAPItarget = 'http://{}.tumblr.com/api/read/json'

    #There are a bunch of possible locations for the photo url
    possiblePhotoSuffixes = [1280, 500, 400, 250, 100]

    #These are the pieces of information we will be gathering,
    #at the end we will convert this to a DataFrame.
    #There are a few other ones we could get like the captions
    #you can read the Tumblr documentation to learn how to get them
    #https://www.tumblr.com/docs/en/api/v1
    postsData = {
        'id' : [],
        'photo-url' : [],
        'date' : [],
        'tags' : [],
        'photo-type' : []
    }

    #Tumblr limits us to a max of 50 posts per request
    for requestNum in range(maxImages // 50):
        requestParams = {
            'start' : requestNum * 50,
            'num' : 50,
            'type' : 'photo'
        }
        r = requests.get(tumblrAPItarget.format(blogName), params = requestParams)
        requestDict = json.loads(r.text[len('var tumblr_api_read = '):-2])
        for postDict in requestDict['posts']:
            #We are dealing with uncleaned data, we can't trust it.
            #Specifically, not all posts are guaranteed to have the fields we want
            try:
                postsData['id'].append(postDict['id'])
                postsData['date'].append(postDict['date'])
                postsData['tags'].append(postDict['tags'])
            except KeyError as e:
                raise KeyError("Post {} from {} is missing: {}".format(postDict['id'], blogName, e))

            foundSuffix = False
            for suffix in possiblePhotoSuffixes:
                try:
                    photoURL = postDict['photo-url-{}'.format(suffix)]
                    postsData['photo-url'].append(photoURL)
                    postsData['photo-type'].append(photoURL.split('.')[-1])
                    foundSuffix = True
                    break
                except KeyError:
                    pass
            if not foundSuffix:
                #Make sure your error messages are useful
                #You will be one of the users
                raise KeyError("Post {} from {} is missing a photo url".format(postDict['id'], blogName))

    return pandas.DataFrame(postsData)
tumblrImageScrape('lolcats-lol-cat', 50)
```

Now we have the urls of a bunch of images and can run OCR on them.

## Stuff for the v2 Tumblr API

probably unnecessary

+ Consumer Key: TgqpubaBeckUPRHWUCTHIe2DzGYyZ0hXYFenh2tiyZMGv874h8
+ Secret Key:  GTXHKip2c8TJyMz9A2iRhrV1cx03FSaSaznXGoVvCW2Fx5lyCv

[https://www.tumblr.com/docs/en/api/v2](https://www.tumblr.com/docs/en/api/v2)

```python
#get the most recent 20 posts
target = 'http://procedural-generation.tumblr.com/api/read/json'

r = requests.get(target)

d = json.loads(r.text[len('var tumblr_api_read = '):-2])

#get a specific post

r = requests.get(target, params = {'id' : '152256405098'})

d = json.loads(r.text[len('var tumblr_api_read = '):-2])
```


```python
'''
REQUEST_TOKEN_URL = 'http://www.tumblr.com/oauth/request_token'
AUTHORIZATION_URL = 'http://www.tumblr.com/oauth/authorize'
ACCESS_TOKEN_URL = 'http://www.tumblr.com/oauth/access_token'

tumblrKey = 'TgqpubaBeckUPRHWUCTHIe2DzGYyZ0hXYFenh2tiyZMGv874h8'
tumblrSecret = 'GTXHKip2c8TJyMz9A2iRhrV1cx03FSaSaznXGoVvCW2Fx5lyCv'
consumer = oauth2.Consumer(tumblrKey, tumblrSecret)
client = oauth2.Client(consumer)

#The token is part of content, which is provided as a binary query string
responseDict, content = client.request(REQUEST_TOKEN_URL, "GET")
content = urllib.parse.parse_qs(content.decode('utf-8'))
oathToken = content['oauth_token'][0]
oathTokenSecret = content['oauth_token_secret'][0]

print("The OATH token is: {}\nThe OATH token secret is: {}".format(oathToken, oathTokenSecret))
'''
```

## OCR

Something about subprocess

`pytesseract` works but requires tesseract binary

# Files

What if the text we want isn't on a webpage? There are a many other sources of text available.

## Raw text

The most basic form of storing text is as a _raw text_ document. Source code (`.py`, `.r`, etc) is usually raw text as are text files (`.txt`) and many other things. Opening an unknown file with a text editor is often a great way of learning what the file is.

We can create a text file with the `open()` function

```python
#example_text_file = 'sometextfile.txt'
#stringToWrite = 'A line\nAnother line\nA line with a few unusual symbols \u2421 \u241B \u20A0 \u20A1 \u20A2 \u20A3 \u0D60\n'
stringToWrite = 'A line\nAnother line\nA line with a few unusual symbols ␡ ␛ ₠ ₡ ₢ ₣ ൠ\n'

with open(example_text_file, 'w', encoding='utf-8') as f:
    f.write(stringToWrite)
```

Notice though the `encoding='utf-8'` argument, the encoding specifies how we map the bits from the file to the glyphs (and whitespace characters like tab (`'\t'`) or newline (`'\n'`)) on the screen. When dealing only with latin letters, arabic numerals and the other symbols on America keyboards you usually do not have to worry about encodings as the ones used today are backwards compatible with [ASCII](https://en.wikipedia.org/wiki/ASCII) which gives the binary representation of 128 characters.

Some people though use other characters. To solve this there is [Unicode](https://en.wikipedia.org/wiki/Unicode) which gives numbers to symbols, e.g. 041 is `'A'` and 03A3 is `'Σ'` (number starting with 0 indicates they are hexadecimal), often non-ASCII characters are called Unicode characters. Unfortunately there are many ways used to map combinations of bits to Unicode symbols. The ones you are likely to encounter are called by Python _utf-8_, _utf-16_ and _latin-1_. _utf-8_ is the standard for Linux and Mac OS while both _utf-16_ and _latin-1_ are used by windows. If you use the wrong encoding characters can appear wrong, sometimes change in number or Python could raise an exception. Lets see what happens when we open the file we just created with different encodings.

```python
with open(example_text_file, encoding='utf-8') as f:
    print("This is with the correct encoding:")
    print(f.read())

with open(example_text_file, encoding='latin-1') as f:
    print("This is with the wrong encoding:")
    print(f.read())
```
Notice that with _latin-1_ the unicode characters are mixed up and there are too many of them. You need to keep in mind encoding when obtaining text files, as determining the encoding can sometime be a lot of work.

## PDF

Another common way text will be stored is in a PDF file. First we will download a pdf in Python. To do that lets grab a chapter from
_Speech and Language Processing_, chapter 20 is on Information Extraction which seems apt. It is stored as a pdf at [https://web.stanford.edu/~jurafsky/slp3/20.pdf](https://web.stanford.edu/~jurafsky/slp3/20.pdf).

```python
#information_extraction_pdf = 'https://web.stanford.edu/~jurafsky/slp3/20.pdf'

infoExtractionRequest = requests.get(information_extraction_pdf, stream=True)
print(infoExtractionRequest.text[:1000])
```

It says `'pdf'`, so thats a good sign, the rest though looks like we are having issues with an encoding. The random characters are not though caused by our encoding being wrong they are cause by there not being an encoding for those parts at all. PDFs are nominally binary files, meaning there are sections of binary that are specific to pdf and nothing else so you need something that knows about pdf to read them. To do that we will be using [`PyPDF2`](https://github.com/mstamy2/PyPDF2) which is a PDF processing library for Python 3.

**NOTE** maybe use `PyPDF2` or `slate`

Because PDFs are a very complicated file format pdfminer requires a large amount of boilerplate code to extract text, we have written a function that takes in an open PDF file and returns the text so you don't have to.

```python
def readPDF(pdfFile):
    #Make utf-8 explicit
    #Based on code from http://stackoverflow.com/a/20905381/4955164
    rsrcmgr = pdfminer.pdfinterp.PDFResourceManager()
    retstr = io.StringIO()
    layoutParams = pdfminer.layout.LAParams()
    device = pdfminer.converter.TextConverter(rsrcmgr, retstr, laparams = layoutParams)
    #We need a device and an interpreter
    interpreter = pdfminer.pdfinterp.PDFPageInterpreter(rsrcmgr, device)
    password = ''
    maxpages = 0
    caching = True
    pagenos=set()
    for page in pdfminer.pdfpage.PDFPage.get_pages(pdfFile, pagenos, maxpages=maxpages, password=password,caching=caching, check_extractable=True):
        interpreter.process_page(page)
    device.close()
    returnedString = retstr.getvalue()
    retstr.close()
    return returnedString
```

But first we need to take the response object and convert it into a 'file like' object so that pdfminer can read it. To do this we will use `io`'s `BytesIO`.

```python
infoExtractionBytes = io.BytesIO(infoExtractionRequest.content)
```

Now we can give it to pdfminer.

```python
print(readPDF(infoExtractionBytes)[:1000])
```

From here we can either look at the full text or fiddle with our PDF reader and get more information about individual blocks of text.

## Word Docs

*NOTE* The package is called python-docx

The other type of document you are likely to encounter is the `.docx`, these are actually a version of [XML](https://en.wikipedia.org/wiki/Office_Open_XML), just like HTML, and like HTML we will use a specialized parser.

For this class we will use [`python-docx`](https://python-docx.readthedocs.io/en/latest/) which provides a nice simple interface for reading `.docx` files

```python
r = requests.get('https://github.com/xiaow2/persp-analysis/raw/02772bc5baf4044ba6410170ca740f14cd6155d5/assignments/short%20paper%201.docx', stream=True)
d = docx.Document(io.BytesIO(r.content))
for paragraph in d.paragraphs[:7]:
    print(paragraph.text)
```
