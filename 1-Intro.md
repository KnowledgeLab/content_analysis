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
import requests #http requests
import bs4 #called 'BeautifulSoup', a html parser
import re #for regexs
import pandas #DataFrames
import urllib.parse #For joining urls
```

We will also be working on the following files/urls

```python
wikipedia_base_url = 'https://en.wikipedia.org'
wikipedia_content_analysis = 'https://en.wikipedia.org/wiki/Content_analysis'
content_analysis_save = 'wikipedia_content_analysis.html'
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

There is one more thing we might want to do before sending this text to be processed, remove the references indicators (`[2]`, `[3]` , etc). To do this we can use a short regular expression.

```python
contentParagraphs = []
for pTag in contentPTags:
    contentParagraphs.append(re.sub(r'\[\d+\]', '', pTag.text))

#convert to a DataFrame
contentParagraphsDF = pandas.DataFrame({'paragraph-text' : contentParagraphs})
print(contentParagraphsDF)
```

Now we have a `DataFrame` of all the relevant text from the page ready to be processed

# Spidering

What if we want to to get a bunch of different pages from wikipedia. We would need to get the url of each of the pages we want, usually we will want pages that are linked to by other pages, so we will need to parse pages and find the links. Right now we will be getting all the links in the body of the content analysis page.

To do this we will need to find all the `<a>` (anchor) tags with `href`s inside of `<p>` tags.

```python
tagLinks = []
for pTag in contentPTags:
    #we only want hrefs that link to wiki pages
    tagLinks += pTag.findAll('a', href=re.compile('/wiki/'), class_=False)
print(tagLinks[:4])
```

We will be adding these new texts to our DataFrame `contentParagraphsDF` so we will need to add 2 more columns to keep track of paragraph numbers and sources.

```python
contentParagraphsDF['source'] = [wikipedia_content_analysis] * len(contentParagraphsDF['paragraph-text'])
contentParagraphsDF['paragraph-number'] = range(len(contentParagraphsDF['paragraph-text']))
contentParagraphsDF
```

Then we can define a function to parse each linked page and add its text to our DataFrame.

```python
#wikipedia_base_url = 'https://en.wikipedia.org'

def getTextFromWikiPage(linkTag):
    #We need to extract the url from the <a> tag
    relurl = linkTag.get('href')
    #The urls are relative so we need to prepend the wikipedia url
    #while both of them are strings using the specialized function means
    #badly formatted relurls will be fixed, if possible
    url = urllib.parse.urljoin(wikipedia_base_url, relurl)
    #Make a dict to store data before adding it to the DataFrame
    parsDict = {'source' : [], 'paragraph-number' : [], 'paragraph-text' : []}
    #Now we get the page
    r = requests.get(url)
    soup = bs4.BeautifulSoup(r.text, 'html.parser')
    #enumerating gives use the paragraph number
    for parNum, pTag in enumerate(soup.body.findAll('p')):
        #same regex as before
        parsDict['paragraph-text'].append(re.sub(r'\[\d+\]', '', pTag.text))
        parsDict['paragraph-number'].append(parNum)
        parsDict['source'].append(url)
    return pandas.DataFrame(parsDict)
```

And run it on our list of link tags

```python
for aTag in tagLinks[:5]:
    #ignore_index means the indices will not be reset after each append
    contentParagraphsDF = contentParagraphsDF.append(getTextFromWikiPage(aTag),ignore_index=True)
contentParagraphsDF
```
