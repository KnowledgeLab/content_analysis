# Week 1 - Intro

Intro stuff ...

# Scraping

Before we can start analyzing content we need to obtain it. Sometimes it will be provided to us before hand, but often we will need to download it. As a starting example we will attempt to download the MACS 3000 syllabus from github. The course page is located at [github.com/UC-MACSS/persp-analysis](https://github.com/UC-MACSS/persp-analysis) so lets start with that.

We can do this by making an HTTP GET request to that url, a GET request is simply a request to the server to provide the contents given by some url. The other request we will be using in this class is called a POST request and requests the server to take some content we provide. While the Python standard library does have the ability do make GET requests we will be using the [_requests_](http://docs.python-requests.org/en/master/) package as it is _'the only Non-GMO HTTP library for Python'_, also it provides a nicer interface.

```python
import requests
#requests.get('https://github.com/UC-MACSS/persp-analysis')
```

`'Response [200]'` means the server responded with what we asked for. If you get another number (e.g. 404) it likely means there was some kind of error, these codes are called HTTP response codes and a list of them can be found [here](https://en.wikipedia.org/wiki/List_of_HTTP_status_codes). The response object contains all the data the server sent including the website's contents and the HTTP header. We are interested in the contents which we can access with the `.text` attribute.

```python
r = requests.get('https://github.com/UC-MACSS/persp-analysis')
print(r.text[:1000])
```

This is a bunch of nonsense, what it is, is the start of the HTML that makes up the website. This is a version of XML and is meant to be read by computers. Luckily we have a computer to parse it for us. To do the parsing we will use [_Beautiful Soup_](https://www.crummy.com/software/BeautifulSoup/) which is a better xml parser than the one in the standard library.

```python
import bs4
soup = bs4.BeautifulSoup(r.text, 'html.parser')
print(soup.text[:200])
```

This is better but there's a bunch of random whitespace and we have way more than just the `README.md` file. This is because the text we requested is for the whole web page, not just for the syllabus.
