# Week 1 - Intro

Intro stuff ...

# Scraping

Before we can start analyzing content we need to obtain it. Sometimes it will be provided to us before hand, but often we will need to download it. As a starting example we will attempt to download the MACS 3000 syllabus from github. The course page is located at [github.com/UC-MACSS/persp-analysis](https://github.com/UC-MACSS/persp-analysis) so lets start with that.

We can do this by making an HTTP GET request to that url, a GET request is simply a request to the server to provide the contents given by some url. The other request we will be using in this class is called a POST request and requests the server to take some content we provide. While the Python standard library does have the ability do make GET requests we will be using the _requests_ package as it provides a nicer interface.

```python
import requests
requests.get('https://github.com/UC-MACSS/persp-analysis')
```
