#This is how the reddit data were obtained
import praw
import pandas

targets = ['talesfromtechsupport', 'badroommates', 'weeabootales', 'relationships']

def getTopN(n, name, reddit):
    sub = reddit.subreddit(name)
    subIter = sub.top('all', limit=None)

    dfDict = {
        'text' : [],
        'score' : [],
        'author' : [],
        'title' : [],
        'url' : [],
        'over_18' : [],
        'subreddit' : [],
    }
    for i in range(n):
        try:
            post = next(subIter)
            while post.stickied or post.media is not None or 'reddit.com/r/{}'.format(name) not in post.url:
                post = next(subIter)
        except StopIteration:
            break
        print("Getting {}: {}".format(name, i))
        dfDict['text'].append(post.selftext.replace('\n', ' '))
        dfDict['score'].append(post.score)
        try:
            dfDict['author'].append(post.author.name)
        except AttributeError:
            dfDict['author'].append('[deleted]')
        dfDict['title'].append(post.title)
        dfDict['url'].append(post.url)
        dfDict['over_18'].append(post.over_18)
        dfDict['subreddit'].append(post.subreddit.title)
    return pandas.DataFrame(dfDict)


def main():
    reddit = praw.Reddit(client_id='IQ5IzVYxoglVaA',
        client_secret='gosgCqSEc0Q7sCJRac79kXLIMxo',
        password='qJAD7GtKj605j4wz',
        user_agent='soci 40133 exmples',
        username='soci40133')

    df = pandas.DataFrame()
    for target in targets:
        df = df.append(getTopN(400, target, reddit), ignore_index=True)
        df.to_csv('redditDat.csv')

if __name__ == '__main__':
    main()
