These are some basic utility functions we will need.

```python
import collections
import os
import random
import re
import glob
import pandas
import requests
import json
import sklearn
import sklearn.feature_extraction.text
import sklearn.decomposition
from sklearn import preprocessing, linear_model
from sklearn.metrics import confusion_matrix
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import seaborn
import numpy as np
import matplotlib.pyplot as plt
from scipy import interp
%matplotlib inline
```

Some functions for splitting data

```python
def split_data(data, prob):
    """split data into fractions [prob, 1 - prob]"""
    results = [], []
    for row in data:
        results[0 if random.random() < prob else 1].append(row)
    return results

def train_test_split(x, y, test_pct):
    data = zip(x, y)                              # pair corresponding values ; zip combines lists to tuples
    train, test = split_data(data, 1 - test_pct)  # split the dataset of pairs
    x_train, y_train = zip(*train)                # magical un-zip trick
    x_test, y_test = zip(*test)
    return x_train, x_test, y_train, y_test
```

# NAIVE BAYES

## Naive Bayes from Scratch

First, let's build a Naive Bayes classifier from scratch. This example drawn
from *Data Science from Scratch* by Joel Grus.

### Mathematical Preliminaries

Recall the key independence assumption of Naive Bayes: $P(X_1 = x_1,\dots,X_n =
x_n\,|\,S) = P(X_1 = x_1\,|\,S)\times \dots
    \times P(X_n = x_n\,|\,S)$

To be concrete, let's assume we are building a spam filter.

Given a vocabulary $w_1,\dots,w_n$, let $X_i$ be the event "message contains
$w_i$." $X_i = x_i, x_i \in \{0,1\}$.

$S$ is the event "message is spam" and $\neg S$ is the event "message is not
spam."

According to Bayes' Theorem

$P(S\,|\,X_1 = x_1,\dots, X_n = x_n) = \frac{P(X_1 = x_1,\dots, X_n =
x_n\,|\,S)P(S)}{P(X_1 = x_1,\dots, X_n = x_n)} = \frac{P(X_1 = x_1,\dots, X_n =
x_n\,|\,S)P(S)}{P(X_1 = x_1,\dots, X_n = x_n\,|\,S)P(S)\, + \,P(X_1 = x_1,\dots,
X_n = x_n\,|\,\neg S)P(\neg S)}$

We further assume that we have no knowledge of the prior probability of spam; so
$P(S) = P(\neg S) = 0.5$ (this is the principle of indifference)

With this simplification, $P(S\,|\,X_1 = x_1,\dots, X_n = x_n) = \frac{P(X_1 =
x_1,\dots, X_n = x_n\,|\,S)}{P(X_1 = x_1,\dots, X_n = x_n\,|\,S)\, +\, P(X_1 =
x_1,\dots, X_n = x_n\,|\,\neg S)}$

Now we make the Naive Bayes assumption: $P(X_1 = x_1,\dots,X_n = x_n\,|\,S) =
P(X_1 = x_1\,|\,S)\times \dots
    \times P(X_n = x_n\,|\,S)$

We can estimate $P(X_i = x_i\,|\,S)$ by computing the fraction of spam messages
containing the word $i$, e.g., Obamacare.

Smoothing: $P(X_i\,|\,S) = \frac{(k + \textrm{number of spams containing}\,
w_i)}{(2k + \textrm{number of spams})}$



### Now we are going to code this up

First we will "tokenize" our input, i.e., turn text into presence/absence of
words.

```python
def tokenize(message): #Note: no stemming
    message = message.lower() # convert to lowercase
    all_words = re.findall("[a-z0-9']+", message) # use re to extract the words
    return set(all_words) # remove duplicates because we are creating a set
```

Now we need to count the number of times each word shows up in spam and non-spam

```python
def count_words(training_set):
    """training set consists of pairs (message, is_spam)"""
    counts = collections.defaultdict(lambda: [0, 0]) #This is a tricky bit of Python magic, makes a dictionary initialized to [0,0]
    for message, is_spam in training_set: #Here I step through the training set.
        for word in tokenize(message):
            counts[word][0 if is_spam else 1] += 1
    return counts
```

We need a function to convert these counts into (smoothed) probabilities

```python
def word_probabilities(counts, total_spams, total_non_spams, k=0.5): #What is the smoothing parameter?
    """turn the word_counts into a list of triplets
    w, p(w | spam) and p(w | ~spam)"""
    return [(w,
             (spam + k) / (total_spams + 2 * k),
             (non_spam + k) / (total_non_spams + 2 * k)) #This uses list comprehension.
             for w, (spam, non_spam) in counts.items()]  #Replace .iteritems with .items for Python3
```

Now we need to come up with a way to compute the spam probability for a message,
given word probabilities. With the Naive Bayes assumption, we *would* be
multiplying together a bunch of probabilities. This is bad (underflow) so we
compute:

$p_1 *\dots*p_n = \exp(\, \log(p_1) + \dots + \log(p_n)\,)$; recall $\log(ab) =
\log a + \log b$ and $\exp(\, \log x \,) = x$

Thank you, John Napier (1550-1617)

```python
def spam_probability(word_probs, message):
    message_words = tokenize(message)
    log_prob_if_spam = log_prob_if_not_spam = 0.0 #Initialize; we are working with log probs to deal with underflow.

    for word, prob_if_spam, prob_if_not_spam in word_probs: #We iterate over all possible words we've observed

        # for each word in the message,
        # add the log probability of seeing it
        if word in message_words:
            log_prob_if_spam += np.log(prob_if_spam) #This is prob of seeing word if spam
            log_prob_if_not_spam += np.log(prob_if_not_spam) #This is prob of seeing word if not spam

        # for each word that's not in the message
        # add the log probability of _not_ seeing it
        else:
            log_prob_if_spam += np.log(1.0 - prob_if_spam)
            log_prob_if_not_spam += np.log(1.0 - prob_if_not_spam)
    P = 1/(1 + np.exp(log_prob_if_not_spam - log_prob_if_spam))
    #prob_if_spam = math.exp(log_prob_if_spam) #Compute numerator
    #prob_if_not_spam = math.exp(log_prob_if_not_spam)
    #return prob_if_spam / (prob_if_spam + prob_if_not_spam) #Compute whole thing and return
    return P
```

Think: how would this change if $P(S) \neq P(\neg S)$

Now we write a class (this is a Python term) for our Naive Bayes Classifier

```python
class NaiveBayesClassifier:

    def __init__(self, k=0.5):
        self.k = k
        self.word_probs = [] #Initializes word_probs as an empty list, sets a default smoothing parameters

    def train(self, training_set): #Operates on the training_set

        # count spam and non-spam messages: first step of training
        num_spams = len([is_spam
                         for message, is_spam in training_set
                         if is_spam]) #This is also list comprehension
        num_non_spams = len(training_set) - num_spams

        # run training data through our "pipeline"
        word_counts = count_words(training_set)
        self.word_probs = word_probabilities(word_counts,
                                             num_spams,
                                             num_non_spams,
                                             self.k) #"Train" classifier

    def classify(self, message):
        return spam_probability(self.word_probs, message) #Now we have all we need to classify a message

```

We'll need a special utility function for reading in the data

```python
def get_subject_data(path):
    data = []
    # regex for stripping out the leading "Subject:" and any spaces after it
    subject_regex = re.compile(r"^Subject:\s+")
    # glob.glob returns every filename that matches the wildcarded path
    for fn in glob.glob(path):
        is_spam = "ham" not in fn

        #with open(fn,'r') as file: #PYTHON 3 USERS: COMMENT THIS OUT AND USE LINE BELOW
        with open(fn,'r',errors='surrogateescape') as file:
            for line in file:
                if line.startswith("Subject:"):
                    subject = subject_regex.sub("", line).strip()
                    data.append((subject, is_spam))

    return data
```

Grab data from: https://spamassassin.apache.org/publiccorpus/

Get the ones with 20021010 prefixes and put them into the data folder under the
current working directory.

```python
path = os.getcwd() + '/data/*/*/*'
```

```python
data = get_subject_data(path)
```

```python
len(data) #How many
```

```python
data[1000] #Let's look
```

To train the model, we'll need to split our data into training & test

```python
random.seed(0) #This is important for replicability
train_data,test_data = split_data(data,0.75) #Recall what the second argument does here
```

```python
classifier = NaiveBayesClassifier() #Create an instance of our classifier
```

```python
classifier.train(train_data) #Train the classifier on the training data
```

```python
len(train_data)
```

Some simple evaluation:

```python
# triplets (subject, actual is_spam, predicted spam probability)
classified = [(subject, is_spam, classifier.classify(subject))
              for subject, is_spam in test_data]

# assume that spam_probability > 0.5 corresponds to spam prediction # and count the combinations of (actual is_spam, predicted is_spam)
counts = collections.Counter((is_spam, spam_probability > 0.5) # (actual, predicted)
                     for _, is_spam, spam_probability in classified)
```

```python
counts #Let's see how we did!
```

Precision:

```python
precision = counts[(True,True)]/(counts[(False,True)]+counts[(True,True)]) #True positives over all positive predictions
print(precision)
```

Recall:

```python
recall = counts[(True,True)]/(counts[(True,False)]+counts[(True,True)])#what fraction of positives identified
print(recall)
```

F-measure:

```python
f_measure = 2 * (precision * recall)/(precision + recall)
print (f_measure)
```

Let's look at how the emails are classified.

```python
df_classification = pandas.DataFrame(classified, columns = ['email', 'spam', 'posterior'])
df_classification = df_classification.round(3)
```

```python
df_classification[:10]
```

The ROC curve:

```python
x, y, _ = sklearn.metrics.roc_curve(df_classification['spam'], df_classification['posterior'])
roc_auc = sklearn.metrics.auc(x,y)
```

```python
plt.figure()
plt.plot(x,y, color = 'darkorange', lw = 2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic curve')
plt.legend(loc="lower right")
plt.show()
```

**Think**: what will happen if I change my spam threshold from 0.5?

**Think**: what decision rule are we using, when we assign the class label
$\hat{y} = S$ when $P(S\,|\,X_1\dots X_n) = \frac{P(S)\prod_i P(X_i =
x_i|S)}{P(\textbf{X})} > 0.5$

**Think**: what will happen if I split the data differently (e.g., less training
data, more testing data). Try it!

We can also find words that lead to a high probability of spam (using Bayes'
Theorem):

```python
def p_spam_given_word(word_prob):
    """uses bayes's theorem to compute p(spam | message contains word)"""
    # word_prob is one of the triplets produced by word_probabilities

    word, prob_if_spam, prob_if_not_spam = word_prob
    return prob_if_spam / (prob_if_spam + prob_if_not_spam)
```

```python
words = sorted(classifier.word_probs,key=p_spam_given_word)
```

```python
spammiest_words = words[-15:]
hammiest_words = words[:15]
```

```python
spammiest_words
```

```python
hammiest_words
```

# Let' Try it on Our Clinton Obama Corpora

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

    #Check for rate limiting
    if r.status_code != 200:
        raise RuntimeError("Github didn't like your request, you have probably been rate limited.")
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

'''
#Uncomment this to download your own data

targetSenator = 'Obama'# = ['Voinovich', 'Obama', 'Whitehouse', 'Snowe', 'Rockefeller', 'Murkowski', 'McCain', 'Kyl', 'Baucus', 'Frist']

ObamaReleases = pandas.DataFrame()

print("Fetching {}'s data".format(targetSenator))
targetDF = getGithubFiles('https://api.github.com/repos/lintool/GrimmerSenatePressReleases/contents/raw/{}'.format(targetSenator), maxFiles = 2000)
targetDF['targetSenator'] = targetSenator
ObamaReleases = ObamaReleases.append(targetDF, ignore_index = True)

targetSenator = 'Clinton'# = ['Voinovich', 'Obama', 'Whitehouse', 'Snowe', 'Rockefeller', 'Murkowski', 'McCain', 'Kyl', 'Baucus', 'Frist']

ClintonReleases = pandas.DataFrame()

print("Fetching {}'s data".format(targetSenator))
targetDF = getGithubFiles('https://api.github.com/repos/lintool/GrimmerSenatePressReleases/contents/raw/{}'.format(targetSenator), maxFiles = 2000)
targetDF['targetSenator'] = targetSenator
ClintonReleases = ClintonReleases.append(targetDF, ignore_index = True)

ObamaClintonReleases = ObamaReleases.append(ClintonReleases, ignore_index = True)
ObamaClintonReleases.to_csv("data/ObamaClintonReleases.csv")

'''



#senReleasesTraining = pandas.read_csv("data/senReleasesTraining.csv")

```

```python
ObamaClintonReleases = pandas.read_csv("data/ObamaClintonReleases.csv")
ObamaClintonReleases = ObamaClintonReleases.dropna(axis=0, how='any')
```

Let's turn the 'targetSenator' column into a binary variable.

```python
def IsObama(targetSenator):
    if targetSenator == 'Obama':
        isObama = True
    elif targetSenator == 'Clinton':
        isObama = False
    return(isObama)
```

```python
ObamaClintonReleases['IsObama'] = ObamaClintonReleases['targetSenator'].apply(lambda x: IsObama(x))
```

Let's split the data into training data and test data.

```python
data = list(zip(ObamaClintonReleases['text'], ObamaClintonReleases['IsObama']))
random.seed(0) #This is important for replicability
train_data,test_data = split_data(data,0.75)
```

```python
print (len(train_data))
print (len(test_data))
```

First, let's try with a logistic regression.
Turn the training dataset into a tf-idf matrix

```python
train_data_df = pandas.DataFrame(train_data, columns = ['email', 'true label'])
```

```python
TFVectorizer = sklearn.feature_extraction.text.TfidfVectorizer(max_df=100, min_df=2, stop_words='english', norm='l2')
TFVects = TFVectorizer.fit_transform(train_data_df['email'])
```

```python
TFVects.shape
```

In a regression, we cannot have more variables than cases. So, we need to first
do a dimension reduction. Let's do a PCA. You have already seen it in week 3.
Here we are less less concerned about visualization, so all principal components
are calculated.

```python
PCA = sklearn.decomposition.PCA
pca = PCA().fit(TFVects.toarray())
reduced_data = pca.transform(TFVects.toarray())
```

Visualization:

```python
colordict = {
True: 'red',
False: 'blue',
    }
colors = [colordict[c] for c in train_data_df['true label']]
fig = plt.figure(figsize = (5,3))
ax = fig.add_subplot(111)
ax.set_frame_on(False)
ax.scatter(reduced_data[:, 0], reduced_data[:, 1], color = colors, alpha = 0.5, label = colors)
plt.xticks(())
plt.yticks(())
plt.title('True Classes, Training Set')
plt.show()
```

PCA cannot distinguish Clinton from Obama well. Let's do a screeplot and see how
many dimensions we need.

```python
n = TFVects.shape[0]
fig = plt.figure(figsize=(12,5))
ax1 = fig.add_subplot(121)
eigen_vals = np.arange(n) + 1
ax1.plot(eigen_vals, pca.explained_variance_ratio_, 'ro-', linewidth=2)
ax1.set_title('Scree Plot')
ax1.set_xlabel('Principal Component')
ax1.set_ylabel('Proportion of Explained Variance')

ax2 = fig.add_subplot(122)
eigen_vals = np.arange(20) + 1
ax2.plot(eigen_vals, pca.explained_variance_ratio_[:20], 'ro-', linewidth=2)
ax2.set_title('Scree Plot (First 20 Principal Components)')
ax2.set_xlabel('Principal Component')
ax2.set_ylabel('Proportion of Explained Variance')
plt.show()
```

Let's choose the first 10 pricipal components as our covariates.

```python
X = reduced_data[:, :10]
```
Transform our predictor variable.
```python
Y = np.array([int(label) for label in train_data_df['true label']])
```

Fit a logistic regression.

```python
logistic = linear_model.LogisticRegression()
logistic.fit(X, Y)
```

Let's see how the logistic regression performs on our training dataset. The mean
accuracy is only about 66%.

```python
logistic.score(X,Y)
```

How does it perform on the testing dataset?

```python
test_data_df = pandas.DataFrame(test_data, columns = ['email', 'true label'])
TFVects_test = TFVectorizer.transform(test_data_df['email'])
reduced_data_test = pca.transform(TFVects_test.toarray())
X_test = reduced_data_test[:, :10]
Y_test = np.array([int(label) for label in test_data_df['true label']])
logistic.score(X_test, Y_test)
```

Slightly Poorer. How about using more dimensions?

```python
X = reduced_data[:, :40]
logistic.fit(X, Y)
X_test = reduced_data_test[:, :40]
print(logistic.score(X,Y))
print(logistic.score(X_test, Y_test))
```

```python
X = reduced_data[:, :100]
logistic.fit(X, Y)
X_test = reduced_data_test[:, :100]
print(logistic.score(X,Y))
print(logistic.score(X_test, Y_test))
```

```python
X = reduced_data[:, :200]
logistic.fit(X, Y)
X_test = reduced_data_test[:, :200]
print(logistic.score(X,Y))
print(logistic.score(X_test, Y_test))
```

```python
X = reduced_data[:, :400]
logistic.fit(X, Y)
X_test = reduced_data_test[:, :400]
print(logistic.score(X,Y))
print(logistic.score(X_test, Y_test))
```

Increasing the number of covariates would overfit our data, and it seems that
using a logistic regression, our prediction accuracy is at best about 80%.

Let's try with Naive Bayes.

```python
classifier = NaiveBayesClassifier()
classifier.train(train_data)
```

Let's evaluate the result on the test data.

```python
classified = [(subject, is_spam, classifier.classify(subject))
              for subject, is_spam in test_data]

counts = collections.Counter((is_spam, spam_probability > 0.5)
                     for _, is_spam, spam_probability in classified)

counts
```

Precision:

```python
precision = counts[(True,True)]/(counts[(False,True)]+counts[(True,True)]) #True positives over all positive predictions
print(precision)
```

Recall:

```python
recall = counts[(True,True)]/(counts[(True,False)]+counts[(True,True)])#what fraction of positives identified
print(recall)
```

F-measure:

```python
f_measure = 2 * (precision * recall)/(precision + recall)
print (f_measure)
```

```python
df_classification = pandas.DataFrame(classified, columns = ['press release', 'is Obama', 'posterior probability'])
df_classification = df_classification.round(3)
```

```python
df_classification
```

Let's take a look at how well our posterior distribution is.

```python
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.grid(False)
ax1.set_frame_on(False)
df_classification[df_classification['is Obama'] == True]['posterior probability'].hist(alpha = 0.5, ax = ax1, bins = 10, label = 'Obama', color = 'red')
df_classification[df_classification['is Obama'] == False]['posterior probability'].hist(alpha = 0.5, ax = ax1, bins = 10, label = 'Clinton', color = 'blue')
ax1.set_xlim((0,1.1))
ax1.legend()
ax1.set_xlabel('posterior')
ax1.set_ylabel('counts')
plt.show()
```

The classification is suprisingly accurate.

```python
def p_obama_given_word(word_prob):
    """uses bayes's theorem to compute p(spam | message contains word)"""
    # word_prob is one of the triplets produced by word_probabilities

    word, prob_if_obama, prob_if_not_obama = word_prob
    return prob_if_obama / (prob_if_obama + prob_if_not_obama)
```

```python
words = sorted(classifier.word_probs,key=p_obama_given_word)
```

```python
Obama_words = words[-15:]
Clinton_words = words[:15]
```

```python
Obama_words
```

```python
Clinton_words
```

## Multinomial Naive Bayes

Well suited to text applications, this generating model assumes that the
features are generated by draws from a multinomial distribution (recall this
gives the probability to observe a particular pattern of counts across
features). Features might be, e.g., the count of various words in a text.

Let's use again the dataset we used in week 3.

```python
data = fetch_20newsgroups() #Free data to play with: documents from a newsgroup corpus.
data.target_names #Possible categories, i.e., the newsgroups
```

This dataset has a built in breakdown into training and testing sets. We can
pick specific categories, and pull the relevant training and testing sets.

```python
categories = ['talk.religion.misc', 'soc.religion.christian', 'sci.space', 'comp.graphics'] #Can change these of course
train = fetch_20newsgroups(subset='train', categories=categories)
test = fetch_20newsgroups(subset='test', categories=categories)
```

```python
train['target']
```

```python
len(train.data) #See how many training examples
```

```python
len(test.data) #Ditto for testing -- it's about 60% training, 40% testing
```

```python
print(train.data[5]) #Look at an example
#print(train.target_names[5])
```

We need to extract features from the text. We can use built-in feature
extraction to do so. We will use a tf-idf vectorizer, which converts the
document into a vector of words with tf-idf weights (term-frequency inverse-
document frequency). This gives high weight to words that show up a lot in a
given document but rarely across documents in the corpus (more distinctive).

We also take advantage of a cool feature of Scikit-Learn: we can make
pipelines...

```python
model = make_pipeline(TfidfVectorizer(max_df=100, min_df=2, stop_words='english', norm='l2'), MultinomialNB()) #This applies the vectorizer, then trains Multinomial NB
```

```python
model.fit(train.data,train.target) #Training syntax: feed the fit method the training data and the training targets
```

```python
labels = model.predict(test.data)
```

```python
labels.size
```

We can even use a confusion matrix!

```python
mat = confusion_matrix(test.target, labels)
seaborn.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
                xticklabels=train.target_names, yticklabels=train.target_names)
plt.xlabel('true label')
plt.ylabel('predicted label');
```

Let's get the precision, recall, and F-measure.

```python
print(sklearn.metrics.precision_score(test.target, labels, average = 'weighted')) #precision
print(sklearn.metrics.recall_score(test.target, labels, average = 'weighted')) #recall
print(sklearn.metrics.f1_score(test.target, labels, average = 'weighted')) #F-1 measure
```

Get the ROC curves.
First, we need to binarize the labels.

```python
lb = preprocessing.LabelBinarizer()
lb.fit(test.target)
print(lb.classes_)
print(lb.transform([1, 3]))
```

```python
y_test = lb.transform(test.target)
y_score = model.predict_proba(test.data)
```

```python
n_classes = 4

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = sklearn.metrics.roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = sklearn.metrics.auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = sklearn.metrics.roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = sklearn.metrics.auc(fpr["micro"], tpr["micro"])
```

```python
lw = 2
n_classes = 4
# Compute macro-average ROC curve and ROC area

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] =  sklearn.metrics.auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = ['aqua', 'darkorange', 'cornflowerblue', 'olive']
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()
```

We can also give the model a string and use the predict method to see if it can
assign it to a category. This might be the main point of a social science
application.

```python
def predict_category(s, train=train, model=model): #We just define a simple function here
    return train.target_names[model.predict([s])]
```

```python
#predict_category('rockets')
```

Try it yourself with your own strings!


**Playtime**: Try altering the categories! See if you can come up with ones that
are more/less likely to be confused. As an extension, you can use different
feature extractors, e.g., CountVectorizer. This just turns each document into a
vector of word counts.

## Now an a more relevant content analysis example

Now, let's add Bernie Sanders to our Obama Clinton dataset.

```python
'''
#Unmark this section if you want to fetch press releases of a nother senator.
targetSenator = 'Sanders'

SandersReleases = pandas.DataFrame()

print("Fetching {}'s data".format(targetSenator))
targetDF = getGithubFiles('https://api.github.com/repos/lintool/GrimmerSenatePressReleases/contents/raw/{}'.format(targetSenator), maxFiles = 2000)
targetDF['targetSenator'] = targetSenator
SandersReleases =SandersReleases.append(targetDF, ignore_index = True)

ObamaClintonSandersReleases = ObamaClintonReleases.append(SandersReleases, ignore_index = True)
ObamaClintonSandersReleases.to_csv("data/ObamaClintonSandersReleases.csv")
'''
```

```python
ObamaClintonSandersReleases = pandas.read_csv("data/ObamaClintonSandersReleases.csv")
ObamaClintonSandersReleases = ObamaClintonSandersReleases.dropna(axis=0, how='any')
```

Let's split the data into training data and test data.

```python
data = list(zip(ObamaClintonSandersReleases['text'], ObamaClintonSandersReleases['targetSenator']))
random.seed(0) #This is important for replicability
train_data,test_data = split_data(data,0.75)
```

```python
print (len(train_data))
print (len(test_data))
```

```python
train.data = [data[0] for data in train_data]
train.target = [data[1] for data in train_data]
test.data = [data[0] for data in test_data]
test.target = [data[1] for data in test_data]
```

```python
model.fit(train.data,train.target) #Training syntax: feed the fit method the training data and the training targets
```

```python
labels = model.predict(test.data)
```

```python
labels.size
```

```python
model.classes_
```

Confusion matrix:

```python
mat = confusion_matrix(test.target, labels)
seaborn.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
                xticklabels=['Obama', 'Clinton', 'Sanders'], yticklabels=['Obama', 'Clinton', 'Sanders'])
plt.xlabel('true label')
plt.ylabel('predicted label');
```

Let's get the precision, recall, and F-measure.

```python
print(sklearn.metrics.precision_score(test.target, labels, average = 'weighted')) #precision
print(sklearn.metrics.recall_score(test.target, labels, average = 'weighted')) #recall
print(sklearn.metrics.f1_score(test.target, labels, average = 'weighted')) #F-1 measure
```

Get the ROC curves.
First, we need to binarize the labels.

```python
lb = preprocessing.LabelBinarizer()
lb.fit(test.target)
print(lb.classes_)
print(lb.transform(['Clinton', 'Obama']))
```

```python
y_test = lb.transform(test.target)
y_score = model.predict_proba(test.data)
```

```python
n_classes = 3

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = sklearn.metrics.roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = sklearn.metrics.auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = sklearn.metrics.roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = sklearn.metrics.auc(fpr["micro"], tpr["micro"])
```

```python
n_classes = 3
# Compute macro-average ROC curve and ROC area

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] =  sklearn.metrics.auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = ['aqua', 'darkorange', 'cornflowerblue']
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()
```

We can also give the model a string and using the predict method see if it can
assign it to a category. This might be the main point of a social science
application.

```python
model.predict(['money'])
```

```python
model.predict(['Wall Street'])
```

```python
model.predict(['Chicago'])
```

# DECISION TREES

We are going to stick to Scikit-Learn here.

Decision trees can be used to predict both categorical/class labels (i.e.,
classification) and continuous labels (i.e., regression).

Let's create something to learn:

```python
from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=300, centers=4,
                random_state=0, cluster_std=1.0)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='rainbow');
```

Now we import our Decision Tree classifier from sklearn.tree (very familiar
syntax!) and fit it using the fit method.

```python
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(max_depth=4,random_state=0).fit(X,y)
```

To see what's going on visually with the classification, we can use this
(complex) visualizer.

```python
def visualize_classifier(model, X, y, Xmod, ymod, ax=None, cmap='rainbow'): #X and y are plotted; Xmod and ymod train
    ax = ax or plt.gca()
    # Plot the training points
    ax.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=cmap,
               clim=(y.min(), y.max()), zorder=3)
    ax.axis('tight')
    ax.axis('off')
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    # fit the estimator
    model.fit(Xmod, ymod)
    xx, yy = np.meshgrid(np.linspace(*xlim, num=200),
                         np.linspace(*ylim, num=200))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    # Create a color plot with the results
    n_classes = len(np.unique(y))
    contours = ax.contourf(xx, yy, Z, alpha=0.3,
                           levels=np.arange(n_classes + 1) - 0.5,
                           cmap=cmap, clim=(y.min(), y.max()),
                           zorder=1)
    ax.set(xlim=xlim, ylim=ylim)
```

```python
Xnew, ynew = make_blobs(n_samples=1000, centers=4,
                random_state=0, cluster_std=1.0)
```

```python
visualize_classifier(DecisionTreeClassifier(max_depth=4,random_state=0), Xnew, ynew, X, y) #We train on the full data
```

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score
accuracy_score(ynew,DecisionTreeClassifier(max_depth=10,random_state=0).fit(X,y).predict(Xnew))
```

```python
depthvec = []
scorevec = []
for i in range(1,20):
    tree2 = DecisionTreeClassifier(max_depth=i,random_state=0).fit(X,y)
    score = accuracy_score(ynew,tree2.predict(Xnew))
    depthvec.append(i)
    scorevec.append(score)
plt.scatter(depthvec,scorevec)
```

Near the cluster boundaries, the shape is pretty weird. Overfitting!

```python
from sklearn.cross_validation import train_test_split #Can use this to split the data
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.5,random_state=1) #test_size means a 50/50 split
print(len(X))
print(len(Xtrain))
print(len(Xtest))
```

With some abuse of notation, we can use the visualizer to plot first a
classifier trained on the "training" half of the data.

```python
visualize_classifier(DecisionTreeClassifier(max_depth=5,random_state=0), X, y, Xtrain, ytrain) #Train with half the data
```

Now we can train on the second half of the data--allegedly the test data--to see
how different training sets affect the decision boundaries.

```python
visualize_classifier(DecisionTreeClassifier(random_state=0), X, y, Xtest, ytest) #Train with other half of the data
```

Finally, we can get a sense of the performance by training on the training data,
but PLOTTING the test data

```python
visualize_classifier(DecisionTreeClassifier(), Xtest, ytest, Xtrain, ytrain) #Trains with train data, plots test data
```

Combining multiple overfitting estimators turns out to be a key idea in machine
learning. This is called **bagging** and is a type of **ensemble** method. The
idea is to make many randomized estimators--each can overfit, as decision trees
are wont to do--but then to combine them, ultimately producing a better
classification. A **random forest** is produced by bagging decision trees.

```python
from sklearn.tree import DecisionTreeClassifier  #Just in case
from sklearn.ensemble import BaggingClassifier #The bagging

tree = DecisionTreeClassifier(max_depth=10) #Create an instance of our decision tree classifier.

bag = BaggingClassifier(tree, n_estimators=100, max_samples=0.8, random_state=1) #Each tree uses up to 80% of the data
```

```python
#?BaggingClassifier #Learn more
```

```python
bag.fit(X,y) #Fit the bagged classifier
```

```python
Xnew, ynew = make_blobs(n_samples=10000, centers=4,
                random_state=0, cluster_std=1.0)
```

```python
visualize_classifier(bag,Xnew,ynew,X,y) #And visualize
#Remember we can give the full data as training data, as bag automatically splits and trains
```

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score
recall_score(ynew,bag.predict(Xnew),average='weighted')
```

# Let's try it on the Clinton Obama Sanders dataset.

```python
TFVects = TFVectorizer.fit_transform(train.data)
```

```python
tree = DecisionTreeClassifier(max_depth=4,random_state=0).fit(TFVects,train.target)
```

```python
TFVects_test = TFVectorizer.transform(test.data)
```

```python
labels = tree.predict(TFVects_test)
```

Confusion matrix:

```python
mat = confusion_matrix(test.target, labels)
seaborn.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
                xticklabels=['Obama', 'Clinton', 'Sanders'], yticklabels=['Obama', 'Clinton', 'Sanders'])
plt.xlabel('true label')
plt.ylabel('predicted label');
```

The precision, recall, and F-measure.

```python
print(sklearn.metrics.precision_score(test.target, labels, average = 'weighted')) #precision
print(sklearn.metrics.recall_score(test.target, labels, average = 'weighted')) #recall
print(sklearn.metrics.f1_score(test.target, labels, average = 'weighted')) #F-1 measure
```

Not really better than Multinomial Naive Bayes. Let's try a random forest.

```python
bag.fit(TFVects,train.target) #Fit the bagged classifier
```

```python
labels = bag.predict(TFVects_test)
```

```python
mat = confusion_matrix(test.target, labels)
seaborn.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
                xticklabels=['Obama', 'Clinton', 'Sanders'], yticklabels=['Obama', 'Clinton', 'Sanders'])
plt.xlabel('true label')
plt.ylabel('predicted label');
```

```python
print(sklearn.metrics.precision_score(test.target, labels, average = 'weighted')) #precision
print(sklearn.metrics.recall_score(test.target, labels, average = 'weighted')) #recall
print(sklearn.metrics.f1_score(test.target, labels, average = 'weighted')) #F-1 measure
```

The performance is better!

## Brief aside: Random forest regression (Optional)

First, let's create a challenging dataset.

```python
rng = np.random.RandomState(42)
x = 10 * rng.rand(200) #200 uniformly distributed random numbers between 0 and 10
def model(x, sigma=0.3):
    fast_oscillation = np.sin(5 * x)
    slow_oscillation = np.sin(0.5 * x)
    noise = sigma * rng.randn(len(x)) #Create 200 random numbers, normally distributed

    return slow_oscillation + fast_oscillation + noise
```

```python
y = model(x)
plt.errorbar(x, y, 0.3, fmt='o'); #Shows one std around point
#plt.scatter(x, y); #Plots the actual points
```

First we will try to learn this with something simple; a single decision tree,
but one that is a regressor, not a classifier.

```python
#First try to learn this with something simple --  a single decision tree regressor
from sklearn.tree import DecisionTreeRegressor

# Fit regression models
regr_1 = DecisionTreeRegressor(max_depth=2)
regr_2 = DecisionTreeRegressor(max_depth=5)
regr_3 = DecisionTreeRegressor(max_depth=10)
regr_1.fit(x[:,None], y)
regr_2.fit(x[:,None], y)
regr_3.fit(x[:,None], y)

# Predict
xfit = np.linspace(0, 10, 1000)
yfit_1 = regr_1.predict(xfit[:, None]) #xfit[:,None] is a work-around to pass a 1d feature matrix
yfit_2 = regr_2.predict(xfit[:, None])
yfit_3 = regr_3.predict(xfit[:, None])
```

Let's see how these two decision trees do... What do you think will happen as we
add depth?

```python
plt.errorbar(x, y, 0.3, fmt='o', alpha=0.5)
plt.plot(xfit, yfit_1, '-r'); #depth = 2
plt.plot(xfit,yfit_2,'-b'); #depth = 5
plt.plot(xfit,yfit_3,'-g'); #depth = 10
```

Or we can use a **random forest regressor**

```python
from sklearn.ensemble import RandomForestRegressor
forest = RandomForestRegressor(200)
forest.fit(x[:, None], y)

yfit = forest.predict(xfit[:, None]) #Predictions of the forest model
ytrue = model(xfit, sigma=0) #This is the underlying data, no noise
```

```python
plt.errorbar(x, y, 0.3, fmt='o', alpha=0.5)
plt.plot(xfit, yfit, '-r');
plt.plot(xfit, ytrue, '-k', alpha=0.5);
```

A beautiful fit of the underlying pattern... a lot of the noise has been washed
out. This is the power of bagging.

# K-Nearest Neighbors

Let's use newsgroup data again.

```python
categories = ['soc.religion.christian', 'sci.space', 'comp.graphics'] #Can change these of course
train = fetch_20newsgroups(subset='train', categories=categories)
test = fetch_20newsgroups(subset='test', categories=categories)
```

Let's visualize it using PCA.

```python
TFVectorizer = sklearn.feature_extraction.text.TfidfVectorizer(max_df=0.5, min_df=3, stop_words='english', norm='l2')
TFVects = TFVectorizer.fit_transform(train.data)
```

```python
TFVects.shape
```

```python
reduced_data = PCA(n_components = 2).fit_transform(TFVects.toarray())
```

```python
train.target_names
```

Visualization:

```python
colordict = {
0: 'red',
1: 'orange',
2: 'green',
    }
colors = [colordict[c] for c in train.target]
fig = plt.figure(figsize = (10,6))
ax = fig.add_subplot(111)
ax.set_frame_on(False)
ax.scatter(reduced_data[:, 0], reduced_data[:, 1], color = colors, alpha = 0.5, label = colors)
plt.xticks(())
plt.yticks(())
plt.title('True Classes, Training Set')
plt.show()
```

Let's initialize our k-nearest neighbors classifier.

```python
n_neighbors = 15
from sklearn import neighbors
clf = neighbors.KNeighborsClassifier(n_neighbors, weights="uniform")
```

```python
clf.fit(TFVects, train.target)
```

```python
TFVects_test = TFVectorizer.transform(test.data)
```

```python
labels = clf.predict(TFVects_test)
```

Confusion matrix:

```python
mat = confusion_matrix(test.target, labels)
seaborn.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
                xticklabels=categories, yticklabels=categories)
plt.xlabel('true label')
plt.ylabel('predicted label');
```

The precision, recall, and F-measure.

```python
print(sklearn.metrics.precision_score(test.target, labels, average = 'weighted')) #precision
print(sklearn.metrics.recall_score(test.target, labels, average = 'weighted')) #recall
print(sklearn.metrics.f1_score(test.target, labels, average = 'weighted')) #F-1 measure
```

# SVMs

Now we will examine Support Vector Machines, using the Cinton/Oboma data

```python
train_data_df[:5]
```

```python
TFVectorizer = sklearn.feature_extraction.text.TfidfVectorizer(max_df=0.5, min_df=3, stop_words='english', norm='l2')
TFVects = TFVectorizer.fit_transform(train_data_df['email'])
```

```python
train_data_df['tf'] = [np.array(v) for v in TFVects.todense()]
```

```python
clf = sklearn.svm.SVC()
clf
```

```python
train_data_df['true label'] = [int(l) for l in train_data_df['true label']]
```

```python
clf.fit(TFVects, train_data_df['true label'])
```

```python
clf.predict(TFVects[67])
```

```python
m.tolist

```

```python
for l in m:
    print(l[:1])
```

```python

```
