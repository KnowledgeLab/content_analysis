# Class Structure

1. Discussion
2. Notebook


# Weeks

1. Scraping
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
2. Corpus linguistics
    + Searching text for keywords
    + Distribution of terms
    + Correlation
    + Word frequencies
    + Conditional frequencies
    + Statistically significant collocations
    + Distinguishing or Important words and phrases (Wordls!)
        + tf-idf
    + POS-tagged words and phrases
    + Lemmatized words and phrases
        + stemmers
    + Dictionary-based annotations
    + divergences
        + kl
3. Clustering / Topic Models
    + packages
        + scikit-learn
        + gensim
    + Introduce ML
    + Training
        + Use sklearn dataset examples
    + Extraction
        + Probabilistic models
        + Deterministic models
        + stop lists
        + cleaning
        + Performance
        + [sklearn](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.cluster)
            + [Tfidf](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
            + [KMeans](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
            + Hierarchical
        + [gensim](http://radimrehurek.com/gensim/apiref.html)
            + [LDA](https://radimrehurek.com/gensim/models/ldamodel.html)
                + Expansions on LDA
            + [word2vec](https://radimrehurek.com/gensim/models/word2vec.html)
            + Doc2Vec
        + Implement our own models
            + regressions
4. Embedding / Spaces
    + Grouped with the above
6. Accuracy
7. Classification
    + Stanford classifiers
8. Semantic Networks
    + packages
        + networkx
        + nltk
        + BeautifulSoup
    + Tree structure
        + walk through a graph
    + Different parsers
9. Beyond Text
    + Image processing
    + Sounds
