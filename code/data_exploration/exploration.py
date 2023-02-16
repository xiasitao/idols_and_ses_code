# %%
from pathlib import Path
SOURCE_PATH = Path(__file__).parent.resolve()
ASSET_PATH = SOURCE_PATH.joinpath('..', '..', 'assets').resolve()
BUILD_PATH = SOURCE_PATH.joinpath("..", "..", "build").resolve()

import pickle
import regex as re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import spacy
from nltk.text import Text
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from sklearn.feature_extraction.text import TfidfVectorizer


# %%
# Load data
articles = pd.read_pickle(BUILD_PATH / 'articles/articles.pkl')
articles_en = articles[articles.language_ml == 'en']
with open(BUILD_PATH / 'articles/corpora.pkl', 'rb') as file:
    corpora = pickle.load(file)
with open(BUILD_PATH / 'articles/sentence_tokens.pkl', 'rb') as file:
    sentence_tokens = pickle.load(file)
with open(BUILD_PATH / 'articles/word_statistic.pkl', 'rb') as file:
    word_statistics = pickle.load(file)


# %%
# Histograms
# plt.title('Sentence count')
# plt.hist(articles[articles.sentences < 200].sentences)
# plt.show()

# plt.title('Word count')
# plt.hist(articles[articles.terms < 2000].terms)
# plt.show()


# %%
# Most common sentences
sentence_tokens
sentence_tokens_frequency_en = FreqDist(sentence_tokens['en'])
sentence_tokens_frequency_en.most_common(200)
# %%
# Most significant words
tfidf_vectorizer = TfidfVectorizer()
tfidf_values = tfidf_vectorizer.fit_transform(articles_en['content_slim'])
tfidf_words = tfidf_vectorizer.get_feature_names_out()
# %%
tfidf_value_means = np.array(np.mean(tfidf_values, axis=0)).flatten()
tfidf = pd.Series(data=tfidf_value_means, index=tfidf_words)
tfidf = tfidf.sort_values(ascending=False)
# %%
tfidf.iloc[0:50]
# %%
tfidf.iloc[50:100]
# %%
