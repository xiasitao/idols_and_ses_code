# %%
from pathlib import Path
SOURCE_PATH = Path(__file__).parent.resolve()
ASSET_PATH = SOURCE_PATH.joinpath('..', '..', 'assets').resolve()
BUILD_PATH = SOURCE_PATH.joinpath("..", "..", "build").resolve()

import pickle
import pandas as pd
from pandarallel import pandarallel
pandarallel.initialize()
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint

from gensim.corpora import Dictionary
from gensim.models import LdaModel
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

from multiprocessing import Pool
from threading import Lock


# %%
def filter_tokens(doc: list) -> list:
    nltk_tokens = word_tokenize(' '.join(doc))
    return [token[0] for token in pos_tag(nltk_tokens)
        if len(token[0]) > 1
        and token[0] != 've'
        and token[1] in ('FW', 'NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ')  # https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
    ]


def equilibrate_role_models(articles: pd.DataFrame, ses_data: pd.DataFrame) -> pd.DataFrame:
    """Equilibrate role models such that for each prevalent_ses the same amount of role models are present

    Args:
        articles (pd.DataFrame): all articles relevant for these role models
        ses_data (pd.DataFrame): substrate

    Returns:
        pd.DataFrame: substrate with equal amounts of role models for each prevalent ses
    """    
    ses_data = ses_data[ses_data.index.isin(articles['role_model'])]
    minimum_count = ses_data.groupby('prevalent_ses').count()['count'].min()
    ses_data = ses_data.sample(frac=1.0, random_state=42)
    ses_data['_group_index'] = ses_data.groupby('prevalent_ses').cumcount()
    ses_data = ses_data[ses_data['_group_index'] < minimum_count]
    ses_data = ses_data.drop('_group_index', axis=1)
    return ses_data


def train_lda_model(n_topics: int) -> tuple:
    """Train an LDA model on the corpus with a predefined number of topics.

    Args:
        n_topics (int): Number of topcis to generate.

    Returns:
        tuple: model, topics (10 most significant words for each topic)
    """    
    iterations = 500
    passes = 4
    model = LdaModel(
        corpus=corpus, id2word=dictionary.id2token, random_state=42,
        iterations=iterations, passes=passes,
        num_topics=n_topics,
    )
    topics = [[topic_entry[1] for topic_entry in topic[0][0:10]] for topic in model.top_topics(corpus)]
    return model, topics


def find_topic_and_entropy(model: LdaModel, doc: list) -> tuple:
    """Find the most probable topic and the topic distribution
    entropt for a document.

    Args:
        model (LdaModel): Topic model
        doc (list): Document as list of tokens

    Returns:
        tuple: topic, entropy
    """    
    topic_probabilities = np.array(model[dictionary.doc2bow(filter_tokens(doc))])
    topics = topic_probabilities[:, 0]
    probabilities = topic_probabilities[:, 1]
    topic = topics[probabilities.argmax()]
    entropy = -probabilities.dot(np.log(probabilities))
    return topic, entropy


 # %%
articles = pd.read_pickle(BUILD_PATH / 'articles/articles_balanced_50.pkl')
ses_scores = pd.read_pickle(BUILD_PATH / 'role_models/ses_scores_balanced.pkl')
articles_en = articles[articles.language_ml == 'en']
ses_scores = equilibrate_role_models(articles_en, ses_scores)
articles_en = articles_en.join(ses_scores[['average_ses', 'rank_weighted_ses', 'significance_weighted_ses', 'prevalent_ses']], on='role_model', how='inner')


# %%
articles_tokenized = articles_en['content_slim'].str.split(' ')
dictionary = Dictionary(articles_tokenized)
dictionary.filter_extremes(no_below=5, no_above=0.3)
dictionary[0]
corpus = [dictionary.doc2bow(filter_tokens(article)) for article in articles_tokenized]


# %%
# Run model
n_topics = 2
model, topics = train_lda_model(n_topics)
print(topics)

#%%
articles_tokenized.parallel_apply(lambda doc: pd.Series(find_topic_and_entropy(model, doc)))
# %%
# Article topic identification
articles_en[['topic', 'topic_entropy']] = articles_tokenized.parallel_apply(lambda doc: pd.Series(find_topic_and_entropy(model, doc)))
articles_en['topic'] = articles_en['topic'].astype(int)
topics_distribution_articles = articles_en[['topic', 'prevalent_ses', 'content']].groupby(['topic', 'prevalent_ses']).count()
topics_distribution_role_models = articles_en[['topic', 'prevalent_ses', 'role_model']].groupby(['topic', 'prevalent_ses']).nunique()


# %%
# Topic assignment
topic_names_en = {
    # 1: ['all articles'],  # bad
    # 2: ['people', 'sports & movies'],  # bad
    # 3: ['people', 'music', 'sports & movies'],  # bad
    # 4: ['sports', 'people-focused entertainment', 'music', 'movies'],  # very good
    # 5: ['sports', 'people', 'music', 'teams  & competition', 'movies'],  # ouhla
    # 6: ['sports', 'emotion', 'movies', 'music', 'series', 'sports'],  # not too bad
    # 7: ['sports', 'emotion', 'movies', 'music', 'teams &  competition', 'series', 'sports'],
    # 8: ['movies', 'emotion', 'sports', 'music', 'movies', 'teams & competition', 'series', 'prestige'],
    # 9: ['emotion', 'movies', 'music', 'sports', 'prestige', 'videos', 'teams & competition', 'competition', 'series'],
    # 10: ['emotion', 'movies', 'sports', 'music (entertainment)', 'prestige', 'videos', 'series', 'actors & roles', 'competition', 'music (meta)'],
}


# %%
# Distribution plotting
plt.figure(figsize=(10, 20))
plt.title(f'SES distribution within topics')
for i, topic_index in enumerate(topics_distribution_articles.index.get_level_values(0).unique()):
    ax = plt.subplot(len(topics_distribution_articles)//3+1, 3, i+1)
    ax.set_title(f'topic {topic_index}' if n_topics not in topic_names_en else topic_names_en[n_topics][i])
    this_topic_articles_distribution = topics_distribution_articles.loc[topic_index]
    this_topic_role_models_distribution = topics_distribution_role_models.loc[topic_index]
    w = 0.2
    ax.bar([0, 2*w], this_topic_articles_distribution['content'], width=w, label='role models')
    ax.bar([0+w/3, 2*w+w/3], this_topic_role_models_distribution['role_model'], width=w, label='articles')
    ax.set_xticks([0, 3*w], ['low', 'high'])
plt.legend()
plt.tight_layout()
plt.show()


# %%
# Hypertopic assignment
# general: emotion, people, competition, prestige
HT_GENERAL, HT_MUSIC, HT_MOVIES, HT_SPORTS = 'general', 'music', 'movies', 'sports'
hypertopics = [HT_GENERAL, HT_MOVIES, HT_MUSIC, HT_SPORTS]

hypertopic_assignment_matrix = pd.DataFrame(index=list(topic_names_en.keys()), columns=np.arange(0, np.max(list(topic_names_en.keys()))), data=-1)
hypertopic_assignment_matrix.loc[1, 0] = HT_GENERAL
hypertopic_assignment_matrix.loc[2, 0:1] = HT_GENERAL, HT_GENERAL
hypertopic_assignment_matrix.loc[3, 0:2] = HT_GENERAL, HT_MUSIC, HT_GENERAL
hypertopic_assignment_matrix.loc[4, 0:3] = HT_SPORTS, HT_GENERAL, HT_MUSIC, HT_MOVIES
hypertopic_assignment_matrix.loc[5, 0:4] = HT_SPORTS, HT_GENERAL, HT_MUSIC, HT_GENERAL, HT_MOVIES
hypertopic_assignment_matrix.loc[6, 0:5] = HT_SPORTS, HT_GENERAL, HT_MOVIES, HT_MUSIC, HT_MOVIES, HT_SPORTS
hypertopic_assignment_matrix.loc[7, 0:6] = HT_SPORTS, HT_GENERAL, HT_MOVIES, HT_MUSIC, HT_GENERAL, HT_MOVIES, HT_SPORTS
hypertopic_assignment_matrix.loc[8, 0:7] = HT_MOVIES, HT_GENERAL, HT_SPORTS, HT_MUSIC, HT_MOVIES, HT_GENERAL, HT_MOVIES, HT_GENERAL
hypertopic_assignment_matrix.loc[9, 0:8] = HT_GENERAL, HT_MOVIES, HT_MUSIC, HT_SPORTS, HT_GENERAL, HT_MOVIES, HT_GENERAL, HT_GENERAL, HT_MOVIES
hypertopic_assignment_matrix.loc[10, 0:9] = HT_GENERAL, HT_MOVIES, HT_SPORTS, HT_MUSIC, HT_GENERAL, HT_MOVIES, HT_MOVIES, HT_MOVIES, HT_GENERAL, HT_MUSIC

def find_hypertopic_frequencies(n_topics: int) -> pd.Series:
    model, topics = train_lda_model(n_topics)
    articles_en[['_topic', '_topic_entropy']] = articles_tokenized.parallel_apply(lambda doc: pd.Series(find_topic_and_entropy(model, doc)))
    topics_distribution = articles_en[['_topic', 'prevalent_ses', 'content']].groupby(['_topic', 'prevalent_ses']).count()
    hypertopic_frequencies = pd.Series(index=hypertopics, data=[[0,0],]*len(hypertopics))
    for topic in topics_distribution.index.get_level_values(0).unique():
        hypertopic = hypertopic_assignment_matrix.loc[n_topics, topic]
        low, high = hypertopic_frequencies[hypertopic]
        hypertopic_frequencies[hypertopic] = [low+topics_distribution.loc[topic, -1.0]['content'], high+topics_distribution.loc[topic, +1.0]['content']]
    return hypertopic_frequencies

hypertopic_frequency_matrix = pd.DataFrame(index=list(topic_names_en.keys()), columns=hypertopics, data=None)
with Pool(len(topic_names_en)) as pool:
    # results = pool.map(find_hypertopic_frequencies, topic_names_en)
    results = [find_hypertopic_frequencies(n_topics) for n_topics in topic_names_en]
    for n_topics, frequencies in zip(topic_names_en, results):
        hypertopic_frequency_matrix.loc[n_topics] = frequencies
articles_en = articles_en.drop(['_topic', '_topic_entropy'], axis=1)


# %%
plt.title('SES distribution by hypertopic')
plt.xlabel('number of topics')
plt.ylabel('relative amount of high-SES articles')
for hypertopic in hypertopics:
    plt.plot(hypertopic_frequency_matrix.index, hypertopic_frequency_matrix[hypertopic].apply(lambda x: x[1]/(x[0]+x[1]) if x[0]+x[1] != 0 else np.nan), label=hypertopic)
plt.legend()
plt.show()
# %%
