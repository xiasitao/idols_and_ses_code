# %%
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy.stats import chisquare
import re
from scipy.stats import chisquare, chi2_contingency

from pathlib import Path
SOURCE_PATH = Path(__file__).parent.resolve()
ASSET_PATH = SOURCE_PATH.joinpath('..', '..', 'assets').resolve()
BUILD_PATH = SOURCE_PATH.joinpath("..", "..", "build").resolve()


# %%
articles_raw = pd.read_pickle(BUILD_PATH / 'articles/articles_balanced_50.pkl')
articles_raw = articles_raw[articles_raw['language_ml']=='en']
ses = pd.read_pickle(BUILD_PATH / 'role_models/ses_scores.pkl')
articles_ses = articles_raw.join(ses, on='role_model', how='inner')
ses_distinct = pd.read_pickle(BUILD_PATH / 'role_models/ses_scores_distinct.pkl')
articles_ses_distinct = articles_raw.join(ses_distinct, on='role_model', how='inner')
human_annotated = pd.read_pickle(BUILD_PATH / 'articles/articles_human_annotated.pkl')
human_annotated = pd.concat([human_annotated, pd.read_pickle(BUILD_PATH / 'articles/articles_human_annotated_distinct.pkl')])

with open(BUILD_PATH / 'topic_modelling/topic_modelling_low_ses_distinct.pkl', 'rb') as file:
    topic_words_low_ses_distinct, article_topics_low_ses_distinct = pickle.load(file)
    articles_low_distinct = articles_ses_distinct.join(article_topics_low_ses_distinct, how='inner', on='article_id')
with open(BUILD_PATH / 'topic_modelling/topic_modelling_high_ses_distinct.pkl', 'rb') as file:
    topic_words_high_ses_distinct, article_topics_high_ses_distinct = pickle.load(file)
    articles_high_distinct = articles_ses_distinct.join(article_topics_high_ses_distinct, how='inner', on='article_id')
articles_per_ses_distinct = len(articles_low_distinct), len(articles_high_distinct)
topic_columns = [column for column in article_topics_low_ses_distinct.columns if not column.endswith('_entropy')]

# %%
def find_topic_distributions(articles_low: pd.DataFrame, articles_high: pd.DataFrame, columns: list) -> dict:   
    """Find the distribution of topics for low and high SES for number of topics available.

    Args:
        articles (pd.DataFrame): Article data.
        category_columns (list): List of column in the data corresponding to topics.

    Returns:
        dict: dict of category distribution data frames for each number of topics.
    """
    topic_distributions = {}
    for n_topics_column in columns:
        topic_distribution = pd.DataFrame(data=None, index=articles_low[n_topics_column].unique(), columns=['low', 'high'])
        topic_distribution['low'] = articles_low.groupby(n_topics_column).count()['content']
        topic_distribution['high'] = articles_high.groupby(n_topics_column).count()['content']
        topic_distributions[n_topics_column] = topic_distribution
    return topic_distributions


def print_topic_words(topic_words: dict, n_topics: int):
    """Print topic words more readably.

    Args:
        topic_words (dict): Topic word lists per n_topics.
        n_topics (int): Display word lists for n_topics topics.
    """    
    topics = topic_words[n_topics]
    print(f'{n_topics} topics:')
    for i, topic in enumerate(topics):
        print(f'\t{i}\t{" ".join(topic)}')


def plot_topic_distribution(topic_distribution: pd.DataFrame, relative=True, additional_title_text: str=None):
    """Plot the distribution of articles over the topics for low and high SES.

    Args:
        topic_distribution (pd.DataFrame): Distribution matrix with categories (index) and SES (columns)
        category_name (str): Name of the category
        relative (bool, optional): Whether to normalize frequencies for each SES level. Defaults to True.
    """    
    topic_distribution = topic_distribution.copy().sort_index()

    fig, ax = plt.gcf(), plt.gca()
    ax.set_xlabel('topic')
    ax.set_ylabel('topic article count')
    if additional_title_text is not None:
        ax.set_title(additional_title_text)
    if relative:
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
        ax.set_ylabel('topic article percentage')
        topic_distribution = topic_distribution.apply(lambda col: col/col.sum())
    if topic_distribution.index.dtype == float:
        topic_distribution.index = topic_distribution.index.astype(int)
    topic_distribution.plot(kind='bar', ax=ax)
    fig.show()


def find_hypertopics(articles: pd.DataFrame, hypertopic_table: dict, columns: list) -> pd.DataFrame:
    articles = articles[['article_id', 'average_ses', 'low_ses', 'high_ses', 'content'] + columns].drop_duplicates().set_index('article_id', drop=True)
    hypertopics = pd.DataFrame(data=None, columns=['average_ses', 'low_ses', 'high_ses', 'content']+columns, index=articles.index)
    hypertopics[['average_ses', 'low_ses', 'high_ses', 'content']] = articles[['average_ses', 'low_ses', 'high_ses', 'content']]
    for column in columns:
        n_topics = int(re.match(r'topic_(\d+)', column).groups()[0])
        hypertopics[f'topic_{n_topics}'] = articles[f'topic_{n_topics}'].apply(lambda topic: hypertopic_table[n_topics][int(topic)])
    return hypertopics


def plot_hypertopic_distribution_by_n(hypertopic_distributions: dict, hypertopics: list, articles_per_SES: tuple=None):
    ns = [re.match(r'topic_(\d+)', column).groups()[0] for column in hypertopic_distributions]
    low_ses_hypertopic_frequencies = pd.DataFrame(data=0, index=hypertopics, columns=ns)
    high_ses_hypertopic_frequencies = pd.DataFrame(data=0, index=hypertopics, columns=ns)
    for n in ns:
        low_ses_hypertopic_frequencies[n] = hypertopic_distributions[f'topic_{n}']['low']
        high_ses_hypertopic_frequencies[n] = hypertopic_distributions[f'topic_{n}']['high']
    
    fig, ax = plt.gcf(), plt.gca()
    ax.set_title('Hypertopic distributions')
    ax.set_ylabel('percentage of low SES articles')
    ax.set_xlabel('number of topics')
    for hypertopic in hypertopics:
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
        ax.plot([int(n) for n in ns], low_ses_hypertopic_frequencies.loc[hypertopic]/(low_ses_hypertopic_frequencies.loc[hypertopic]+high_ses_hypertopic_frequencies.loc[hypertopic]), label=hypertopic)
    if articles_per_SES is not None:
        overall_ratio = articles_per_SES[0]/(articles_per_SES[0]+articles_per_SES[1])
        ax.plot(ns, len(ns)*[overall_ratio], '--', label='all articles', color='grey', )
    ax.legend()
    ax.grid()
    fig.show()


def chi2_per_label_test(distribution: pd.DataFrame, articles_per_SES: tuple) -> pd.DataFrame:
    """Perform a chi2 test on the absolute frequencies articles in each category.

    Args:
        category_distribution (pd.DataFrame): Distributions of SES (columns) in the cateogories (index)
        articles_per_SES (tuple): Number of overall articles per SES (low, high)

    Raises:
        ValueError: If relative frequencies are supplied.

    Returns:
        pd.DataFrame: chi2 and p per category
    """    
    if not (distribution == distribution.astype(int)).all().all():
        raise ValueError('Cannot accept relative frequencies.')

    results = pd.DataFrame(None, columns=['chi2', 'p'], index=distribution.index)
    for category in distribution.index:
        frequencies = distribution.loc[category]
        expected_frequencies = np.array(articles_per_SES)/np.sum(np.array(articles_per_SES)) * np.sum(frequencies)
        result = chisquare(distribution.loc[category], expected_frequencies)
        results.loc[category] = [result.statistic, result.pvalue]
    return results.sort_index()


def chi2_contingency_test(distribution: pd.DataFrame) -> tuple:
    """Perform a chi2 test checking whether the labels of a category are differently distributed for low and the high SES.

    Args:
        distribution (pd.DataFrame): Low and high SES distribution of labels in a category.

    Returns:
        tuple: chi2, p values
    """    
    result = chi2_contingency(np.array(distribution.T))
    return result.statistic, result.pvalue


# %%
print_topic_words(topic_words_low_ses_distinct, 10)
print_topic_words(topic_words_high_ses_distinct, 10)


# %%
topic_distribution_distinct = find_topic_distributions(articles_low_distinct, articles_high_distinct, topic_columns)
plot_topic_distribution(topic_distribution_distinct['topic_5'])


# %%
HT_MOVIE, HT_SPORT, HT_MUSIC, HT_LIFE = 'movie', 'sport', 'music', 'life'
hypertopics = [HT_MOVIE, HT_SPORT, HT_MUSIC, HT_LIFE]
print_topic_words(topic_words_high_ses_distinct, 15)
hypertopic_table_low = {
    5: [HT_SPORT, HT_MOVIE, HT_LIFE, HT_LIFE, HT_MUSIC],
    10: [HT_LIFE, HT_LIFE, HT_LIFE, HT_LIFE, HT_MOVIE, HT_MUSIC, HT_SPORT, HT_SPORT, HT_SPORT, HT_MUSIC],
    15: [
        HT_SPORT, HT_LIFE, HT_LIFE, HT_MUSIC, HT_LIFE,  # 0
        HT_MUSIC, HT_LIFE, HT_MOVIE, HT_LIFE, HT_LIFE,  # 5
        HT_SPORT, HT_SPORT, HT_LIFE, HT_LIFE, HT_SPORT, # 10
    ],
}
hypertopic_table_high = {
    5: [HT_MOVIE, HT_LIFE, HT_SPORT, HT_MOVIE, HT_LIFE],
    10: [HT_MOVIE, HT_LIFE, HT_LIFE, HT_SPORT, HT_MUSIC, HT_SPORT, HT_MUSIC, HT_LIFE, HT_LIFE, HT_MUSIC],
    15: [
        HT_MOVIE, HT_MUSIC, HT_SPORT, HT_LIFE, HT_MOVIE,  # 0
        HT_SPORT, HT_MOVIE, HT_LIFE, HT_SPORT, HT_MUSIC,  # 5
        HT_MOVIE, HT_MUSIC, HT_LIFE, HT_LIFE, HT_MUSIC,  # 10
    ]
}
assert(all([len(hypertopic_table_low[n]) == n for n in hypertopic_table_low]))
assert(all([len(hypertopic_table_high[n]) == n for n in hypertopic_table_high]))


# %%
articles_low_distinct_hypertopics = find_hypertopics(articles_low_distinct, hypertopic_table_low, ['topic_10', 'topic_15'])
articles_high_distinct_hypertopics = find_hypertopics(articles_high_distinct, hypertopic_table_high, ['topic_10', 'topic_15'])
hypertopics_distributions_distinct = find_topic_distributions(articles_low_distinct_hypertopics, articles_high_distinct_hypertopics, ['topic_10', 'topic_15'])


# %%
to_evaluate = 'topic_15'
plot_topic_distribution(hypertopics_distributions_distinct[to_evaluate])
print(chi2_contingency_test(hypertopics_distributions_distinct[to_evaluate]))
print(chi2_per_label_test(hypertopics_distributions_distinct[to_evaluate], articles_per_ses_distinct))

# %%
