import sys
sys.dont_write_bytecode = True

import pandas as pd
import pickle
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import re

from scipy.stats import chisquare, chi2_contingency
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

from pathlib import Path
SOURCE_PATH = Path(__file__).parent.resolve()
ASSET_PATH = SOURCE_PATH.joinpath('..', '..', 'assets').resolve()
BUILD_PATH = SOURCE_PATH.joinpath("..", "..", "build").resolve()


def retrieve_data():
    """Retrieve all necessary data

    Returns:
        tuple: articles, human_annotated, articles_distinct, human_annotated_distinct, topic_columns, topic_words
    """    
    articles_raw = pd.read_pickle(BUILD_PATH / 'articles/articles_balanced_50.pkl')
    articles_raw = articles_raw[articles_raw['language_ml']=='en']
    ses = pd.read_pickle(BUILD_PATH / 'role_models/ses_scores.pkl')
    ses_distinct = pd.read_pickle(BUILD_PATH / 'role_models/ses_scores_distinct.pkl')
    human_annotated = pd.read_pickle(BUILD_PATH / 'articles/articles_human_annotated.pkl')
    human_annotated_distinct = pd.read_pickle(BUILD_PATH / 'articles/articles_human_annotated_distinct.pkl')
    with open(BUILD_PATH / 'topic_modelling/topic_modelling.pkl', 'rb') as file:
        topic_words, article_topics = pickle.load(file)
    topic_columns = [column for column in article_topics.columns if not column.endswith('_entropy') and not column.endswith('_p')]

    def load_prepare_articles(articles: pd.DataFrame, ses: pd.DataFrame, article_topics: pd.DataFrame):
        """Combine article data, ses, and topic data.

        Args:
            articles (pd.DataFrame): Articles
            ses (pd.DataFrame): SES scores
            article_topics (pd.DataFrame): Topics associations of the articles.

        Returns:
            pd.DataFrame: articles
        """
        articles = articles.join(ses, how='inner', on='role_model')
        articles = articles.join(article_topics, how='inner', on='article_id')
        return articles
    articles = load_prepare_articles(articles_raw, ses, article_topics)
    articles_distinct = load_prepare_articles(articles_raw, ses_distinct, article_topics)

    return articles, human_annotated, articles_distinct, human_annotated_distinct, topic_columns, topic_words


def find_articles_per_SES(articles: pd.DataFrame, column='content') -> tuple:
    """Count the number of articles with low and high SES having a valid entry in a certain column.

    Args:
        articles (pd.DataFrame): article data
        column (str, optional): Reference column to determine if article is to be considered in the counting. If nan/None, then do not count. Defaults to 'content'.

    Returns:
        tuple: _description_
    """    
    low, high = articles[articles['low_ses']].count()[column], articles[articles['high_ses']].count()[column]
    return low, high


def filter_out_low_entropy_labels(articles: pd.DataFrame, percentile: float, topic_columns: list) -> pd.DataFrame:
    """Set the label of all articles with entropy above a certain percentil to nan.

    Args:
        articles (pd.DataFrame): articles
        topic_columns (list): list of all topic columns to filter for
        percentile (float): Entropy percentile above which everying is to be filtered out. Between 0.0 and 1.0.

    Returns:
        list: Article ids, their category values and their entropies that have entropy lower than the percentile.
    """
    articles = articles.copy()
    for column in topic_columns:
        percentile_boundary = np.percentile(articles[f'{column}_entropy'], 100*percentile)
        articles[column] = articles[column].mask(articles[f'{column}_entropy'] > percentile_boundary)
    return articles


def find_topic_distributions(articles: pd.DataFrame, columns: list) -> dict:   
    """Find the distribution of topics for low and high SES for number of topics available.

    Args:
        articles (pd.DataFrame): Article data.
        category_columns (list): List of column in the data corresponding to topics.

    Returns:
        dict: dict of category distribution data frames for each number of topics.
    """
    topic_distributions = {}
    for n_topics_column in columns:
        n_topic_articles = articles[~articles[n_topics_column].isna()]
        topic_distribution = pd.DataFrame(data=None, index=n_topic_articles[n_topics_column].unique(), columns=['low', 'high'])
        topic_distribution['low'] = n_topic_articles[n_topic_articles['low_ses']].groupby(n_topics_column).count()['content']
        topic_distribution['high'] = n_topic_articles[n_topic_articles['high_ses']].groupby(n_topics_column).count()['content']
        topic_distributions[n_topics_column] = topic_distribution
    return topic_distributions


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


def plot_topic_distribution(topic_distribution: pd.DataFrame, relative=True, additional_title_text: str=None, ax: plt.axis=None):
    """Plot the distribution of articles over the topics for low and high SES.

    Args:
        topic_distribution (pd.DataFrame): Distribution matrix with categories (index) and SES (columns)
        category_name (str): Name of the category
        relative (bool, optional): Whether to normalize frequencies for each SES level. Defaults to True.
        ax (plt.axis, optional): axis to plot on
    """    
    topic_distribution = topic_distribution.copy().sort_index()

    fig = None
    if ax is None:
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
    if fig is not None:
        fig.show()


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


def find_hypertopics(articles: pd.DataFrame, hypertopic_table: dict, columns: list) -> pd.DataFrame:
    articles = articles[['article_id', 'average_ses', 'low_ses', 'high_ses', 'content'] + columns].drop_duplicates().set_index('article_id', drop=True)
    hypertopics = pd.DataFrame(data=None, columns=['average_ses', 'low_ses', 'high_ses', 'content']+columns, index=articles.index)
    hypertopics[['average_ses', 'low_ses', 'high_ses', 'content']] = articles[['average_ses', 'low_ses', 'high_ses', 'content']]
    for column in columns:
        n_topics = int(re.match(r'topic_(\d+)', column).groups()[0])
        hypertopics[f'topic_{n_topics}'] = articles[f'topic_{n_topics}'].apply(lambda topic: hypertopic_table[n_topics][int(topic)] if not np.isnan(topic) else np.nan)
    return hypertopics


def plot_hypertopic_distribution_by_n(hypertopic_distributions: dict, hypertopics: list, articles_per_SES: tuple=None, ax: plt.axis=None):
    """Plot the low-SES article portion of each hypertopic over varying n_topics. 
    This is for consistency checking: If the lines converge, one can assume that varying n_topics doesn't change the prediction of the topic distribution much.

    Args:
        hypertopic_distributions (dict): Dict with the hypertopic distributions for all relevant n_topics. Output of find_topics_distributions.
        hypertopics (list): Hypertopics to draw graphs for.
        articles_per_SES (tuple, optional): Reference distribution of articles to compare to. Defaults to None.
        ax (plt.axis, optional): axis to plot on
    """    
    ns = [re.match(r'topic_(\d+)', column).groups()[0] for column in hypertopic_distributions]
    low_ses_hypertopic_frequencies = pd.DataFrame(data=0, index=hypertopics, columns=ns)
    high_ses_hypertopic_frequencies = pd.DataFrame(data=0, index=hypertopics, columns=ns)
    for n in ns:
        low_ses_hypertopic_frequencies[n] = hypertopic_distributions[f'topic_{n}']['low']
        high_ses_hypertopic_frequencies[n] = hypertopic_distributions[f'topic_{n}']['high']
    
    fig = None
    if ax is None:
        fig, ax = plt.gcf(), plt.gca()
    ax.set_ylabel('percentage of low-SES articles')
    ax.set_xlabel('number of topics')
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    for hypertopic in hypertopics:
        ax.plot([int(n) for n in ns], low_ses_hypertopic_frequencies.loc[hypertopic]/(low_ses_hypertopic_frequencies.loc[hypertopic]+high_ses_hypertopic_frequencies.loc[hypertopic]), label=hypertopic)
    if articles_per_SES is not None:
        overall_ratio = articles_per_SES[0]/(articles_per_SES[0]+articles_per_SES[1])
        ax.plot([int(n) for n in ns], len(ns)*[overall_ratio], '--', label='all articles', color='grey', )
    ax.legend()
    ax.grid()
    if fig is not None:
        fig.show()


def plot_human_annotation_confusion_matrix(article_hypertopic_data: pd.DataFrame, human_annotated: pd.DataFrame, n_topics: int, ax: plt.axis=None, cmap=None) -> ConfusionMatrixDisplay:
    """Plot the confusion matrix for hypertopics.

    Args:
        article_hypertopic_data (pd.DataFrame): Dataframe with the hypertopic for each article.
        human_annotated (pd.DataFrame): Human annotated articles.
        n_topics (int): n_topics to plot confusion matrix for.
        ax (plt.axis, optional): axis to plot on
    """    
    annotation_column = 'topic'
    if annotation_column not in human_annotated.columns:
        raise Exception('No topics in human annotations.')
    hypertopic_column = f'topic_{n_topics}'
    if hypertopic_column not in article_hypertopic_data.columns:
        raise Exception(f'No {hypertopic_column} in articles with hypertopics.')
    
    human_annotated_topic = human_annotated[~human_annotated[annotation_column].isna()]
    if len(human_annotated_topic) == 0:
        return
    articles_with_annotation = article_hypertopic_data.join(human_annotated_topic[[annotation_column]], on='article_id', how='inner')[['content', hypertopic_column, annotation_column]]
    topic_labels = np.unique(articles_with_annotation[[hypertopic_column, annotation_column]].values.ravel())
    topic_confusion_matrix = confusion_matrix(y_true=articles_with_annotation[annotation_column], y_pred=articles_with_annotation[hypertopic_column], labels=topic_labels)
    
    fig = None
    if ax is None:
        fig, ax = plt.gcf(), plt.gca()
    ax.set_title(f'{n_topics} topics')
    display = ConfusionMatrixDisplay(topic_confusion_matrix, display_labels=topic_labels)
    display.plot(ax=ax, colorbar=False, cmap=cmap)
    if fig is not None:
        fig.show()
    return display


def plot_accuracy_by_n(article_hypertopic_data: pd.DataFrame,  human_annotated: pd.DataFrame, ax: plt.axis=None):
    """Plot the accuracy over varying n_topics.

    Args:
        article_hypertopic_data (pd.DataFrame): Dataframe indicating the hypertopic for each article.
        human_annotated (pd.DataFrame): Human annotation data for a subset of the articles.
        ax (plt.axis, optional): axis to plot on
    """    
    annotation_column = 'topic'
    ns = [int(re.match(r'topic_(\d+)', column).groups()[0]) for column in article_hypertopic_data.columns if re.match(r'topic_(\d+)', column) is not None]

    human_annotated_topic = human_annotated[~human_annotated[annotation_column].isna()]
    if len(human_annotated_topic) == 0:
        return
    articles_with_annotation = article_hypertopic_data.join(human_annotated_topic[[annotation_column]], on='article_id', how='inner')

    accuracies = []
    for n in ns:
        articles_with_annotation_for_n = articles_with_annotation[~articles_with_annotation[f'topic_{n}'].isna()]
        true_hypertopics = articles_with_annotation_for_n[annotation_column]
        predicted_hypertopics = articles_with_annotation_for_n[f'topic_{n}']
        accuracy = accuracy_score(true_hypertopics, predicted_hypertopics)
        accuracies.append(accuracy)
    
    fig = None
    if ax is None:
        fig, ax = plt.gcf(), plt.gca()
    ax.set_xlabel('number of topics')
    ax.set_ylabel('accuracy')
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    ax.plot(ns, accuracies)
    ax.grid()
    if fig is not None:
        fig.show()


def evaluate_topics_for_n(
        articles: pd.DataFrame,
        topic_columns: list,
        n_topics: int,
        articles_per_SES: tuple,
        relative_dist_plot: bool=True,
        is_distinct: bool=None,
        ax: plt.axis=None
    ):
    """Plot the topic distributions, calculate the chi2 scores.

    Args:
        articles (pd.DataFrame): Articles with topics.
        topic_columns (list): List of topic columns
        n_topics (int): n_topics to evaluate for.
        articles_per_SES (tuple): Reference article count per SES.
        relative_dist_plot (bool, optional): Whether to plot relative frequencies in the distribution plot. Defaults to True.
        is_distinct (bool, optional): Is the distinct-SES dataset used? Used for annotation the plot. Defaults to None.
        ax (plt.axis, optional): axis to plot on
    """    
    column_name = f'topic_{n_topics}'
    topic_distributions = find_topic_distributions(articles, topic_columns)
    
    distinct_text = None
    if is_distinct is not None:
        distinct_text = 'distinct-SES' if is_distinct else 'mixed-SES'
    plot_topic_distribution(topic_distributions[column_name], relative=relative_dist_plot, additional_title_text=distinct_text, ax=ax)

    contingency_chi2, contingency_p = chi2_contingency_test(topic_distributions[column_name])
    print(f'Distribution chi2 test:\nchi2={contingency_chi2:.1f}, p={contingency_p:.3e}\n')

    print('Per-label chi2 test:')
    print(chi2_per_label_test(topic_distributions[column_name], articles_per_SES))


HT_MOVIE, HT_SPORT, HT_MUSIC, HT_LIFE = 'movie', 'sport', 'music', 'life'
hypertopics = [HT_MOVIE, HT_SPORT, HT_MUSIC, HT_LIFE]
hypertopic_table = {
    2: [HT_LIFE, HT_MOVIE],
    3: [HT_SPORT, HT_MUSIC, HT_MOVIE],
    4: [HT_SPORT, HT_LIFE, HT_MOVIE, HT_LIFE],
    5: [HT_LIFE, HT_SPORT, HT_MOVIE, HT_LIFE, HT_MUSIC],
    6: [HT_MUSIC, HT_SPORT, HT_LIFE, HT_LIFE, HT_MOVIE, HT_LIFE],
    7: [HT_MUSIC, HT_SPORT, HT_MOVIE, HT_LIFE, HT_LIFE, HT_LIFE, HT_LIFE],
    8: [HT_SPORT, HT_LIFE, HT_MOVIE, HT_LIFE, HT_LIFE, HT_LIFE, HT_LIFE, HT_MUSIC],
    9: [HT_SPORT, HT_MOVIE, HT_LIFE, HT_LIFE, HT_LIFE, HT_LIFE, HT_LIFE, HT_MUSIC, HT_SPORT],
    10: [HT_SPORT, HT_MOVIE, HT_LIFE, HT_LIFE, HT_LIFE, HT_SPORT, HT_LIFE, HT_LIFE, HT_MUSIC, HT_LIFE],
    11: [HT_SPORT, HT_MOVIE, HT_LIFE, HT_LIFE, HT_MUSIC, HT_LIFE, HT_LIFE, HT_SPORT, HT_LIFE, HT_SPORT, HT_LIFE],
    12: [HT_SPORT, HT_LIFE, HT_MOVIE, HT_LIFE, HT_MUSIC, HT_LIFE, HT_LIFE, HT_LIFE, HT_LIFE, HT_LIFE, HT_SPORT, HT_SPORT],
    13: [HT_SPORT, HT_LIFE, HT_MOVIE, HT_LIFE, HT_MUSIC, HT_LIFE, HT_LIFE, HT_SPORT, HT_SPORT, HT_LIFE, HT_LIFE, HT_LIFE, HT_LIFE],
    14: [HT_SPORT, HT_LIFE, HT_MOVIE, HT_LIFE, HT_MUSIC, HT_LIFE, HT_LIFE, HT_SPORT, HT_SPORT, HT_LIFE, HT_LIFE, HT_LIFE, HT_LIFE, HT_LIFE],
    15: [HT_LIFE, HT_SPORT, HT_MOVIE, HT_LIFE, HT_MUSIC, HT_LIFE, HT_LIFE, HT_SPORT, HT_LIFE, HT_LIFE, HT_SPORT, HT_LIFE, HT_LIFE, HT_LIFE, HT_LIFE],
    20: [
        HT_LIFE, HT_SPORT, HT_SPORT, HT_MOVIE, HT_LIFE,  #0
        HT_MUSIC, HT_LIFE, HT_LIFE, HT_LIFE, HT_LIFE,  # 5
        HT_LIFE, HT_LIFE, HT_SPORT, HT_LIFE, HT_LIFE,  # 10
        HT_LIFE, HT_LIFE, HT_LIFE, HT_LIFE, HT_SPORT,  # 15
    ],
    25: [
        HT_LIFE, HT_SPORT, HT_SPORT, HT_MOVIE, HT_LIFE,  # 0
        HT_MUSIC, HT_LIFE, HT_LIFE, HT_SPORT, HT_LIFE,  # 5
        HT_LIFE, HT_LIFE, HT_SPORT, HT_LIFE, HT_LIFE,  # 10
        HT_SPORT, HT_LIFE, HT_LIFE, HT_LIFE, HT_LIFE, # 15
        HT_SPORT, HT_LIFE, HT_LIFE, HT_MUSIC, HT_LIFE # 20
    ],
    30: [
        HT_LIFE, HT_SPORT, HT_MUSIC, HT_SPORT, HT_MUSIC,  # 0
        HT_SPORT, HT_LIFE, HT_MOVIE, HT_LIFE, HT_LIFE,  # 5
        HT_SPORT, HT_LIFE, HT_LIFE, HT_LIFE, HT_LIFE,  # 10
        HT_LIFE, HT_LIFE, HT_LIFE, HT_LIFE, HT_SPORT,  # 15
        HT_MOVIE, HT_LIFE, HT_LIFE, HT_LIFE, HT_LIFE,  # 20
        HT_LIFE, HT_LIFE, HT_SPORT, HT_LIFE, HT_LIFE,  # 25
    ],
    35: [
        HT_LIFE, HT_SPORT, HT_SPORT, HT_SPORT, HT_MUSIC,  # 0
        HT_LIFE, HT_SPORT, HT_LIFE, HT_MOVIE, HT_LIFE,  # 5 
        HT_LIFE, HT_LIFE, HT_LIFE, HT_LIFE, HT_MOVIE, # 10
        HT_MUSIC, HT_LIFE, HT_LIFE, HT_LIFE, HT_LIFE,  # 15
        HT_LIFE, HT_SPORT, HT_LIFE, HT_MOVIE, HT_LIFE,  # 20
        HT_SPORT, HT_LIFE, HT_LIFE, HT_SPORT, HT_LIFE,  # 25
        HT_LIFE, HT_MUSIC, HT_LIFE, HT_LIFE, HT_LIFE,  # 30
    ],
    40: [
        HT_LIFE, HT_SPORT, HT_MUSIC, HT_MUSIC, HT_LIFE,  # 0
        HT_LIFE, HT_LIFE, HT_LIFE, HT_SPORT, HT_LIFE,  # 5
        HT_MOVIE, HT_LIFE, HT_MOVIE, HT_SPORT, HT_LIFE,  # 10
        HT_LIFE, HT_LIFE, HT_LIFE, HT_SPORT, HT_LIFE,  # 15
        HT_LIFE, HT_SPORT, HT_MUSIC, HT_MUSIC, HT_LIFE,  # 20
        HT_LIFE, HT_MOVIE, HT_LIFE, HT_LIFE, HT_LIFE,  # 25
        HT_SPORT, HT_LIFE, HT_LIFE, HT_LIFE, HT_SPORT,  # 30
        HT_LIFE, HT_MUSIC, HT_LIFE, HT_SPORT, HT_SPORT,  # 35
    ],
    45: [
        HT_LIFE, HT_SPORT, HT_MUSIC, HT_MOVIE, HT_LIFE,  # 0
        HT_LIFE, HT_SPORT, HT_LIFE, HT_SPORT, HT_LIFE,  # 5
        HT_MOVIE, HT_LIFE, HT_LIFE, HT_LIFE,HT_LIFE,  # 10
        HT_LIFE, HT_LIFE, HT_LIFE, HT_LIFE, HT_LIFE,  # 15
        HT_SPORT, HT_LIFE, HT_MUSIC, HT_LIFE, HT_SPORT,  # 20
        HT_LIFE, HT_LIFE, HT_LIFE, HT_MUSIC, HT_MUSIC,  # 25
        HT_LIFE, HT_LIFE, HT_LIFE, HT_LIFE, HT_LIFE,  # 30
        HT_LIFE, HT_LIFE, HT_LIFE, HT_LIFE, HT_SPORT,  # 35
        HT_LIFE, HT_LIFE, HT_LIFE, HT_LIFE, HT_LIFE,  # 40
    ],
    50: [
        HT_LIFE, HT_MOVIE, HT_SPORT, HT_MUSIC, HT_SPORT,  # 0
        HT_LIFE, HT_LIFE, HT_MOVIE, HT_LIFE, HT_LIFE,  # 5
        HT_LIFE, HT_LIFE, HT_LIFE, HT_LIFE, HT_MUSIC,  # 10
        HT_LIFE, HT_LIFE, HT_MUSIC, HT_LIFE, HT_SPORT,  # 15
        HT_MUSIC, HT_LIFE, HT_LIFE, HT_LIFE, HT_LIFE,  # 20
        HT_SPORT, HT_LIFE, HT_LIFE, HT_LIFE, HT_LIFE,  # 25
        HT_LIFE, HT_LIFE, HT_MUSIC, HT_LIFE, HT_SPORT,  # 30
        HT_SPORT, HT_SPORT, HT_LIFE, HT_LIFE, HT_LIFE,  # 35
        HT_LIFE, HT_LIFE, HT_LIFE, HT_LIFE, HT_LIFE,  # 40
        HT_LIFE, HT_SPORT, HT_SPORT, HT_SPORT, HT_LIFE,  # 45
    ],
    55: [
        HT_LIFE, HT_SPORT, HT_LIFE, HT_MUSIC, HT_MOVIE,  # 0
        HT_SPORT, HT_LIFE, HT_LIFE, HT_LIFE, HT_LIFE,  # 5
        HT_LIFE, HT_MUSIC, HT_SPORT, HT_LIFE, HT_LIFE,  # 10
        HT_LIFE, HT_LIFE, HT_LIFE, HT_LIFE, HT_SPORT,  # 15
        HT_LIFE, HT_MOVIE, HT_MUSIC, HT_LIFE, HT_LIFE,  # 20
        HT_LIFE, HT_LIFE, HT_LIFE, HT_MOVIE, HT_LIFE,  # 25
        HT_LIFE, HT_MOVIE, HT_LIFE, HT_LIFE, HT_MUSIC,  # 30
        HT_SPORT, HT_LIFE, HT_LIFE, HT_LIFE, HT_LIFE,  # 35
        HT_LIFE, HT_LIFE, HT_LIFE, HT_LIFE, HT_SPORT,  # 40
        HT_LIFE, HT_LIFE, HT_LIFE, HT_SPORT, HT_SPORT,  # 45
        HT_LIFE, HT_LIFE, HT_LIFE, HT_LIFE, HT_SPORT,  # 50
    ],
    60: [
        HT_SPORT, HT_LIFE, HT_MUSIC, HT_MOVIE, HT_LIFE,  # 0
        HT_SPORT, HT_SPORT, HT_LIFE, HT_LIFE, HT_MUSIC,  # 5
        HT_MUSIC, HT_LIFE, HT_LIFE, HT_LIFE, HT_LIFE,  # 10
        HT_LIFE, HT_LIFE, HT_MUSIC, HT_MUSIC, HT_LIFE,  # 15
        HT_LIFE, HT_LIFE, HT_LIFE, HT_MOVIE, HT_LIFE,  # 20
        HT_LIFE, HT_LIFE, HT_LIFE, HT_LIFE, HT_LIFE,  # 25
        HT_MOVIE, HT_LIFE, HT_LIFE, HT_LIFE, HT_SPORT,  # 30
        HT_SPORT, HT_SPORT, HT_LIFE, HT_LIFE, HT_LIFE,  # 35
        HT_SPORT, HT_LIFE, HT_SPORT, HT_LIFE, HT_LIFE,  # 40
        HT_SPORT, HT_SPORT, HT_LIFE, HT_LIFE, HT_LIFE,  # 45
        HT_LIFE, HT_MOVIE, HT_LIFE, HT_LIFE, HT_LIFE,  # 50
        HT_LIFE, HT_LIFE, HT_SPORT, HT_LIFE, HT_LIFE,  # 55
    ]
}
# assert(all([len(hypertopic_table[n]) == n for n in hypertopic_table]))
hypertopics_columns = [f'topic_{n}' for n in hypertopic_table]