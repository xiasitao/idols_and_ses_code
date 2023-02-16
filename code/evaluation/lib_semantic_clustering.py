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
    """Retrieve the article data, ses data, cluster data, and combine it.

    Returns:
        tuple: articles, human_annotated, articles_distinct, human_annotated_distinct, cluster_columns, cluster_topics, cluster_adjectives
    """    
    articles_raw = pd.read_pickle(BUILD_PATH / 'articles/articles_balanced_50.pkl')
    articles_raw = articles_raw[articles_raw['language_ml']=='en']
    ses = pd.read_pickle(BUILD_PATH / 'role_models/ses_scores.pkl')
    ses_distinct = pd.read_pickle(BUILD_PATH / 'role_models/ses_scores_distinct.pkl')
    human_annotated = pd.read_pickle(BUILD_PATH / 'articles/articles_human_annotated.pkl')
    human_annotated_distinct = pd.read_pickle(BUILD_PATH / 'articles/articles_human_annotated_distinct.pkl')
    article_clusters = pd.read_pickle(BUILD_PATH / 'semantic_similarity/semantic_clusters.pkl')
    cluster_columns = [column for column in article_clusters.columns]
    with open(BUILD_PATH / 'semantic_similarity/semantic_topics.pkl', 'rb') as file:
        cluster_topics = pickle.load(file)
    with open(BUILD_PATH / 'semantic_similarity/semantic_topics_adjectives.pkl', 'rb') as file:
        cluster_adjectives = pickle.load(file)

    def load_prepare_articles(articles: pd.DataFrame, ses: pd.DataFrame, article_clusters: pd.DataFrame):
        """Combine article data, ses, and topic data.

        Args:
            articles (pd.DataFrame): Articles
            ses (pd.DataFrame): SES scores
            article_topics (pd.DataFrame): Topics associations of the articles.

        Returns:
            tuple: articles [pd.DataFrame], articles_per_SES [tuple]
        """
        articles = articles.join(ses, how='inner', on='role_model')
        articles = articles.join(article_clusters, how='inner', on='article_id')
        return articles
    articles = load_prepare_articles(articles_raw, ses, article_clusters)
    articles_distinct = load_prepare_articles(articles_raw, ses_distinct, article_clusters)

    return articles, human_annotated, articles_distinct, human_annotated_distinct, cluster_columns, cluster_topics, cluster_adjectives


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
        topic_distribution = pd.DataFrame(data=None, index=articles[n_topics_column].unique(), columns=['low', 'high', 'low_rel', 'high_rel'])
        topic_distribution['low'] = articles[articles['low_ses']].groupby(n_topics_column).count()['content']
        topic_distribution['low_rel'] = topic_distribution['low']/topic_distribution['low'].sum()
        topic_distribution['high'] = articles[articles['high_ses']].groupby(n_topics_column).count()['content']
        topic_distribution['high_rel'] = topic_distribution['high']/topic_distribution['high'].sum()
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
    distribution = distribution[['low', 'high']]
    if not (distribution == distribution.astype(int)).all().all():
        raise ValueError('Cannot accept relative frequencies.')

    results = pd.DataFrame(None, columns=['chi2', 'p'], index=distribution.index)
    for topic in distribution.index:
        frequencies = distribution.loc[topic]
        expected_frequencies = np.array(articles_per_SES)/np.sum(np.array(articles_per_SES)) * np.sum(frequencies)
        result = chisquare(distribution.loc[topic], expected_frequencies)
        results.loc[topic] = [result.statistic, result.pvalue]
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

    
def print_cluster_topics_words(cluster_topics: dict, n_clusters: int):
    """Print cluster topic words more readably.

    Args:
        cluster_topics (dict): Cluster topics word lists per n_clusters.
        n_clusters (int): Display word lists for n_clusters topics.
    """    
    topics = cluster_topics[f'cluster_{n_clusters}']
    print(f'{n_clusters} clusters topics:')
    for cluster in sorted(topics):
        topic = topics[cluster]
        print(f'\t{cluster}\t{" ".join(topic)}')


def plot_topic_distribution(topic_distribution: pd.DataFrame, relative=True, additional_title_text: str=None, ax: plt.axis=None):
    """Plot the distribution of articles over the topics for low and high SES.

    Args:
        topic_distribution (pd.DataFrame): Distribution matrix with categories (index) and SES (columns)
        category_name (str): Name of the category
        relative (bool, optional): Whether to normalize frequencies for each SES level. Defaults to True.
    """    
    topic_distribution = topic_distribution.copy().sort_index()[['low', 'high']]

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


def find_hypertopics(articles: pd.DataFrame, hypertopic_table: dict, columns: list) -> pd.DataFrame:
    """Find the hypertopic of each cluster from the cluster association and a hypertopic table.

    Args:
        articles (pd.DataFrame): articles with hypertopic information
        hypertopic_table (dict): cluster to hypertopic translation
        columns (list): Columns to consider in the article dataframe.

    Returns:
        pd.DataFrame: Dataframe with article id and hypertopics with the same column names
    """    
    articles = articles[['article_id', 'average_ses', 'low_ses', 'high_ses', 'content'] + columns].drop_duplicates().set_index('article_id', drop=True)
    hypertopics = pd.DataFrame(data=None, columns=['average_ses', 'low_ses', 'high_ses', 'content']+columns, index=articles.index)
    hypertopics[['average_ses', 'low_ses', 'high_ses', 'content']] = articles[['average_ses', 'low_ses', 'high_ses', 'content']]
    for column in columns:
        n_clusters = int(re.match(r'cluster_(\d+)', column).groups()[0])
        hypertopics[f'cluster_{n_clusters}'] = articles[f'cluster_{n_clusters}'].apply(lambda cluster: hypertopic_table[n_clusters][int(cluster)])
    return hypertopics


def plot_human_annotation_confusion_matrix(article_hypertopic_data: pd.DataFrame, human_annotated: pd.DataFrame, n_clusters: int, ax: plt.axis=None, cmap=None) -> ConfusionMatrixDisplay:
    """Plot the confusion matrix for hypertopics.

    Args:
        article_hypertopic_data (pd.DataFrame): Dataframe with the hypertopic for each article.
        human_annotated (pd.DataFrame): Human annotated articles.
        n_topics (int): n_topics to plot confusion matrix for.
        ax (plt.axis, optional): Axis to plot on if given
        cmap (optional): Colormap to use for plotting.

    Returns:
        Confusion matrix display
    """    
    annotation_column = 'topic'
    if annotation_column not in human_annotated.columns:
        raise Exception('No topics in human annotations.')
    hypertopic_column = f'cluster_{n_clusters}'
    if hypertopic_column not in article_hypertopic_data.columns:
        raise Exception(f'No {hypertopic_column} in articles with hypertopics.')
    
    human_annotated_topic = human_annotated[~human_annotated[annotation_column].isna()]
    if len(human_annotated_topic) == 0:
        return
    articles_with_annotation = article_hypertopic_data.join(human_annotated_topic[[annotation_column]], on='article_id', how='inner')[['content', hypertopic_column, annotation_column]]
    topic_labels = np.unique(articles_with_annotation[[hypertopic_column, annotation_column]].values.ravel())
    topic_confusion_matrix = confusion_matrix(y_true=articles_with_annotation[annotation_column], y_pred=articles_with_annotation[hypertopic_column], labels=topic_labels)
    accuracy = accuracy_score(articles_with_annotation[annotation_column], articles_with_annotation[hypertopic_column])
    print(f'Accuracy for {n_clusters} clusters: {accuracy*100:.2f}%')

    display = ConfusionMatrixDisplay(topic_confusion_matrix, display_labels=topic_labels)
    if ax is None:
        ax = plt.gca()
    display.plot(ax=ax, colorbar=False, cmap=cmap)
    return display


def plot_accuracy_by_n(article_hypertopic_data: pd.DataFrame,  human_annotated: pd.DataFrame, ax: plt.axis=None):
    """Plot the accuracy over varying n_clusters.

    Args:
        article_hypertopic_data (pd.DataFrame): Dataframe indicating the hypertopic for each article.
        human_annotated (pd.DataFrame): Human annotation data for a subset of the articles.
        ax (plt.axis, optional): Axis to plot on
    """    
    annotation_column = 'topic'
    ns = [int(re.match(r'cluster_(\d+)', column).groups()[0]) for column in article_hypertopic_data.columns if re.match(r'cluster_(\d+)', column) is not None]

    human_annotated_topic = human_annotated[~human_annotated[annotation_column].isna()]
    if len(human_annotated_topic) == 0:
        return
    articles_with_annotation = article_hypertopic_data.join(human_annotated_topic[[annotation_column]], on='article_id', how='inner')

    accuracies = []
    for n in ns:
        true_hypertopics = articles_with_annotation[annotation_column]
        predicted_hypertopics = articles_with_annotation[f'cluster_{n}']
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


def plot_hypertopic_distribution_by_n(hypertopic_distributions: dict, hypertopics: list, articles_per_SES: tuple=None, ax: plt.axis=None):
    """Plot the low-SES article portion of each hypertopic over varying n_topics. 
    This is for consistency checking: If the lines converge, one can assume that varying n_topics doesn't change the prediction of the topic distribution much.

    Args:
        hypertopic_distributions (dict): Dict with the hypertopic distributions for all relevant n_topics. Output of find_topics_distributions.
        hypertopics (list): Hypertopics to draw graphs for.
        articles_per_SES (tuple, optional): Reference distribution of articles to compare to. Defaults to None.
        ax (matplotlib axis, optional): Axis to plot on.
    """    
    ns = [re.match(r'cluster_(\d+)', column).groups()[0] for column in hypertopic_distributions]
    low_ses_hypertopic_frequencies = pd.DataFrame(data=0, index=hypertopics, columns=ns)
    high_ses_hypertopic_frequencies = pd.DataFrame(data=0, index=hypertopics, columns=ns)
    for n in ns:
        low_ses_hypertopic_frequencies[n] = hypertopic_distributions[f'cluster_{n}']['low']
        high_ses_hypertopic_frequencies[n] = hypertopic_distributions[f'cluster_{n}']['high']
    
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


def evaluate_hypertopics_for_n(
        hypertopic_distribution: pd.DataFrame,
        articles_per_SES: tuple,
        relative_dist_plot: bool=True,
        is_distinct: bool=None,
        ax: plt.axis=None,
    ):
    """Plot the hypertopic distributions, calculate the chi2 scores.

    Args:
        hypertopic_distribution (pd.DataFrame): Distribution of hypertopics.
        n_topics (int): n_topics to evaluate for.
        articles_per_SES (tuple): Reference article count per SES.
        relative_dist_plot (bool, optional): Whether to plot relative frequencies in the distribution plot. Defaults to True.
        is_distinct (bool, optional): Is the distinct-SES dataset used? Used for annotation the plot. Defaults to None.
        ax (plt.axis, optional): Axis to plot the distribution on
    """        
    distinct_text = None
    if is_distinct is not None:
        distinct_text = 'distinct-SES' if is_distinct else 'mixed-SES'
    plot_topic_distribution(hypertopic_distribution, relative=relative_dist_plot, additional_title_text=distinct_text, ax=ax)

    contingency_chi2, contingency_p = chi2_contingency_test(hypertopic_distribution)
    print(f'Distribution chi2 test:\nchi2={contingency_chi2:.1f}, p={contingency_p:.3e}\n')

    print('Per-label chi2 test:')
    print(chi2_per_label_test(hypertopic_distribution, articles_per_SES))


HT_MOVIE, HT_SPORT, HT_MUSIC, HT_LIFE = 'movie', 'sport', 'music', 'life'
hypertopics = [HT_MOVIE, HT_SPORT, HT_MUSIC, HT_LIFE]
hypertopic_table = {
    4: [HT_LIFE, HT_LIFE, HT_SPORT, HT_MOVIE], 
    5: [HT_LIFE, HT_SPORT, HT_LIFE, HT_MUSIC, HT_LIFE],
    6: [HT_MUSIC, HT_LIFE, HT_LIFE, HT_LIFE, HT_SPORT, HT_LIFE],
    10: [
        HT_LIFE, HT_LIFE, HT_SPORT, HT_SPORT, HT_SPORT,  # 0
        HT_MOVIE, HT_SPORT, HT_MUSIC, HT_MUSIC, HT_LIFE,  #5
    ],
    15: [
        HT_LIFE, HT_LIFE, HT_SPORT, HT_MOVIE, HT_MUSIC,  # 0
        HT_MOVIE, HT_LIFE, HT_SPORT, HT_SPORT, HT_LIFE,  # 5
        HT_LIFE, HT_MOVIE, HT_SPORT, HT_LIFE, HT_SPORT,  # 10
    ],
    20: [
        HT_MOVIE, HT_LIFE, HT_SPORT, HT_LIFE, HT_MUSIC,  # 0
        HT_MOVIE, HT_LIFE, HT_LIFE, HT_SPORT, HT_MOVIE,  # 5
        HT_LIFE, HT_SPORT, HT_SPORT, HT_LIFE, HT_MUSIC,  # 10
        HT_SPORT, HT_LIFE, HT_SPORT, HT_LIFE, HT_LIFE,  # 15
    ],
    25: [
        HT_LIFE, HT_MOVIE, HT_LIFE, HT_SPORT, HT_LIFE,  # 0
        HT_SPORT, HT_LIFE, HT_LIFE, HT_LIFE, HT_MOVIE,  # 5
        HT_MOVIE, HT_SPORT, HT_LIFE, HT_SPORT, HT_SPORT,  # 10
        HT_MUSIC, HT_MUSIC, HT_MOVIE, HT_MOVIE, HT_LIFE,  # 15
        HT_LIFE, HT_SPORT, HT_MUSIC, HT_LIFE, HT_LIFE,  # 20
    ],
    30: [
        HT_SPORT, HT_LIFE, HT_LIFE, HT_SPORT, HT_MUSIC,  # 0
        HT_LIFE, HT_LIFE, HT_SPORT, HT_MUSIC, HT_LIFE,  # 5
        HT_SPORT, HT_LIFE, HT_LIFE, HT_MUSIC, HT_MOVIE,  # 10
        HT_SPORT, HT_LIFE, HT_MUSIC, HT_LIFE, HT_LIFE,  # 15
        HT_SPORT, HT_LIFE, HT_MOVIE, HT_MOVIE, HT_MOVIE,  # 20
        HT_SPORT, HT_LIFE, HT_SPORT, HT_LIFE, HT_LIFE,  # 25
    ],
    35: [
        HT_SPORT, HT_LIFE, HT_MUSIC, HT_SPORT, HT_MOVIE,  # 0
        HT_MUSIC, HT_MUSIC, HT_MOVIE, HT_LIFE, HT_LIFE,  # 5
        HT_MUSIC, HT_MOVIE, HT_SPORT, HT_LIFE, HT_SPORT,  # 10
        HT_LIFE, HT_MUSIC, HT_MOVIE, HT_LIFE, HT_MUSIC,  # 15
        HT_MOVIE, HT_SPORT, HT_LIFE, HT_LIFE, HT_SPORT,  # 20
        HT_SPORT, HT_LIFE, HT_LIFE, HT_LIFE, HT_MUSIC,  # 25
        HT_MUSIC, HT_SPORT, HT_LIFE, HT_LIFE, HT_LIFE,  # 30
    ],
    40: [
        HT_MOVIE, HT_MUSIC, HT_LIFE, HT_SPORT, HT_LIFE,  # 0
        HT_MUSIC, HT_LIFE, HT_LIFE, HT_MOVIE, HT_SPORT,  # 5
        HT_LIFE, HT_LIFE, HT_LIFE, HT_LIFE, HT_SPORT,  # 10
        HT_MUSIC, HT_SPORT, HT_LIFE, HT_LIFE, HT_LIFE,  # 15
        HT_LIFE, HT_SPORT, HT_MOVIE, HT_LIFE, HT_LIFE,  # 20
        HT_SPORT, HT_LIFE, HT_SPORT, HT_MUSIC, HT_LIFE,  # 25
        HT_LIFE, HT_MUSIC, HT_MUSIC, HT_MOVIE, HT_MOVIE,  # 30
        HT_LIFE, HT_LIFE, HT_SPORT, HT_SPORT, HT_MOVIE,  # 35
    ],
    45: [
        HT_LIFE, HT_LIFE, HT_SPORT, HT_MUSIC, HT_LIFE,  #0
        HT_MUSIC, HT_LIFE, HT_SPORT, HT_MUSIC, HT_LIFE,  # 5
        HT_LIFE, HT_LIFE, HT_SPORT, HT_LIFE, HT_MUSIC,  # 10
        HT_SPORT, HT_LIFE, HT_MOVIE, HT_LIFE, HT_MOVIE,  # 15
        HT_SPORT, HT_MUSIC, HT_LIFE, HT_LIFE, HT_MOVIE,  # 20
        HT_SPORT, HT_MUSIC, HT_LIFE, HT_MUSIC, HT_SPORT,  # 25
        HT_LIFE, HT_LIFE, HT_LIFE, HT_LIFE, HT_LIFE,  # 30
        HT_MOVIE, HT_LIFE, HT_LIFE, HT_LIFE, HT_LIFE,  # 35
        HT_LIFE, HT_MOVIE, HT_LIFE, HT_LIFE, HT_MUSIC,  # 40
    ],
    50: [
        HT_MUSIC, HT_SPORT, HT_LIFE, HT_MUSIC, HT_MOVIE,  # 0
        HT_LIFE, HT_LIFE, HT_MOVIE, HT_LIFE, HT_LIFE,  # 5
        HT_LIFE, HT_SPORT, HT_MOVIE, HT_LIFE, HT_LIFE,  # 10
        HT_MUSIC, HT_LIFE, HT_MUSIC, HT_LIFE, HT_LIFE,  # 15
        HT_MUSIC, HT_SPORT, HT_MOVIE, HT_LIFE, HT_LIFE,  # 20
        HT_MOVIE, HT_LIFE, HT_LIFE, HT_MUSIC, HT_LIFE,  # 25
        HT_LIFE, HT_SPORT, HT_LIFE, HT_SPORT, HT_LIFE,  # 30
        HT_MUSIC, HT_LIFE, HT_LIFE, HT_MOVIE, HT_LIFE,  # 35
        HT_LIFE, HT_LIFE, HT_MOVIE, HT_LIFE, HT_LIFE,  # 40
        HT_MUSIC, HT_MOVIE, HT_MOVIE, HT_MUSIC, HT_MUSIC,  # 45
    ]

}
hypertopics_columns = [f'cluster_{n}' for n in hypertopic_table]