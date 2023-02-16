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

from transformers import pipeline


# %%
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


# %%
articles = pd.read_pickle(BUILD_PATH / 'articles/articles_balanced_50.pkl')
ses_scores = pd.read_pickle(BUILD_PATH / 'role_models/ses_scores_balanced.pkl')
articles_en = articles[articles.language_ml == 'en']
ses_scores = equilibrate_role_models(articles_en, ses_scores)
articles_en = articles_en.join(ses_scores[['average_ses', 'rank_weighted_ses', 'significance_weighted_ses', 'prevalent_ses']], on='role_model', how='inner')


# %%
zs_classifier = pipeline(
    'zero-shot-classification',
    model='facebook/bart-large-mnli'
)
candidate_labels = ['emotional', 'serious', 'joyful']


# %%
%%time
zs_classifier(articles_en['content'].iloc[0:2].to_list(), candidate_labels)
# %%
def zs_classify_articles(model: pipeline, articles: list, candidate_labels: list) -> tuple:
    """Classify a batch of articles using a pipeline against a list of candidate topics.

    Args:
        model (pipeline): Huggingface zero-shot classification pipeline
        articles (list): List of articles to classify
        candidate_labels (list): List of possible labels

    Returns:
        tuple: List of labels, list of entropies
    """    
    results = model(articles, candidate_labels)
    assigned_labels = [result['labels'][0] for result in results]
    score_list = np.array([result['scores'] for result in results])
    entropies = -np.sum(score_list * np.log(score_list), axis=1)
    return assigned_labels, entropies


# %%
%%time
zs_classify_articles(zs_classifier, articles_en['content'].iloc[0:3].to_list(), candidate_labels)
# %%
