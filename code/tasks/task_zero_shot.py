""" This script is best executed with CUDA / on colab with CUDA.
    You need to adapt the paths to your Colab directory structure.
"""
import pytask
import pandas as pd
import numpy as np
from transformers import pipeline

from pathlib import Path
SOURCE_PATH = Path(__file__).parent.resolve()
ASSET_PATH = SOURCE_PATH.joinpath('..', '..', 'assets').resolve()
BUILD_PATH = SOURCE_PATH.joinpath("..", "..", "build").resolve()

CLASSIFICATION_CATEGORIES = {
    'article_type': ['news', 'report', 'opinion', 'entertainment'],
    'difficulty': ['easy', 'difficult'],
    'crime': ['innocent', 'criminal'],
    'crime_type': ['innocent', 'drugs', 'violence', 'sexual assault'],
    'emotion': ['sadness', 'happiness', 'fear', 'anger', 'surprise', 'disgust'],
    'prosociality': ['prosocial', 'antisocial'], 
    'relatability': ['relatable', 'distant'], 
    'sentiment': ['positive', 'negative'],
    'sentiment_n': ['positive', 'neutral', 'negative'],
    'success': ['success', 'failure'],  # NEW
    'topic': ['movie', 'music', 'sport', 'life'],
    'writing_style': ['expository', 'persuasive', 'narrative', 'embellished'],
}


def zs_classify_articles(model: pipeline, articles: list, candidate_labels: list) -> tuple:
    """Classify a batch of articles using a pipeline against a list of candidate topics.

    Args:
        model (pipeline): Huggingface zero-shot classification pipeline
        articles (list): List of articles to classify
        candidate_labels (list): List of possible labels

    Returns:
        tuple: List of labels, list of entropies, list of probabilities of the assigned label
    """    
    results = model(articles, candidate_labels)
    assigned_labels = [result['labels'][0] for result in results]
    score_list = np.array([result['scores'] for result in results])
    entropies = -np.sum(score_list * np.log(score_list), axis=1)
    probabilities = score_list[:, 0]
    return assigned_labels, entropies, probabilities


ZERO_SHOT_FILENAME_WILDCARD  = 'zero_shot_classification/zero_shot_classification_{}.pkl'
@pytask.mark.skip()  # Execute with CUDA on Colab
@pytask.mark.depends_on(BUILD_PATH / 'articles/articles_balanced_50.pkl')
@pytask.mark.parametrize(
    "category, candidate_labels, produces, n_articles, incremental",
    [(
        category,
        CLASSIFICATION_CATEGORIES[category],
        BUILD_PATH / ZERO_SHOT_FILENAME_WILDCARD.format(category),
        10,  # n_articles
        True,  # incremental
    ) for category in CLASSIFICATION_CATEGORIES]
)
def task_zero_shot_classification(category: str, candidate_labels: list, produces: Path, n_articles=10, incremental=True, device=None):
    """Perform zero-shot classification on the 50-articles-per-role-model data set.
    Perform with CUDA or on Colab (with CUDA) by setting the device to "cuda:0".

    Args:
        produces (Path): Output file path
        n_articles (int, optional): Number of articles to process in this call. Defaults to 10.
        incremental (bool, optional): Whether to only process articles that have not been processed yet. Defaults to False.
    """
    incremental = incremental and produces.exists()
    existing_data = None if not incremental else pd.read_pickle(produces)

    articles = pd.read_pickle(BUILD_PATH / 'articles/articles_balanced_50.pkl')
    articles_en = articles[articles['language_ml'] == 'en']
    zs_classifier = pipeline(
        'zero-shot-classification',
        model='facebook/bart-large-mnli',
        device=device
    )
    
    article_data = articles_en[['article_id', 'content']].drop_duplicates()
    if incremental:
        article_data = article_data[~(article_data['article_id'].isin(existing_data.index))]
    if n_articles is not None and n_articles > 0:
        article_data = article_data.sample(n=min(n_articles, len(article_data)), random_state=42)

    if len(article_data) > 0:
        print(f'Performing "{category}" ZSC on {len(article_data)} articles ({"incremental, excluding "+str(len(existing_data)) if incremental else "from start"}).')
    else:
        print(f'No articles to perform ZSC on, aborting.')
        return
    
    classification_data = pd.DataFrame(
        data=None,
        index=article_data['article_id'],
        columns=[category, f'{category}_entropy', f'{category}_p']
    )

    labels, entropies, probabilities = zs_classify_articles(
        model=zs_classifier,
        articles=article_data['content'].to_list(),
        candidate_labels=candidate_labels
    )
    classification_data[category] = labels
    classification_data[f'{category}_entropy'] = entropies
    classification_data[f'{category}_p'] = probabilities

    if incremental:
        classification_data = pd.concat([existing_data, classification_data])
    classification_data.to_pickle(produces)


if __name__ == '__main__':
    category = 'topic'
    task_zero_shot_classification(
        category=category,
        candidate_labels=CLASSIFICATION_CATEGORIES[category],
        produces=BUILD_PATH / ZERO_SHOT_FILENAME_WILDCARD.format(category),
        n_articles=2,
        incremental=True,
    )