import pytask
import pickle
import pandas as pd
from sklearn.utils import shuffle

from pathlib import Path
SOURCE_PATH = Path(__file__).parent.resolve()
ASSET_PATH = SOURCE_PATH.joinpath('..', '..', 'assets').resolve()
BUILD_PATH = SOURCE_PATH.joinpath("..", "..", "build").resolve()


def balance_role_models(data, n_target = 50, downsample=True, upsample=True, max_upsampling_factor=None):
    """Balance the number of articles for all role models in a data set.
    Does not distinguish between languages.

    Args:
        data (pd.DataFrame): Articles to balance
        n_target (int, optional): Number of articles per role model. Defaults to 50.
        downsample (bool, optional): Whether to downsample if too many articles are present for a role model. Defaults to True.
        upsample (bool, optional): Whether to upsample if too few articles are present for a role model. Defaults to True.
        max_upsampling_factor (int, optional): Maximum factor by which a role model's articles should be upsampled. Role models where the target can't be met are discarded. Default None:

    Returns:
        pd.DataFrame: Data frame with balanced numbers of articles per role model
    """
    dtypes = data.dtypes
    new_data = pd.DataFrame(data=None, columns=list(data.columns)+['article_id'])
    dtypes['article_id'] = pd.StringDtype()
    new_data = new_data.astype(dtypes)

    # Check for each role model if downsampling or upsampling are necessary
    for role_model in data['role_model'].unique():
        role_model_data = shuffle(data[data['role_model'] == role_model], random_state=42)
        role_model_data['article_id'] = role_model_data.index
        if downsample and len(role_model_data) > n_target:
            role_model_data = role_model_data.iloc[0:n_target].reset_index(drop=True)
        if upsample and len(role_model_data) < n_target:
            full_repetitions = n_target // len(role_model_data) + 1
            if max_upsampling_factor is not None and full_repetitions > max_upsampling_factor:
                continue
            role_model_data = role_model_data.loc[[*role_model_data.index]*full_repetitions].reset_index(drop=True).iloc[0:n_target]
        new_data = pd.concat([new_data, role_model_data], ignore_index=True)
    return new_data


@pytask.mark.depends_on(BUILD_PATH / 'articles/articles.pkl')
@pytask.mark.parametrize(
    "n, produces",
    [(n, BUILD_PATH / f'articles/articles_balanced_{n}.pkl') for n in (50, 100, 200, 500)]
)
def task_balancing(produces: Path, n: int):
    """This task balances the number of article per role model and language by downsampling and upsampling.
    It provides data sets with 50, 100, 200, and 500 articles per role model.

    Args:
        n (int): Number of articles per role model and language.
        produces (Path): Path to respective target file.
    """    
    articles = pd.read_pickle(BUILD_PATH / 'articles/articles.pkl')
    balanced_articles = None
    for language in articles['language_ml'].unique():
        language_articles = articles[articles['language_ml'] == language]
        balanced_language_articles = balance_role_models(data=language_articles, n_target=n, downsample=True, upsample=True, max_upsampling_factor=10)
        balanced_articles = balanced_language_articles if balanced_articles is None else pd.concat([balanced_articles, balanced_language_articles])
    
    balanced_articles = balanced_articles.reset_index(drop=True)
    balanced_articles.to_pickle(produces)

