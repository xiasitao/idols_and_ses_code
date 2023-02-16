# %%
import pandas as pd
import numpy as np

from pathlib import Path
SOURCE_PATH = Path(__file__).parent.resolve()
ASSET_PATH = SOURCE_PATH.joinpath('..', '..', 'assets').resolve()
BUILD_PATH = SOURCE_PATH.joinpath("..", "..", "build").resolve()


# %%
articles = pd.read_pickle(BUILD_PATH / 'articles/articles.pkl')
articles_balanced_50 = pd.read_pickle(BUILD_PATH / 'articles/articles_balanced_50.pkl')
scores = pd.read_pickle(BUILD_PATH / 'role_models/ses_scores.pkl')

articles_en = articles[articles['language_ml']=='en']
articles_en_only_scores = articles_en.join(scores, on='role_model', how='inner')

articles_balanced_50_en = articles_balanced_50[articles_balanced_50['language_ml']=='en']
articles_balanced_50_en_only_scores = articles_balanced_50_en.join(scores, on='role_model', how='inner')


# %%
print(f'Articles: #en={len(articles_en)}, #rm={len(articles_en["role_model"].unique())}')
print(f'Articles with scores: #en={len(articles_en_only_scores)}, #rm={len(articles_en_only_scores["role_model"].unique())}')
print(f'Balanced: #en={len(articles_balanced_50_en)} #rm={len(articles_balanced_50_en["role_model"].unique())}')
print(f'Balanced only scores: #en={len(articles_balanced_50_en_only_scores)} #rm={len(articles_balanced_50_en_only_scores["role_model"].unique())}')
# %%
