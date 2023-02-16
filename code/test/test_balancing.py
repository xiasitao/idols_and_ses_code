# %%
import pandas as pd
from pathlib import Path
SOURCE_PATH = Path(__file__).parent.resolve()
ASSET_PATH = SOURCE_PATH.joinpath('..', '..', 'assets').resolve()
BUILD_PATH = SOURCE_PATH.joinpath("..", "..", "build").resolve()


# %%
balanced_50 = pd.read_pickle(BUILD_PATH / 'articles/articles_balanced_50.pkl')


# %%
balanced_50.groupby(['language_ml', 'role_model']).count()


# %%
balanced_50.groupby(['language_ml']).count()


# %%
balanced_50.dtypes


# %%
balanced_50[balanced_50['language_ml']=='en'][['article_id', 'content']].drop_duplicates().count()
# %%