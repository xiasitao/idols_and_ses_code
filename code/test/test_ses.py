# %%
import pandas as pd
from pathlib import Path
SOURCE_PATH = Path(__file__).parent.resolve()
ASSET_PATH = SOURCE_PATH.joinpath('..', '..', 'assets').resolve()
BUILD_PATH = SOURCE_PATH.joinpath("..", "..", "build").resolve()


# %%
ses_data = pd.read_pickle(BUILD_PATH / 'role_models/ses.pkl')
mention_data = pd.read_pickle(BUILD_PATH / 'role_models/ses_mentions.pkl')
score_data = pd.read_pickle(BUILD_PATH / 'role_models/ses_scores.pkl')
score_data_filtered = pd.read_pickle(BUILD_PATH / 'role_models/ses_scores_distinct.pkl')
ses_scores_balanced = pd.read_pickle(BUILD_PATH / 'role_models/ses_scores_balanced.pkl')
# %%
ses_data
# %%
mention_data
# %%
score_data
# %%
print(score_data_filtered['average_ses'].unique())
score_data_filtered.groupby('average_ses').count()
# %%
ses_scores_balanced.groupby('average_ses').count()
# %%
