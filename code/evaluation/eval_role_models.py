# %%
import pandas as pd
import numpy as np

from pathlib import Path
SOURCE_PATH = Path(__file__).parent.resolve()
ASSET_PATH = SOURCE_PATH.joinpath('..', '..', 'assets').resolve()
BUILD_PATH = SOURCE_PATH.joinpath("..", "..", "build").resolve()


# %%
ses = pd.read_pickle(BUILD_PATH / 'role_models/ses.pkl')
mentions = pd.read_pickle(BUILD_PATH / 'role_models/ses_mentions.pkl')
scores = pd.read_pickle(BUILD_PATH / 'role_models/ses_scores.pkl')
scores_distinct = pd.read_pickle(BUILD_PATH / 'role_models/ses_scores_distinct.pkl')
role_model_data = pd.read_pickle(BUILD_PATH / 'role_models/role_model_data.pkl')
articles_balanced_50 = pd.read_pickle(BUILD_PATH / 'articles/articles_balanced_50.pkl')
articles_balanced_50 = articles_balanced_50[articles_balanced_50['language_ml']=='en']

mentioned_role_model_data = role_model_data[role_model_data.index.isin(scores.index)]
scores_after_balancing = scores[scores.index.isin(articles_balanced_50['role_model'])]
scores_distinct_after_balancing = scores_distinct[scores_distinct.index.isin(articles_balanced_50['role_model'])]

# %%
print('Number of children with sensible role models:', len(ses))
print('Number of mentions: ', len(mentions))
print('Mentions by rank:', mentions.groupby('rank').count()['id'])


# %%
print('Number of mentioned role models:', len(mentioned_role_model_data))
print('Role models by gender:', [f'{gender}: {number}' for (gender, number) in zip(*np.unique(mentioned_role_model_data['sex'].fillna(2), return_counts=True))])
print('Role models by count:', [f'{gender}: {number}' for (gender, number) in zip(*np.unique(scores['count'], return_counts=True))])

# %%
print(f'General set: #={len(scores)}, #low={len(scores[scores["low_ses"]==True])}, #high={len(scores[scores["high_ses"]==True])} #high&low={len(scores[(scores["low_ses"]==True) & (scores["high_ses"]==True)])}')
print(f'General set after balancing: #={len(scores_after_balancing)}, #low={len(scores_after_balancing[scores_after_balancing["low_ses"]==True])}, #high={len(scores_after_balancing[scores_after_balancing["high_ses"]==True])} #high&low={len(scores_after_balancing[(scores_after_balancing["low_ses"]==True) & (scores_after_balancing["high_ses"]==True)])}')


# %%

print(f'Distinct set: #={len(scores_distinct)}, #low={len(scores_distinct[scores_distinct["low_ses"]==True])}, #high={len(scores_distinct[scores_distinct["high_ses"]==True])}')
print(f'Distinct set after balancing: #={len(scores_distinct_after_balancing)}, #low={len(scores_distinct_after_balancing[scores_distinct_after_balancing["low_ses"]==True])}, #high={len(scores_distinct_after_balancing[scores_distinct_after_balancing["high_ses"]==True])}')


# %%
