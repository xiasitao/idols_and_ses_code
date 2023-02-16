# %%
from pathlib import Path
SOURCE_PATH = Path(__file__).parent.resolve()
ASSET_PATH = SOURCE_PATH.joinpath('..', '..', 'assets').resolve()
BUILD_PATH = SOURCE_PATH.joinpath("..", "..", "build").resolve()

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from sklearn.utils import shuffle
# %%
# Load data
articles = pd.read_pickle(BUILD_PATH / 'articles/articles.pkl')
articles_en = articles[articles.language_ml == 'en']
with open(BUILD_PATH / 'articles/corpora.pkl', 'rb') as file:
    corpora = pickle.load(file)
with open(BUILD_PATH / 'articles/sentence_tokens.pkl', 'rb') as file:
    sentence_tokens = pickle.load(file)
with open(BUILD_PATH / 'articles/word_statistic.pkl', 'rb') as file:
    word_statistics = pickle.load(file)

# %%
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
    new_data = pd.DataFrame(data=None, columns=list(data.columns)+['orig_id'])
    dtypes['orig_id'] = pd.StringDtype()
    new_data = new_data.astype(dtypes)

    # Check for each role model if downsampling or upsampling are necessary
    for role_model in data['role_model'].unique():
        role_model_data = shuffle(data[data['role_model'] == role_model], random_state=42)
        role_model_data['orig_id'] = role_model_data.index
        if downsample and len(role_model_data) > n_target:
            role_model_data = role_model_data.iloc[0:n_target].reset_index(drop=True)
        if upsample and len(role_model_data) < n_target:
            full_repetitions = n_target // len(role_model_data) + 1
            if max_upsampling_factor is not None and full_repetitions > max_upsampling_factor:
                continue
            role_model_data = role_model_data.loc[[*role_model_data.index]*full_repetitions].reset_index(drop=True).iloc[0:n_target]
        new_data = pd.concat([new_data, role_model_data], ignore_index=True)
    return new_data
balanced = balance_role_models(articles_en, max_upsampling_factor=15)


# %%
articles['role_model'].nunique(), balanced['role_model'].nunique()


# %%
# Article per role model distribution
role_models, counts = np.unique(articles['role_model'], return_counts=True)
role_models_balanced, counts_balanced = np.unique(balanced['role_model'], return_counts=True)
sorted_counts, sorted_counts_balanced = np.sort(counts), np.sort(counts_balanced)
counts_cumulative, counts_cumulative_balanced = sorted_counts.cumsum(), sorted_counts_balanced.cumsum()
plt.title('Articles per role model')
plt.xlabel('percentile')
plt.ylabel('Cumulative relative amount of articles')
plt.plot(np.arange(counts_cumulative.size)/(counts_cumulative.size)*100, counts_cumulative/np.max(counts_cumulative)*100, label='data')
plt.plot(np.arange(counts_cumulative_balanced.size)/(counts_cumulative_balanced.size)*100, counts_cumulative_balanced/np.max(counts_cumulative_balanced)*100, label='balanced data')
plt.gca().xaxis.set_major_formatter(PercentFormatter())
plt.gca().yaxis.set_major_formatter(PercentFormatter())
plt.margins(x=0, y=0)
plt.grid()
plt.legend()
plt.show()


# %%
# Language distribution
role_models, counts = np.unique(articles['language_ml'], return_counts=True)
plt.title('Language distribution')
plt.bar(role_models, counts/np.sum(counts)*100, align='center')
plt.gca().yaxis.set_major_formatter(PercentFormatter())
plt.show()


# %%
# Article and role model gender
article_genders, counts = np.unique(articles['sex'].dropna(), return_counts=True)
plt.title('Article gender distribution')
plt.bar(['male', 'female'], counts/sum(counts)*100, align='center')
plt.gca().yaxis.set_major_formatter(PercentFormatter())
plt.show()

role_model_genders, counts = np.unique(articles[['role_model', 'sex']].drop_duplicates()['sex'].dropna(), return_counts=True)
plt.title('Role model gender distribution')
plt.bar(['male', 'female'], counts/sum(counts)*100, align='center')
plt.gca().yaxis.set_major_formatter(PercentFormatter())
plt.show()


# %%
