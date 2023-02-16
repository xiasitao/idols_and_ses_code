# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
SOURCE_PATH = Path(__file__).parent.resolve()
ASSET_PATH = SOURCE_PATH.joinpath('..', '..', 'assets').resolve()
BUILD_PATH = SOURCE_PATH.joinpath("..", "..", "build").resolve()


# %%
articles = pd.read_pickle(BUILD_PATH / 'articles/articles_balanced_50.pkl')
# ses = pd.read_pickle(BUILD_PATH / 'role_models/ses_scores_distinct.pkl')
# articles = articles.join(ses, how='inner', on='role_model')

topic_data = pd.read_pickle(BUILD_PATH / 'zero_shot_classification/zero_shot_classification_topic.pkl')
articles = articles.join(topic_data, how='inner', on='article_id')


# %%
# Sanity checks
# Check if for each role model, there are exactly 50 articles (uncomment after every articles has gone through zsc)
# assert(articles.groupby('role_model').count()['content'].unique() == np.array([50]))


# %%
topic_data


# %%
# Show correlation of entropy and topic probability
plt.title('Label probability and entropy')
plt.xlabel('label probability')
plt.ylabel('entropy')
plt.scatter(topic_data['topic_p'], topic_data['topic_entropy'])
plt.show()


# %%
