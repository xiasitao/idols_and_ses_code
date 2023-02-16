# %%
from pathlib import Path
SOURCE_PATH = Path(__file__).parent.resolve()
ASSET_PATH = SOURCE_PATH.joinpath('..', '..', 'assets').resolve()
BUILD_PATH = SOURCE_PATH.joinpath("..", "..", "build").resolve()

import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from transformers import AutoTokenizer

from sklearn.manifold import TSNE


# %%
articles = pd.read_pickle(BUILD_PATH / 'articles/articles.pkl')
articles_en = articles[articles.language_ml == 'en']
articles_en['contains_music'] = articles_en['content'].str.contains(r'\bmusic\b')
articles_en['contains_film'] = articles_en['content'].str.contains(r'\b(films?|movies?|actors?)\b')
articles_en['contains_game'] = articles_en['content'].str.contains(r'\bgames?\b')
articles_en['contains_work'] = articles_en['content'].str.contains(r'\bworks?\b')
subset_en = articles_en.iloc[0:5000]


#%%
# Count Vectorize slim
count_slim_embedding = CountVectorizer().fit_transform(subset_en['content_slim'])
count_slim_embedding_2d = TSNE(n_components=2, init='random', random_state=42).fit_transform(count_slim_embedding)
# %%
plt.title('Count vectorizer with slim text')
plt.scatter(count_slim_embedding_2d[:, 0], count_slim_embedding_2d[:, 1], s=6, c=subset_en['contains_work'].map({True: 'red', False: 'grey'}))
plt.show()


# %%
# BERT slim
bert_slim_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
bert_slim_embedding = bert_slim_tokenizer.batch_encode_plus(subset_en['content_slim'].to_list(), truncation=True, padding=True)
bert_slim_embedding_values = np.array(bert_slim_embedding['input_ids'])
bert_slim_embedding_2d = TSNE(n_components=2, random_state=42).fit_transform(bert_slim_embedding_values)
bert_slim_embedding_3d = TSNE(n_components=3, random_state=42).fit_transform(bert_slim_embedding_values)
# %%
plt.title('BERT with slim text')
plt.scatter(bert_slim_embedding_2d[:,0], bert_slim_embedding_2d[:,1], s=6, c=subset_en['contains_music'].map({True: 'red', False: 'grey'}))
plt.show()
# %%
outliers = (bert_slim_embedding_3d**2).sum(axis=1)**(1/2) < 30
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_title('BERT with slim text')
ax.scatter(bert_slim_embedding_3d[outliers][:,0], bert_slim_embedding_3d[outliers][:,1], bert_slim_embedding_3d[outliers][:,2], s=6, c=subset_en[outliers]['contains_music'].map({True: 'red', False: 'grey'}))
plt.show()
# %%
# Interesting indices: Top right: 1, 4. Bottom left: 5, 12. Center: 10, 25, 44, 47
plt.title('BERT with slim text')
plt.scatter(bert_slim_embedding_2d[:,0], bert_slim_embedding_2d[:,1], s=6, c=pd.Series((subset_en.index == subset_en.index[47])).map({True: 'red', False: 'grey'}))
plt.show()
#%%
left_idx = (bert_slim_embedding_2d[:, 0] < -15)
upper_center_idx = (bert_slim_embedding_2d[:, 0] > -15) & (bert_slim_embedding_2d[:, 0] < 15) & (bert_slim_embedding_2d[:, 1] > 0)
lower_center_idx = (bert_slim_embedding_2d[:, 0] > -15) & (bert_slim_embedding_2d[:, 0] < 15) & (bert_slim_embedding_2d[:, 1] < 0)
right_idx = (bert_slim_embedding_2d[:, 0] > 15)

left_2d, upper_center_2d, lower_center_2d, right_2d = bert_slim_embedding_2d[left_idx], bert_slim_embedding_2d[upper_center_idx], bert_slim_embedding_2d[lower_center_idx], bert_slim_embedding_2d[right_idx]
left_data, upper_center_data, lower_center_data, right_data = subset_en[left_idx], subset_en[upper_center_idx], subset_en[lower_center_idx], subset_en[right_idx]


plt.title('BERT with slim text')
for label, data in zip(['left', 'upper center', 'lower center', 'right'], [left_2d, upper_center_2d, lower_center_2d, right_2d]):
    plt.scatter(data[:,0], data[:,1], s=6)
plt.show()
#%%
right_data['sentences_length'].replace(np.inf, np.nan).dropna().mean()
# right_data.columns

# %%
# BERT content
bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
bert_embedding = bert_tokenizer.batch_encode_plus(subset_en['content'].to_list(), truncation=True, padding=True)
bert_embedding_values = np.array(bert_embedding['input_ids'])
bert_embedding_2d = TSNE(n_components=2, random_state=42).fit_transform(bert_embedding_values)
# %%
plt.title('BERT with normal text')
plt.scatter(bert_embedding_2d[:,0], bert_embedding_2d[:,1], s=6, c=subset_en['contains_game'].map({True: 'red', False: 'grey'}))
plt.show()
# %%