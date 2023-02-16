# %%
from pathlib import Path
SOURCE_PATH = Path(__file__).parent.resolve()
ASSET_PATH = SOURCE_PATH.joinpath('..', '..', 'assets').resolve()
BUILD_PATH = SOURCE_PATH.joinpath("..", "..", "build").resolve()

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sentence_transformers as st
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids


 # %%
articles = pd.read_pickle(BUILD_PATH / 'articles/articles_balanced_50.pkl')
ses_scores = pd.read_pickle(BUILD_PATH / 'role_models/ses_scores.pkl')
articles_en = articles[articles.language_ml == 'en']
articles_en = articles_en.join(ses_scores[['average_ses', 'rank_weighted_ses', 'significance_weighted_ses']], on='role_model', how='inner')


# %%
sentence_bert_encoder = st.SentenceTransformer('all-MiniLM-L6-v2')
embeddings_path = BUILD_PATH / 'semantic_similarity/sbert_embeddings_en.pkl'
if embeddings_path.exists():
    with open(embeddings_path, 'rb') as file:
        embeddings = pickle.load(file)
else:
    embeddings = sentence_bert_encoder.encode(articles_en['content'].to_list())
    # embeddings = sentence_bert_encoder.encode(subset_en['content_slim'].to_list())
    with open(embeddings_path, 'wb') as file:
        pickle.dump(embeddings, file)


# %%
embeddings_50d = PCA(n_components=50).fit_transform(embeddings)
embeddings_2d = TSNE(n_components=2, random_state=42).fit_transform(embeddings_50d)
embeddings_3d = TSNE(n_components=3, random_state=42).fit_transform(embeddings_50d)
PCA_embeddings_2d = PCA(n_components=2).fit_transform(embeddings_50d)
PCA_embeddings_3d = PCA(n_components=3).fit_transform(embeddings_50d)
PCA_embeddings_4d = PCA(n_components=4).fit_transform(embeddings_50d)
# %%
plt.title('Embeddings')
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=6)
plt.show()


# %%
kmedoids = KMedoids(n_clusters=5).fit(embeddings_3d)
kmedoids_clusters = kmedoids.predict(embeddings_3d)
medoids = kmedoids.medoid_indices_
articles_en['cluster'] = kmedoids_clusters
# %%
plt.title('KMedoids')
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=6, c=articles_en['cluster'])
plt.scatter(embeddings_2d[medoids][:, 0], embeddings_2d[medoids][:, 1], marker='*', color='red')
# %%
cluster_professions = articles_en.groupby(['cluster', 'profession']).agg({'content': 'count'}).sort_values(['cluster', 'content'], ascending=False)
cluster_professions.groupby('cluster')['content'].nlargest(3)


# %%
kmeans = KMeans(n_clusters=10)
kmeans.fit_transform(embeddings_3d)
kmeans_clusters = kmeans.predict(embeddings_3d)
articles_en['cluster'] = kmeans_clusters
# %%
plt.title('KMeans')
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=6, c=articles_en['cluster'])
# %%
cluster_professions = articles_en.groupby(['cluster', 'profession']).agg({'content': 'count'}).sort_values(['cluster', 'content'], ascending=False)
cluster_professions.groupby('cluster')['content'].nlargest(3)
# %%
cluster_ses = articles_en.groupby(['cluster'])['average_ses'].mean()
cluster_ses_std = articles_en.groupby(['cluster'])['average_ses'].std()
pd.DataFrame(data={'ses': cluster_ses, 'std': cluster_ses_std})

#articles_en['average_ses'].std()

# %%
# SES
plt.title('SES')
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=6, c=articles_en['significance_weighted_ses'])
plt.colorbar()
# %%

