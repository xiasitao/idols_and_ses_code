# %%
import pandas as pd
import pickle
import matplotlib.pyplot as plt

from pathlib import Path
SOURCE_PATH = Path(__file__).parent.resolve()
ASSET_PATH = SOURCE_PATH.joinpath('..', '..', 'assets').resolve()
BUILD_PATH = SOURCE_PATH.joinpath("..", "..", "build").resolve()


# %%
with open(BUILD_PATH / 'semantic_similarity/sbert_vectors.pkl', 'rb') as file:
    embeddings = pickle.load(file)
embeddings


# %%
semantic_clusters = pd.read_pickle(BUILD_PATH / 'semantic_similarity/semantic_clusters.pkl')
semantic_clusters


# %%
with open(BUILD_PATH / 'semantic_similarity/semantic_topics.pkl', 'rb') as file:
    semantic_topics = pickle.load(file)
semantic_topics
# %%

plt.scatter(embeddings['sbert_2'][:, 0], embeddings['sbert_2'][:, 1], 5, c=semantic_clusters['cluster_5'])

# %%
