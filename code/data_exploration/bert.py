# %%
from pathlib import Path
SOURCE_PATH = Path(__file__).parent.resolve()
ASSET_PATH = SOURCE_PATH.joinpath('..', '..', 'assets').resolve()
BUILD_PATH = SOURCE_PATH.joinpath("..", "..", "build").resolve()

import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


#%%
articles = pd.read_pickle(BUILD_PATH / 'articles/articles.pkl')
articles_en = articles[articles.language_ml == 'en']
subset_en = articles_en.iloc[0:5000]


# %%
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')


# %%
texts = ["I ate an apple.", "The startup was acquired by Apple."]
ids = tokenizer(texts, padding=True)['input_ids']
# %%
tokenizer.convert_ids_to_tokens(ids[0])

# %%
ids
# %%
