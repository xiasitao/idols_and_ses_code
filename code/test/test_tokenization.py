# %%
import pickle
from pathlib import Path
SOURCE_PATH = Path(__file__).parent.resolve()
ASSET_PATH = SOURCE_PATH.joinpath('..', '..', 'assets').resolve()
BUILD_PATH = SOURCE_PATH.joinpath("..", "..", "build").resolve()
# %%
with open(BUILD_PATH / 'articles/sentence_tokens.pkl', 'rb') as file:
    sentence_tokens = pickle.load(file)
# %%
sentence_tokens
# %%
with open(BUILD_PATH / 'articles/word_statistic.pkl', 'rb') as file:
    word_statistics = pickle.load(file)
# %%
word_statistics
# %%
# %%
word_statistics[
    (word_statistics['language'] == 'de')
    & ~(word_statistics['stopword'])
    & ~(word_statistics['punctuation'])
].iloc[50:100]
# %%
