import pytask
import pickle
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

from pathlib import Path
SOURCE_PATH = Path(__file__).parent.resolve()
ASSET_PATH = SOURCE_PATH.joinpath('..', '..', 'assets').resolve()
BUILD_PATH = SOURCE_PATH.joinpath("..", "..", "build").resolve()

@pytask.mark.skip()
@pytask.mark.depends_on(BUILD_PATH / 'articles/corpora.pkl')
@pytask.mark.produces(BUILD_PATH / 'articles/sentence_tokens.pkl')
def task_sentence_tokenization(produces):
    """Collect all sentences in the corpus per language.

    Args:
        produces (Path): Path to destination file
    """    
    with open(BUILD_PATH / 'articles/corpora.pkl', 'rb') as file:
        corpora = pickle.load(file)
    
    sentences = {}
    for language in corpora:
        corpus = corpora[language]
        sentences[language] = sent_tokenize(corpus)
    
    with open(produces, 'wb') as file:
        pickle.dump(sentences, file)


@pytask.mark.skip()
@pytask.mark.depends_on(BUILD_PATH / 'articles/sentence_tokens.pkl')
@pytask.mark.produces(BUILD_PATH / 'articles/word_statistic.pkl')
def task_word_statistic(produces):
    """Find the occurrence statistic of all words of the corpus per language

    Args:
        produces (Path): Path to destination file
    """    
    with open(BUILD_PATH / 'articles/sentence_tokens.pkl', 'rb') as file:
        sentence_tokens = pickle.load(file)
    
    frequencies = pd.DataFrame(columns=['language', 'frequency', 'stopword', 'punctuation'])
    for language in sentence_tokens:
        language_frequencies = pd.DataFrame(columns=['language', 'frequency', 'stopword', 'punctuation'])
        words_statistic = {}
        sentences = sentence_tokens[language]
        for i, sentence in enumerate(sentences):
            for word in word_tokenize(sentence, preserve_line=True):
                word = word.lower()
                if word in words_statistic:
                    words_statistic[word] += 1
                else:
                    words_statistic[word] = 1
        language_frequencies['frequency'] = pd.Series(words_statistic, name='frequency')
        language_frequencies['language'] = language
        language_frequencies['punctuation'] = language_frequencies.index.str.match(r'^[.!?;:,-/+*&()\[\]{}"\'<>`#»«©]|``$')
        language_str = 'german' if language == 'de' else 'english' # TODO more general
        language_frequencies['stopword'] = language_frequencies.index.isin(stopwords.words(language_str))

        frequencies = pd.concat([frequencies, language_frequencies])

    frequencies['language'] = frequencies['language'].astype(str)
    frequencies['frequency'] = frequencies['frequency'].astype(int)
    frequencies['stopword'] = frequencies['stopword'].astype(bool)
    frequencies['punctuation'] = frequencies['punctuation'].astype(bool)

    frequencies = frequencies.sort_values(['language', 'frequency'], ascending=[True, False])

    with open(produces, 'wb') as file:
        pickle.dump(frequencies, file)

if __name__ == '__main__':
    raise Exception('Can only be executed by pytask')