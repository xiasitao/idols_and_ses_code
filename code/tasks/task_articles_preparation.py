import pytask
import re
import pickle
import pandas as pd
from pandarallel import pandarallel
pandarallel.initialize()
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import spacy

from pathlib import Path
SOURCE_PATH = Path(__file__).parent.resolve()
ASSET_PATH = SOURCE_PATH.joinpath('..', '..', 'assets').resolve()
BUILD_PATH = SOURCE_PATH.joinpath("..", "..", "build").resolve()

# Data cleaning
def _clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """Clean a dataframe by removing nans

    Args:
        data (pd.DataFrame): substrate

    Returns:
        pd.DataFrame: substrate
    """
    data = data.dropna(axis=0)
    return data


# Text cleaning
def _letter_shift(doc: str, shift: int = 1, common_replacements=True) -> str:
    """Shift all letters by a number of places in the alphabet

    Args:
        doc (str): substrate
        shift (int, optional): number of alphabet places to shift. Defaults to 1.
    """
    def apply_common_char_replacements(char: str) -> str:
        if char == 'å':
            return 'ä'
        if char == '÷':
            return 'ö'
        if char == 'ý':
            return 'ü'
        if char == 'à':
            return 'ß'
        return char
    def replace_char(char):
        if (ord(char) >= 33 - shift and ord(char) <= ord('z') - shift):
            return chr(ord(char) + shift)
        if common_replacements:
            return apply_common_char_replacements(char)
        return char

    new_chars = [replace_char(char) for char in doc]
    return ''.join(new_chars)


hint_words = [_letter_shift(word) for word in ('und', 'auch', 'ein', 'eine', 'der', 'die', 'das', 'for', 'and', 'with')]


def _remove_URLs(doc: str) -> str:
    regex = r'[(http(s)?):\/\/(www\.)?a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&\/\/=]*)'
    return re.sub(regex, '', doc)


def _remove_non_latin(doc: str) -> str:
    regex = r'[^\x00-\x7F\x80-\xFF\u0100-\u017F\u0180-\u024F\u1E00-\u1EFF]'
    return re.sub(regex, '', doc)


def _correct_sentence_boundaries(doc: str) -> str:
    """Correct sentence boundaries where no space was inserted.

    Args:
        doc (str): substrate

    Returns:
        str: substrate with spaces after dots and colons where a sentence boundary is assumed
    """
    # No space after period or colon
    regex = r'(:|\.+)([A-Z])'
    doc = re.sub(regex, r'\1 \2', doc)

    # No space between lowercase letter and uppercase letter
    regex = r'([a-z])([A-Z])'
    doc = re.sub(regex, r'\1 \2', doc)
    return doc


def _remove_punctuation(doc: str) -> str:
    """Remove punctuation (no single quotes)

    Args:
        doc (str): substrate

    Returns:
        str: substrate without punctuation
    """    
    regex = r'[.!?;:,-/+*&()\[\]{}"<>=$`#»«©]'  # TODO improve
    return re.sub(regex, '', doc)


def _remove_single_quotes(doc: str) -> str:
    """Remove single quotes from a document

    Args:
        doc (str): substrate

    Returns:
        str: substrate without
    """
    regex = r'[\']'
    return re.sub(regex, '', doc)


def _remove_numbers(doc: str) -> str:
    """Remove numerals and percentage signs from the doc

    Args:
        doc (str): substrate

    Returns:
        str: substrate without numerals and percentage signs
    """
    regex = r'[0-9%]'
    return re.sub(regex, '', doc)


def _to_lower(doc: str) -> str:
    """Lowercase a doc.

    Args:
        doc (str): substrate

    Returns:
        str: substrate in lowercase
    """
    return doc.lower()


def _lemmatize(doc: str, spacy_nlp: str) -> str:
    """Lemmatize words

    Args:
        doc (str): substrate

    Returns:
        str: lemmatized substrate
    """
    lemmatized_doc = [word.lemma_ for word in spacy_nlp(doc)]
    return ' '.join(lemmatized_doc)


def _remove_stop_words(doc: str, language: str) -> str:
    selected_stopwords = None
    if language.lower() in ['en', 'english', 'en-us', 'en-ca', 'en-gb', 'en-ie', 'en-au', 'en-ng', 'en-nz']:
        selected_stopwords = stopwords.words('english')
    elif language.lower() in ['de', 'german']:
        selected_stopwords = stopwords.words('german')
    else:
        raise Exception(f'Unknown language: {language}')

    return ' '.join([word for word in word_tokenize(doc) if word.lower() not in selected_stopwords])


def get_spacy_nlp(language: str) -> spacy.language.Language:
    spacy_nlp = None
    if language.lower() in ['en', 'english', 'en-us', 'en-ca', 'en-gb', 'en-ie', 'en-au', 'en-ng', 'en-nz']:
        spacy_nlp = spacy.load('en_core_web_md')
    elif language.lower() in ['de', 'german']:
        spacy_nlp = spacy.load('de_core_news_md')
    else:
        raise Exception(f'Unknown language: {language}')
    return spacy_nlp


def correct_doc(doc: str) -> str:
    """Correct a document by
    - removing URLs
    - removing non-latin characters
    - correcting sentence boundaries

    Args:
        doc (str): _description_

    Returns:
        str: _description_
    """
    doc = _remove_URLs(doc)
    doc = _remove_non_latin(doc)
    doc = _correct_sentence_boundaries(doc)
    return doc


def slim_doc(doc: str, language: str, spacy_nlp: spacy.language.Language) -> str:
    """Remove punctuation lowercase, and stop words

    Args:
        doc (str): _description_

    Returns:
        str: _description_
    """
    doc = _to_lower(doc)
    doc = _remove_stop_words(doc, language)
    doc = _remove_punctuation(doc)
    doc = _remove_numbers(doc)
    doc = _lemmatize(doc, spacy_nlp)
    doc = _remove_single_quotes(doc)
    return doc


@pytask.mark.persist()
@pytask.mark.depends_on(BUILD_PATH / 'role_models/role_model_data.pkl')
@pytask.mark.produces(BUILD_PATH / 'articles/articles.pkl')
def task_article_preparation(produces: Path):
    """Load source files,
    clean content, add generate a slimmed-down version of the content without stopwords, numbers and punctuation,
    and identify obfuscated content.

    Args:
        produces (Path): destionation file name
    """
    articles = None
    source_files = [ASSET_PATH / 'role_model_articles_de.pkl',  ASSET_PATH / 'role_model_articles_en.pkl']
    for source_file in source_files:
        these_articles = pd.read_pickle(source_file)
        articles = these_articles if articles is None else pd.concat([articles, these_articles])
    articles = articles.rename(columns={'addedAt': 'added_at'})

    processed_articles = None
    # Process in chunks in order to avoid memory leak
    for language in articles.language_ml.unique():
        language_articles = articles[articles['language_ml'] == language]
        spacy_nlp = get_spacy_nlp(language)
        for chunk in np.array_split(language_articles, len(language_articles) // 10000 + 1):
            chunk['content_raw'] = chunk['content']
            chunk['content'] = chunk['content_raw'].parallel_apply(lambda text: correct_doc(text))
            chunk['content_slim'] = chunk['content'].parallel_apply(lambda text: slim_doc(text, language, spacy_nlp))
            chunk['obfuscated'] = chunk['content'].str.contains('|'.join([rf'\b{word}\b' for word in hint_words]))
            processed_articles = chunk if processed_articles is None else pd.concat([processed_articles, chunk])

    processed_articles['content'] = processed_articles['content'].astype(str)
    processed_articles['content_slim'] = processed_articles['content_slim'].astype(str)
    processed_articles['obfuscated'] = processed_articles['obfuscated'].astype(bool)

    # Adding role model information
    role_model_data = pd.read_pickle(BUILD_PATH / 'role_models/role_model_data.pkl')
    processed_articles = processed_articles.join(role_model_data, on='role_model')

    processed_articles.to_pickle(produces)


@pytask.mark.skip()
@pytask.mark.depends_on(BUILD_PATH / 'articles/articles.pkl')
@pytask.mark.produces(BUILD_PATH / 'articles/corpora.pkl')
def task_corpus_extraction(produces):
    articles = pd.read_pickle(BUILD_PATH / 'articles/articles.pkl')
    articles = articles[~(articles['obfuscated'])]

    corpora = {}
    for language in articles['language_ml'].unique():
        corpus = '. '.join(articles[articles['language_ml'] == language].content)
        corpora[language] = corpus
    with open(produces, 'wb') as file:
        pickle.dump(corpora, file)



if __name__ == '__main__':
    raise Exception('Can only be executed by pytask')