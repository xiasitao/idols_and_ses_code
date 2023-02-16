import pytask
import pickle
import pandas as pd
from pandarallel import pandarallel
pandarallel.initialize()
import numpy as np

from gensim.corpora import Dictionary
from gensim.models import LdaModel
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

from pathlib import Path
SOURCE_PATH = Path(__file__).parent.resolve()
ASSET_PATH = SOURCE_PATH.joinpath('..', '..', 'assets').resolve()
BUILD_PATH = SOURCE_PATH.joinpath("..", "..", "build").resolve()


def filter_tokens(doc: list) -> list:
    nltk_tokens = word_tokenize(' '.join(doc))
    return [token[0] for token in pos_tag(nltk_tokens)
        if len(token[0]) > 1
        and token[0] != 've'
        and token[1] in ('FW', 'NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ')  # https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
    ]


def train_lda_model(n_topics: int, dictionary: Dictionary, corpus: list) -> tuple:
    """Train an LDA model on the corpus with a predefined number of topics.

    Args:
        n_topics (int): Number of topcis to generate.

    Returns:
        tuple: model, topics (10 most significant words for each topic)
    """    
    iterations = 500
    passes = 4
    model = LdaModel(
        corpus=corpus, id2word=dictionary.id2token, random_state=42,
        iterations=iterations, passes=passes,
        num_topics=n_topics,
    )
    topics = [[topic_entry[1] for topic_entry in topic[0][0:10]] for topic in model.top_topics(corpus)]
    return model, topics


def find_topic_entropy_and_probability(model: LdaModel, dictionary: Dictionary, doc: list) -> tuple:
    """Find the most probable topic, the topic distribution entropy, and the topic's probability for a document.

    Args:
        model (LdaModel): Topic model
        doc (list): Document as list of tokens

    Returns:
        tuple: topic, entropy, probability
    """    
    topic_probabilities = np.array(model[dictionary.doc2bow(filter_tokens(doc))])
    topics = topic_probabilities[:, 0]
    probabilities = topic_probabilities[:, 1]
    topic = topics[probabilities.argmax()]
    entropy = -probabilities.dot(np.log(probabilities))
    probability = probabilities.max()
    return topic, entropy, probability


TOPIC_MODELLING_BUILD_PATH = BUILD_PATH / 'topic_modelling/topic_modelling.pkl'
@pytask.mark.skip()
@pytask.mark.depends_on(BUILD_PATH / 'articles/articles_balanced_50.pkl')
@pytask.mark.produces(TOPIC_MODELLING_BUILD_PATH)
def task_topic_modelling(produces: Path, all_n_topics=[2,3,4,5,6,7,8,9,10], articles: pd.DataFrame=None):
    """Perform topic modelling on the 50-articles-per-role-model balanced article data set.

    Args:
        produces (Path): Destination file path
        all_n_topics (list[int]): List of all n_topics to model for.
        articles (pd.DataFrame): Preset set of articles to perform topic modelling on.
    """    
    if articles is None:
        articles = pd.read_pickle(BUILD_PATH / 'articles/articles_balanced_50.pkl')
    articles_en = articles[articles.language_ml == 'en']
    articles_en = articles_en[['article_id', 'content_slim']]
    articles_en['content_tokenized'] = articles_en['content_slim'].str.split(' ')
    
    # Building corpus with balanced articles
    dictionary = Dictionary(articles_en['content_tokenized'])
    dictionary.filter_extremes(no_below=5, no_above=0.3)
    dictionary[0]
    corpus = [dictionary.doc2bow(filter_tokens(article)) for article in articles_en['content_tokenized']]

    # Predicting with unique articles
    unique_articles = articles_en[['article_id', 'content_slim']].drop_duplicates().set_index('article_id', drop=True)
    unique_articles['content_tokenized'] = unique_articles['content_slim'].str.split(' ')
    article_topics = pd.DataFrame(data=None, columns=[wildcard.format(n) for n in all_n_topics for wildcard in ('topic_{}', 'topic_{}_entropy', 'topic_{}_p')], index=unique_articles.index)
    topic_words = {}
    for n_topics in all_n_topics:
        print(f'Modelling for {n_topics} topics...')
        model, these_topic_words = train_lda_model(n_topics, corpus=corpus, dictionary=dictionary)
        topic_words[n_topics] = these_topic_words
        article_topics[[f'topic_{n_topics}', f'topic_{n_topics}_entropy', f'topic_{n_topics}_p']] = unique_articles['content_tokenized'].parallel_apply(lambda doc: pd.Series(find_topic_entropy_and_probability(model, dictionary, doc)))
    
    # Save topic-classified articles
    with open(produces, 'wb') as file:
        pickle.dump((topic_words, article_topics,), file)


def task_ses_separated_topic_modelling(all_n_topics=[5, 10, 15, 20]):
    articles_raw = pd.read_pickle(BUILD_PATH / 'articles/articles_balanced_50.pkl')
    articles_raw = articles_raw[articles_raw.language_ml == 'en']
    ses = pd.read_pickle(BUILD_PATH / 'role_models/ses_scores.pkl')
    articles = articles_raw.join(ses, how='inner', on='role_model')
    ses_distinct = pd.read_pickle(BUILD_PATH / 'role_models/ses_scores_distinct.pkl')
    articles_distinct = articles_raw.join(ses_distinct, how='inner', on='role_model')
    
    low_ses_articles = articles[articles['low_ses']]
    print('Modelling with low-SES articles...')
    task_topic_modelling(BUILD_PATH / 'topic_modelling/topic_modelling_low_ses.pkl', all_n_topics=all_n_topics, articles=low_ses_articles)
    high_ses_articles = articles[articles['high_ses']]
    print('Modelling with high-SES articles...')
    task_topic_modelling(BUILD_PATH / 'topic_modelling/topic_modelling_high_ses.pkl', all_n_topics=all_n_topics, articles=high_ses_articles)

    low_ses_articles_distinct = articles_distinct[articles_distinct['low_ses']]
    print('Modelling with distinct low-SES articles...')
    task_topic_modelling(BUILD_PATH / 'topic_modelling/topic_modelling_low_ses_distinct.pkl', all_n_topics=all_n_topics, articles=low_ses_articles_distinct)
    high_ses_articles_distinct = articles_distinct[articles_distinct['high_ses']]
    print('Modelling with distinct high-SES articles...')
    task_topic_modelling(BUILD_PATH / 'topic_modelling/topic_modelling_high_ses_distinct.pkl', all_n_topics=all_n_topics, articles=high_ses_articles_distinct)


if __name__ == '__main__':
    task_topic_modelling(
        TOPIC_MODELLING_BUILD_PATH,
        all_n_topics=[2,3,4,5,6,7,8,9,10,11,12,13,14,15,20,25,30,35,40,45,50,55,60,65,70]
    )

    # task_ses_separated_topic_modelling()