import pytask
import pandas as pd
import pickle

import sentence_transformers as st
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

from gensim.corpora import Dictionary
from gensim.models import LdaModel
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

from pathlib import Path
SOURCE_PATH = Path(__file__).parent.resolve()
ASSET_PATH = SOURCE_PATH.joinpath('..', '..', 'assets').resolve()
BUILD_PATH = SOURCE_PATH.joinpath("..", "..", "build").resolve()


def filter_tokens(doc: list, pos_to_keep=['verbs', 'nouns']) -> list:
    """Filter out tokens from a token list. Filtering out tokes with length 0 or 1, and filtering out
    tokens whose part of speech is not among the selected POS.

    Args:
        doc (list): _description_
        pos_to_keep (list, optional): _description_. Defaults to ['verbs', 'nouns'].

    Returns:
        list: _description_
    """    
    nltk_tokens = word_tokenize(' '.join(doc))

    pos_labels = []
    if 'verbs' in pos_to_keep:
        pos_labels += ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    if 'nouns' in pos_to_keep:
        pos_labels += ['FW', 'NN', 'NNS', 'NNP', 'NNPS',]
    if 'adjectives' in pos_to_keep:
        pos_labels += ['JJ', 'JJR', 'JJS']
    if 'adverbs' in pos_to_keep:
        pos_labels += ['RB', 'RBR', 'RBS']

    return [token[0] for token in pos_tag(nltk_tokens)
        if len(token[0]) > 1
        and token[0] != 've'
        and token[1] in pos_labels  # https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
    ]


def find_topic(articles_slim: "list[str]", pos: "list[str]") -> "list[str]":
    """Find the major topic in a list of slimmed articles.

    Args:
        articles_slim (list[str]): Slimmed down article texts.
        pos (list[str]): List of parts of speech to keep.

    Returns:
        list[str]: List of words characterizing the topic
    """    
    articles_tokenized = [article.split(' ') for article in articles_slim]
    dictionary = Dictionary(articles_tokenized)
    dictionary.filter_extremes(no_below=5, no_above=0.3)
    dictionary[0]
    corpus = [dictionary.doc2bow(filter_tokens(article, pos)) for article in articles_tokenized]

    iterations = 500
    passes = 4
    model = LdaModel(
        corpus=corpus, id2word=dictionary.id2token, random_state=42,
        iterations=iterations, passes=passes,
        num_topics=1,
    )
    topics = [[topic_entry[1] for topic_entry in topic[0][0:10]] for topic in model.top_topics(corpus)]
    topic = topics[0]
    return topic


SBERT_VECTORS_PATH = BUILD_PATH / 'semantic_similarity/sbert_vectors.pkl'
@pytask.mark.persist()
@pytask.mark.skip()  # remove
@pytask.mark.depends_on(BUILD_PATH / 'articles/articles_balanced_50.pkl')
@pytask.mark.produces(SBERT_VECTORS_PATH)
def task_semantic_embedding(produces: Path):
    """Finds the Sentence-BERT embeddings of the balanced 50 articles data set.
    Dimensionality-reduces the embeddings to 50, 10, 3, and 2.

    Args:
        produces (Path): destination file
    """
    articles = pd.read_pickle(BUILD_PATH / 'articles/articles_balanced_50.pkl')
    articles_en = articles[articles['language_ml']=='en']
    articles_en = articles_en[['article_id', 'content']].drop_duplicates()

    sentence_bert_encoder = st.SentenceTransformer('all-MiniLM-L6-v2')
    print('Finding SBERT embeddings...')
    embeddings_sbert = sentence_bert_encoder.encode(articles_en['content'].to_list())
    print('Reducing to 50...')
    embeddings_50d = PCA(n_components=50, random_state=42).fit_transform(embeddings_sbert)
    print('Reducing to 10...')
    embeddings_10d = PCA(n_components=10, random_state=42).fit_transform(embeddings_sbert)
    print('Reducing to 3...')
    embeddings_3d = TSNE(n_components=3, random_state=42).fit_transform(embeddings_50d)
    print('Reducing to 2...')
    embeddings_2d = TSNE(n_components=2, random_state=42).fit_transform(embeddings_50d)

    all_embeddings = {}
    all_embeddings['index'] = articles_en['article_id'].to_list()
    all_embeddings['sbert'] = embeddings_sbert
    all_embeddings['sbert_50'] = embeddings_50d
    all_embeddings['sbert_10'] = embeddings_10d
    all_embeddings['sbert_3'] = embeddings_3d
    all_embeddings['sbert_2'] = embeddings_2d

    with open(produces, 'wb') as file:
        pickle.dump(all_embeddings, file)


SEMANTIC_CLUSTERS_PATH = BUILD_PATH / 'semantic_similarity/semantic_clusters.pkl'
@pytask.mark.produces(SEMANTIC_CLUSTERS_PATH)
@pytask.mark.depends_on(BUILD_PATH / 'semantic_similarity/sbert_vectors.pkl')
def task_semantic_clustering(produces: Path, all_n_clusters=[4,5,6,8,10,15,20,25,30,35,40,45,50]):
    """Cluster the semantic embeddings with KMeans.

    Args:
        produces (Path): Destination path
        all_n_clusters (list, optional): List of all n_clusters to cluster for. Defaults to [2,3,4,5,6,7,8,9,10].
    """    
    with open(BUILD_PATH / 'semantic_similarity/sbert_vectors.pkl', 'rb') as file:
        embeddings = pickle.load(file)
    article_ids = embeddings['index']
    vectors_for_clustering = embeddings['sbert_3']
    
    cluster_data = pd.DataFrame(data=None, index=article_ids)
    for n_clusters in all_n_clusters:
        print(f'Clustering with n_clusters={n_clusters}...')
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit_transform(vectors_for_clustering)
        kmeans_clusters = kmeans.predict(vectors_for_clustering)
        cluster_data[f'cluster_{n_clusters}'] = kmeans_clusters
    cluster_data.to_pickle(produces)


SEMANTIC_TOPICS_PATH = BUILD_PATH / 'semantic_similarity/semantic_topics.pkl'
SEMANTIC_TOPICS_PATH_ADJECTIVES = BUILD_PATH / 'semantic_similarity/semantic_topics_adjectives.pkl'
@pytask.mark.produces(SEMANTIC_CLUSTERS_PATH)
@pytask.mark.depends_on(BUILD_PATH / 'semantic_similarity/semantic_clusters.pkl')
@pytask.mark.depends_on(BUILD_PATH / 'articles/articles_balanced_50.pkl')
def task_semantic_cluster_topic_modelling(produces: Path, pos=['nouns', 'verbs']):
    """Find topic word lists for every cluster.

    Args:
        produces (Path): Destination path.
        pos (list(str)): List of parts of speech to
    """    
    cluster_data = pd.read_pickle(BUILD_PATH / 'semantic_similarity/semantic_clusters.pkl')
    articles_raw = pd.read_pickle(BUILD_PATH / 'articles/articles_balanced_50.pkl')
    articles_raw = articles_raw[articles_raw['language_ml']=='en']
    articles = articles_raw.join(cluster_data, on='article_id', how='inner')

    cluster_topics = {}
    for n_clusters_column in cluster_data.columns:
        print(f'Finding topics for {n_clusters_column}, cluster ', end='', flush=True)
        cluster_labels = cluster_data[n_clusters_column].unique()
        this_n_topics = {}
        for cluster_label in cluster_labels:
            print(f'{cluster_label} ', end='', flush=True)
            cluster_article_ids = cluster_data[cluster_data[n_clusters_column]==cluster_label].index.unique()
            cluster_articles = articles[articles['article_id'].isin(cluster_article_ids)]
            topic = find_topic(cluster_articles['content_slim'].to_list(), pos=pos)
            this_n_topics[cluster_label] = topic
        print()
        cluster_topics[n_clusters_column] = this_n_topics

    with open(produces, 'wb') as file:
        pickle.dump(cluster_topics, file)


if __name__ == '__main__':
    # task_semantic_embedding(SBERT_VECTORS_PATH)
    # task_semantic_clustering(SEMANTIC_CLUSTERS_PATH)
    # task_semantic_cluster_topic_modelling(SEMANTIC_TOPICS_PATH)
    task_semantic_cluster_topic_modelling(SEMANTIC_TOPICS_PATH_ADJECTIVES, pos=['adjectives', 'adverbs'])
    