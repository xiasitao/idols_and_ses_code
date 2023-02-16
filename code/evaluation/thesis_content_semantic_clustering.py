"""This script produces assets for the unsupervised chapter of the thesis.
"""
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import PercentFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
cm = 1/2.54
thesis_red = "#FF0000"
thesis_gray = "#474847"
thesis_rebecca = "#693EA3"
thesis_fuchsia = "#9A5B91"
thesis_blue = "#50BDE9"
thesis_color_map = matplotlib.colors.LinearSegmentedColormap.from_list("Custom", [thesis_gray, thesis_red], N=1000)
thesis_color_map_special = matplotlib.colors.LinearSegmentedColormap.from_list("Custom", ['white', thesis_red], N=1000)
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=[thesis_red, thesis_blue, thesis_rebecca, thesis_gray, thesis_fuchsia])
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 9

from lib_semantic_clustering import *

articles, human_annotated, articles_distinct, human_annotated_distinct, cluster_columns, cluster_topics, cluster_adjectives = retrieve_data()
article_hypertopic_data = find_hypertopics(articles=articles, hypertopic_table=hypertopic_table, columns=hypertopics_columns)
article_hypertopic_data_distinct = find_hypertopics(articles=articles_distinct, hypertopic_table=hypertopic_table, columns=hypertopics_columns)
article_hypertopic_distribution = find_topic_distributions(articles=article_hypertopic_data, columns=hypertopics_columns)
article_hypertopic_distribution_distinct = find_topic_distributions(articles=article_hypertopic_data_distinct, columns=hypertopics_columns)

cluster_distributions = find_topic_distributions(articles, cluster_columns)
cluster_distributions_distinct = find_topic_distributions(articles_distinct, cluster_columns)


def sigstars(p: float) -> str:
    if p <= 1e-2:
        return '***'
    elif p <= 5e-2:
        return '**'
    elif p <= 10e-2:
        return '*'
    return ''


CLUSTER_DIAGRAM_PATH = BUILD_PATH / 'thesis/50-unsupervised/cluster_diagram.pgf'
def plot_cluster_diagram():
    with open(BUILD_PATH / 'semantic_similarity/sbert_vectors.pkl', 'rb') as file:
        embeddings = pickle.load(file)
    semantic_clusters = pd.read_pickle(BUILD_PATH / 'semantic_similarity/semantic_clusters.pkl')

    coords = embeddings['sbert_2'][0:6000]
    semantic_clusters = semantic_clusters.iloc[0:6000]

    fig = plt.figure(figsize=(7*cm, 7*cm))
    ax = fig.gca()
    ax.scatter(coords[:, 0], coords[:, 1], 3, c=semantic_clusters['cluster_5'], cmap=thesis_color_map_special)
    fig.savefig(CLUSTER_DIAGRAM_PATH, bbox_inches='tight')


SEMANTIC_CLUSTERING_ACCURACY_DIAGRAM_PATH = BUILD_PATH / 'thesis/50-unsupervised/semantic_clustering_accuracy_diagram.pgf'
def plot_semantic_clustering_accuracy_diagram():
    fig = plt.figure(figsize=(11*cm, 5*cm))
    ax = fig.gca()
    plot_accuracy_by_n(article_hypertopic_data=article_hypertopic_data, human_annotated=human_annotated, ax=ax)
    fig.savefig(SEMANTIC_CLUSTERING_ACCURACY_DIAGRAM_PATH, bbox_inches='tight')


SEMANTIC_CLUSTERING_HYPERTOPIC_CONSISTENCY_DIAGRAM_PATH = BUILD_PATH / 'thesis/50-unsupervised/semantic_clustering_hypertopic_consistency_diagram.pgf'
def plot_semantic_clustering_hypertopic_consistency_diagram():
    fig = plt.figure(figsize=(12*cm, 8*cm))
    ax = fig.gca()
    plot_hypertopic_distribution_by_n(hypertopic_distributions=article_hypertopic_distribution, hypertopics=hypertopics, articles_per_SES=find_articles_per_SES(articles), ax=ax)
    fig.savefig(SEMANTIC_CLUSTERING_HYPERTOPIC_CONSISTENCY_DIAGRAM_PATH, bbox_inches='tight')


SEMANTIC_CLUSTERING_CONFUSION_MATRIX_PATH = BUILD_PATH / 'thesis/50-unsupervised/semantic_clustering_confusion_matrix.pgf'
def plot_semantic_clustering_confusion_matrix():
    fig = plt.figure(figsize=(8*cm, 8*cm))
    ax = fig.gca()
    plot_human_annotation_confusion_matrix(article_hypertopic_data, human_annotated, 20, ax, cmap=thesis_color_map_special)
    fig.savefig(SEMANTIC_CLUSTERING_CONFUSION_MATRIX_PATH, bbox_inches='tight')


SEMANTIC_CLUSTERING_HYPERTOPIC_DISTRIBUTION_PATH = BUILD_PATH / 'thesis/50-unsupervised/semantic_clustering_hypertopic_distribution.pgf'
SEMANTIC_CLUSTERING_HYPERTOPIC_DISTRIBUTION_DISTINCT_PATH = BUILD_PATH / 'thesis/50-unsupervised/semantic_clustering_hypertopic_distribution_distinct.pgf'
SEMANTIC_CLUSTERING_HYPERTOPIC_CHI2_TABLE_PATH = BUILD_PATH / 'thesis/50-unsupervised/semantic_clustering_hypertopic_chi2_table.tex'
SEMANTIC_CLUSTERING_TOPIC_HYPERTOPIC_TABLE_PATH = BUILD_PATH / 'thesis/50-unsupervised/semantic_clustering_topic_hypertopic_table.tex'
def plot_semantic_clustering_hypertopic_distribution():
    selection = 'cluster_20'

    fig = plt.figure(figsize=(6.5*cm, 6*cm))
    ax = fig.gca()
    plot_topic_distribution(article_hypertopic_distribution[selection], ax=ax)
    fig.savefig(SEMANTIC_CLUSTERING_HYPERTOPIC_DISTRIBUTION_PATH, bbox_inches='tight')

    fig = plt.figure(figsize=(6.5*cm, 6*cm))
    ax = fig.gca()
    plot_topic_distribution(article_hypertopic_distribution_distinct[selection], ax=ax)
    fig.savefig(SEMANTIC_CLUSTERING_HYPERTOPIC_DISTRIBUTION_DISTINCT_PATH, bbox_inches='tight')

    # Chi2 table
    distribution, distribution_distinct = article_hypertopic_distribution[selection], article_hypertopic_distribution_distinct[selection]
    contingency_chi2, contingency_p = chi2_contingency_test(distribution)
    per_label = chi2_per_label_test(distribution, find_articles_per_SES(articles))
    contingency_chi2_distinct, contingency_p_distinct = chi2_contingency_test(distribution_distinct)
    per_label_distinct = chi2_per_label_test(distribution_distinct, find_articles_per_SES(articles_distinct))
    table_str = r"\begin{tabular}{lcccr@{\hskip0pt}llcccr@{\hskip0pt}l}\toprule & \multicolumn{5}{c}{\textit{mixed-SES}} && \multicolumn{5}{c}{\textit{distinct-SES}} \\ "
    table_str += r"& low & high & $\chi^2$ & \multicolumn{2}{c}{$p$} && low & high & $\chi^2$ & \multicolumn{2}{c}{$p$} \\\toprule" + '\n'
    table_str += rf"all & & & ${contingency_chi2:.2f}$ & $\SI{{{contingency_p:.0e}}}{{}}$ & {sigstars(contingency_p)} && "
    table_str += rf"&& ${contingency_chi2_distinct:.2f}$ & $\SI{{{contingency_p_distinct:.0e}}}{{}}$ & {sigstars(contingency_p_distinct)} \\"  + '\n'
    for hypertopic in per_label.index:
        low, high = distribution.loc[hypertopic]['low_rel'], distribution.loc[hypertopic]['high_rel']
        chi2, p = per_label.loc[hypertopic]['chi2'], per_label.loc[hypertopic]['p']
        low_distinct, high_distinct = distribution_distinct.loc[hypertopic]['low_rel'], distribution_distinct.loc[hypertopic]['high_rel']
        chi2_distinct, p_distinct = per_label_distinct.loc[hypertopic]['chi2'], per_label_distinct.loc[hypertopic]['p']
        table_str += rf"\textit{{{hypertopic}}} & \SI{{{low*100:.0f}}}{{\percent}} & \SI{{{high*100:.0f}}}{{\percent}} & ${chi2:.1f}$ & $\SI{{{p:.0e}}}{{}}$ & {sigstars(p)} && "
        table_str += rf"\SI{{{low_distinct*100:.0f}}}{{\percent}} & \SI{{{high_distinct*100:.0f}}}{{\percent}} & ${chi2_distinct:.1f}$ & $\SI{{{p_distinct:.0e}}}{{}}$ & {sigstars(p_distinct)} \\" + '\n'
    table_str += r"\bottomrule\end{tabular}"
    SEMANTIC_CLUSTERING_HYPERTOPIC_CHI2_TABLE_PATH.write_text(table_str)

    # Topic-hypertopic table
    table_str = r"\begin{tabular}{clc}\toprule topic & characterizing words (nouns and verbs) & hypertopic\\\toprule" + '\n'
    for i, hypertopic in enumerate(hypertopic_table[20]):
        word_list = cluster_topics['cluster_20'][i]
        table_str += rf"{i} & {' '.join(word_list)} & \textit{{{hypertopic}}} \\" + '\n'
    table_str += r"\bottomrule\end{tabular}"
    SEMANTIC_CLUSTERING_TOPIC_HYPERTOPIC_TABLE_PATH.write_text(table_str)



SEMANTIC_CLUSTERING_CLUSTER_DISTRIBUTION_PATH = BUILD_PATH / 'thesis/50-unsupervised/semantic_clustering_cluster_distribution.pgf'
SEMANTIC_CLUSTERING_CLUSTER_DISTRIBUTION_DISTINCT_PATH = BUILD_PATH / 'thesis/50-unsupervised/semantic_clustering_cluster_distribution_distinct.pgf'
SEMANTIC_CLUSTERING_CLUSTER_CHI2_TABLE_PATH = BUILD_PATH / 'thesis/50-unsupervised/semantic_clustering_cluster_chi2_table.tex'
def plot_semantic_clustering_cluster_distribution():
    selection = 'cluster_20'

    fig = plt.figure(figsize=(14*cm, 7*cm))
    ax = fig.gca()
    plot_topic_distribution(cluster_distributions[selection], ax=ax)
    fig.savefig(SEMANTIC_CLUSTERING_CLUSTER_DISTRIBUTION_PATH, bbox_inches='tight')

    fig = plt.figure(figsize=(14*cm, 7*cm))
    ax = fig.gca()
    plot_topic_distribution(cluster_distributions_distinct[selection], ax=ax)
    fig.savefig(SEMANTIC_CLUSTERING_CLUSTER_DISTRIBUTION_DISTINCT_PATH, bbox_inches='tight')

    distribution, distribution_distinct = cluster_distributions[selection], cluster_distributions_distinct[selection]
    contingency_chi2, contingency_p = chi2_contingency_test(distribution)
    per_label = chi2_per_label_test(distribution, find_articles_per_SES(articles))
    contingency_chi2_distinct, contingency_p_distinct = chi2_contingency_test(distribution_distinct)
    per_label_distinct = chi2_per_label_test(distribution_distinct, find_articles_per_SES(articles_distinct))
    table_str = r"\begin{tabular}{lcccr@{\hskip0pt}llcccr@{\hskip0pt}l}\toprule & \multicolumn{5}{c}{\textit{mixed-SES}} && \multicolumn{5}{c}{\textit{distinct-SES}} \\ "
    table_str += r"& low & high & $\chi^2$ & \multicolumn{2}{c}{$p$} && low & high & $\chi^2$ & \multicolumn{2}{c}{$p$} \\\toprule" + '\n'
    table_str += rf"all & & & ${contingency_chi2:.2f}$ & $\SI{{{contingency_p:.0e}}}{{}}$ & {sigstars(contingency_p)} && "
    table_str += rf"&& ${contingency_chi2_distinct:.2f}$ & $\SI{{{contingency_p_distinct:.0e}}}{{}}$ & {sigstars(contingency_p_distinct)} \\"  + '\n'
    for hypertopic in per_label.index:
        low, high = distribution.loc[hypertopic]['low_rel'], distribution.loc[hypertopic]['high_rel']
        chi2, p = per_label.loc[hypertopic]['chi2'], per_label.loc[hypertopic]['p']
        low_distinct, high_distinct = distribution_distinct.loc[hypertopic]['low_rel'], distribution_distinct.loc[hypertopic]['high_rel']
        chi2_distinct, p_distinct = per_label_distinct.loc[hypertopic]['chi2'], per_label_distinct.loc[hypertopic]['p']
        table_str += rf"\textit{{{hypertopic}}} & \SI{{{low*100:.0f}}}{{\percent}} & \SI{{{high*100:.0f}}}{{\percent}} & ${chi2:.1f}$ & $\SI{{{p:.0e}}}{{}}$ & {sigstars(p)} && "
        table_str += rf"\SI{{{low_distinct*100:.0f}}}{{\percent}} & \SI{{{high_distinct*100:.0f}}}{{\percent}} & ${chi2_distinct:.1f}$ & $\SI{{{p_distinct:.0e}}}{{}}$ & {sigstars(p_distinct)} \\" + '\n'
    table_str += r"\bottomrule\end{tabular}"
    SEMANTIC_CLUSTERING_CLUSTER_CHI2_TABLE_PATH.write_text(table_str)


SEMANTIC_CLUSTERING_ADJECTIVES_ADVERBS_TOPIC_TABLE_PATH = BUILD_PATH / 'thesis/50-unsupervised/semantic_clustering_adjectives_adverbs_topic_table.tex'
def produce_adjectives_adverbs_topic_table():
    table_str = r"\begin{tabular}{cl}\toprule cluster & characterizing words (adjectives and adverbs) \\\toprule" + '\n'
    for i in sorted(cluster_adjectives['cluster_20']):
        table_str += rf"{i} & {' '.join(cluster_adjectives['cluster_20'][i])} \\" + '\n'
    table_str += r"\bottomrule\end{tabular}"
    SEMANTIC_CLUSTERING_ADJECTIVES_ADVERBS_TOPIC_TABLE_PATH.write_text(table_str)


if __name__ == '__main__':
    plot_cluster_diagram()
    plot_semantic_clustering_accuracy_diagram()
    plot_semantic_clustering_hypertopic_consistency_diagram()
    plot_semantic_clustering_confusion_matrix()
    plot_semantic_clustering_hypertopic_distribution()
    plot_semantic_clustering_cluster_distribution()
    produce_adjectives_adverbs_topic_table()