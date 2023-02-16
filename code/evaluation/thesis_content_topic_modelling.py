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
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=[thesis_red, thesis_blue, thesis_rebecca, thesis_gray, thesis_fuchsia])
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 9

from lib_topic_modelling import *
articles, human_annotated, articles_distinct, human_annotated_distinct, topic_columns, topic_words = retrieve_data()
article_hypertopic_data = find_hypertopics(articles, hypertopic_table, hypertopics_columns)
article_hypertopic_distributions = find_topic_distributions(article_hypertopic_data, hypertopics_columns)


LDA_ACCURACY_DIAGRAM_PATH = BUILD_PATH / 'thesis/50-unsupervised/lda_accuracy_diagram.pgf'
def plot_topic_modelling_accuracy_diagram():
    fig = plt.figure(figsize=(11*cm, 5*cm))
    ax = fig.gca()
    plot_accuracy_by_n(article_hypertopic_data, human_annotated, ax=ax)
    fig.savefig(LDA_ACCURACY_DIAGRAM_PATH, bbox_inches='tight')


LDA_HYPERTOPIC_CONSISTENCY_DIAGRAM_PATH = BUILD_PATH / 'thesis/50-unsupervised/lda_hypertopic_consistency_diagram.pgf'
def plot_hypertopic_consistency_plot():
    fig = plt.figure(figsize=(12*cm, 8*cm))
    ax = fig.gca()
    plot_hypertopic_distribution_by_n(article_hypertopic_distributions, hypertopics, find_articles_per_SES(articles), ax=ax)
    fig.savefig(LDA_HYPERTOPIC_CONSISTENCY_DIAGRAM_PATH, bbox_inches='tight')


if __name__ == '__main__':
    plot_topic_modelling_accuracy_diagram()
    plot_hypertopic_consistency_plot()