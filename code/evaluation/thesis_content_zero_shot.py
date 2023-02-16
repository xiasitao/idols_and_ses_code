"""This script produces assets for the supervised chapter of the thesis.
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


def sigstars(p: float) -> str:
    if p <= 1e-2:
        return '***'
    elif p <= 5e-2:
        return '**'
    elif p <= 10e-2:
        return '*'
    return ''


from lib_zero_shot import *
articles, human_annotated, articles_distinct, human_annotated_distinct, category_columns = retrieve_data()
distributions = find_category_distributions(articles=articles, category_columns=category_columns)
distributions_distinct = find_category_distributions(articles=articles_distinct, category_columns=category_columns)

ZERO_SHOT_RESULT_TABLE_PATH = BUILD_PATH / 'thesis/70-supervised/zero_shot_result_table.tex'
def produce_zero_shot_result_table():
    table_str = r"\begin{tabular}{lccccr@{\hskip0pt}llccccr@{\hskip0pt}l}\toprule & \multicolumn{6}{c}{\textit{mixed-SES}} && \multicolumn{6}{c}{\textit{distinct-SES}} \\ "
    table_str += r"& low & high & $\Braket{\widetilde{H}}$ & $\chi^2$ & \multicolumn{2}{c}{$p$} && low & high & $\Braket{\widetilde{H}}$ & $\chi^2$ & \multicolumn{2}{c}{$p$} \\\toprule" + '\n'

    for category in category_columns:
        category_to_display = category.replace('topic_l', 'topic (soc.)').replace('sentiment_n', 'sentiment (n)').replace('_', ' ')
        distribution, distribution_distinct = distributions[category], distributions_distinct[category]
        contingency_chi2, contingency_p = chi2_contingency_test(distribution)
        contingency_entropy = calculate_average_entropy(articles, category, norm=True)
        per_label_chi2 = chi2_per_label_test(distribution, find_articles_per_SES(articles, category))
        contingency_chi2_distinct, contingency_p_distinct = chi2_contingency_test(distribution_distinct)
        contingency_entropy_distinct = calculate_average_entropy(articles_distinct, category, norm=True)
        per_label_chi2_distinct = chi2_per_label_test(distribution_distinct, find_articles_per_SES(articles_distinct, category))
        table_str += rf"\textbf{{{category_to_display}}} & & & ${contingency_entropy:.2f}$ & ${contingency_chi2:.2f}$ & $\SI{{{contingency_p:.0e}}}{{}}$ & {sigstars(contingency_p)} && "
        table_str += rf"& & ${contingency_entropy_distinct:.2f}$ & ${contingency_chi2_distinct:.2f}$ & $\SI{{{contingency_p_distinct:.0e}}}{{}}$ & {sigstars(contingency_p_distinct)} \\"  + '\n'
        for label in per_label_chi2.index:
            low, high, entropy = distribution.loc[label]['low_rel'], distribution.loc[label]['high_rel'], distribution.loc[label]['entropy_norm']
            chi2, p = per_label_chi2.loc[label]['chi2'], per_label_chi2.loc[label]['p']
            low_distinct, high_distinct, entropy_distinct = distribution_distinct.loc[label]['low_rel'], distribution_distinct.loc[label]['high_rel'], distribution_distinct.loc[label]['entropy_norm']
            chi2_distinct, p_distinct = per_label_chi2_distinct.loc[label]['chi2'], per_label_chi2_distinct.loc[label]['p']
            table_str += rf"\textit{{{label}}} & \SI{{{low*100:.0f}}}{{\percent}} & \SI{{{high*100:.0f}}}{{\percent}} & ${entropy:.2f}$ & ${chi2:.1f}$ & $\SI{{{p:.0e}}}{{}}$ & {sigstars(p)} && "
            table_str += rf"\SI{{{low_distinct*100:.0f}}}{{\percent}} & \SI{{{high_distinct*100:.0f}}}{{\percent}} & ${entropy_distinct:.2f}$ & ${chi2_distinct:.1f}$ & $\SI{{{p_distinct:.0e}}}{{}}$ & {sigstars(p_distinct)} \\" + '\n'
        if category != category_columns[-1]:
            table_str += r'\midrule'
    table_str += r"\bottomrule\end{tabular}"
    ZERO_SHOT_RESULT_TABLE_PATH.write_text(table_str)


ZERO_SHOT_CONFUSION_MATRIX_TOPIC_PATH = BUILD_PATH / 'thesis/70-supervised/zero_shot_confusion_matrix_topic.pgf'
ZERO_SHOT_CONFUSION_MATRIX_TOPIC_L_PATH = BUILD_PATH / 'thesis/70-supervised/zero_shot_confusion_matrix_topic_l.pgf'
def plot_zero_shit_confusion_matrices():
    width = 6.5*cm
    fig = plt.figure(figsize=(width, width))
    ax = fig.gca()
    plot_human_annotation_confusion_matrix(articles, human_annotated, 'topic', ax=ax, cmap=thesis_color_map_special)
    fig.savefig(ZERO_SHOT_CONFUSION_MATRIX_TOPIC_PATH, bbox_inches='tight')

    fig = plt.figure(figsize=(width, width))
    ax = fig.gca()
    plot_human_annotation_confusion_matrix(articles, human_annotated, 'topic_l', ax=ax, cmap=thesis_color_map_special)
    fig.savefig(ZERO_SHOT_CONFUSION_MATRIX_TOPIC_L_PATH, bbox_inches='tight')


ZERO_SHOT_DISTRIBUTION_PLOT_BASEPATH = BUILD_PATH / 'thesis/70-supervised/'
ZERO_SHOT_DISTRIBUTION_PLOT_BASEPATH_WILDCARD = 'zero_shot_distribution_{}.pgf'
def plot_category_distributions():
    for category in category_columns:
        width = 6.5*cm
        height = 0.7*width
        fig = plt.figure(figsize=(width, height))
        ax = fig.gca()
        plot_category_distribution(distributions, category, show_title=False, ax=ax)
        fig.savefig(
            ZERO_SHOT_DISTRIBUTION_PLOT_BASEPATH / ZERO_SHOT_DISTRIBUTION_PLOT_BASEPATH_WILDCARD.format(category), 
            bbox_inches='tight'
        )
        plt.close()

        fig = plt.figure(figsize=(width, height))
        ax = fig.gca()
        plot_category_distribution(distributions_distinct, category, show_title=False, ax=ax)
        fig.savefig(
            ZERO_SHOT_DISTRIBUTION_PLOT_BASEPATH / ZERO_SHOT_DISTRIBUTION_PLOT_BASEPATH_WILDCARD.format(f'{category}_distinct'), 
            bbox_inches='tight'
        )
        plt.close()


if __name__ == '__main__':
    produce_zero_shot_result_table()
    plot_zero_shit_confusion_matrices()
    plot_category_distributions()