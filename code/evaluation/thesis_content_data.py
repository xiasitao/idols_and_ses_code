"""This script produces assets for the data chapter of the thesis.
"""
import pandas as pd
import numpy as np


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

from pathlib import Path
SOURCE_PATH = Path(__file__).parent.resolve()
ASSET_PATH = SOURCE_PATH.joinpath('..', '..', 'assets').resolve()
BUILD_PATH = SOURCE_PATH.joinpath("..", "..", "build").resolve()


ROLE_MODEL_LIST_PATH = BUILD_PATH / 'thesis/30-data/role_model_overview.tex'
def produce_role_model_list(produces: Path):
    """Produce the list of role models.

    Args:
        produces (Path): destination path
    """    
    role_model_data = pd.read_pickle(BUILD_PATH / 'role_models/role_model_data.pkl')
    scores = pd.read_pickle(BUILD_PATH / 'role_models/ses_scores.pkl')
    articles_balanced_50 = pd.read_pickle(BUILD_PATH / 'articles/articles_balanced_50.pkl')
    articles_balanced_50 = articles_balanced_50[articles_balanced_50['language_ml']=='en']
    scores = scores[scores.index.isin(articles_balanced_50['role_model'])]
    role_model_data = role_model_data.join(scores, how='inner')
    role_model_data = role_model_data.sort_index(key=lambda name: name.str.lower())

    latex = r"\begin{longtable}{lccccccc}\toprule role model & sex & main prof. & y.o.b. & \#low & \#high & $\textit{distinct}$ \\\toprule"
    for i in range(len(role_model_data)):
        role_model = role_model_data.iloc[i].fillna("")
        line = f'{role_model.name} & {"m" if role_model["sex"]==0.0 else "f"} & {role_model["profession"]} & {role_model["birth_year"]} & {role_model["low_ses_count"]} & {role_model["high_ses_count"]} & {"XYZXYZxmark" if role_model["low_ses"] and role_model["high_ses"] else r"XYZXYZcmark"}'
        line = line.replace('XYZXYZ', '\\')
        latex += f'{line} \\\\\n'
    latex += r"\bottomrule\caption{Overview of the role models. Abbreviations: \textit{nat.}: nationality, \textit{main prof.}: main profession, \textit{y.o.b.}: year of birth, \textit{\#low/\#high}: number of mentions by low-SES/high-SES study participants, \textit{distinct}: whether they are in the set of role models with distinct SES association.}\label{tab:role_model_overview}\end{longtable}"
    produces.write_text(latex)


ARTICLES_PER_ROLE_MODEL_PATH = BUILD_PATH / 'thesis/30-data/role_model_article_distribution.pgf'
def plot_articles_per_role_model_distribution(produces: Path):
    articles = pd.read_pickle(BUILD_PATH / 'articles/articles.pkl')
    articles = articles[articles['language_ml']=='en']
    balanced = pd.read_pickle(BUILD_PATH / 'articles/articles_balanced_50.pkl')
    balanced = balanced[balanced['language_ml']=='en']

    role_models, counts = np.unique(articles['role_model'], return_counts=True)
    role_models_balanced, counts_balanced = np.unique(balanced['role_model'], return_counts=True)
    sorted_counts, sorted_counts_balanced = np.sort(counts), np.sort(counts_balanced)
    counts_cumulative, counts_cumulative_balanced = sorted_counts.cumsum(), sorted_counts_balanced.cumsum()

    plt.figure(figsize=(8*cm, 5*cm))
    plt.xlabel('role model percentile')
    plt.ylabel('cumulative percentage of articles')
    plt.plot(np.arange(counts_cumulative.size)/(counts_cumulative.size)*100, counts_cumulative/np.max(counts_cumulative)*100, label='English articles')
    plt.plot(np.arange(counts_cumulative_balanced.size)/(counts_cumulative_balanced.size)*100, counts_cumulative_balanced/np.max(counts_cumulative_balanced)*100, label='balanced')
    plt.gca().xaxis.set_major_formatter(PercentFormatter())
    plt.gca().yaxis.set_major_formatter(PercentFormatter())
    plt.margins(x=0, y=0)
    plt.grid()
    plt.legend()
    plt.savefig(produces, bbox_inches='tight')


if __name__ == '__main__':
    produce_role_model_list(ROLE_MODEL_LIST_PATH)
    plot_articles_per_role_model_distribution(ARTICLES_PER_ROLE_MODEL_PATH)