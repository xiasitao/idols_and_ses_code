import pytask
import pickle
import pandas as pd
import numpy as np

from pathlib import Path
SOURCE_PATH = Path(__file__).parent.resolve()
ASSET_PATH = SOURCE_PATH.joinpath('..', '..', 'assets').resolve()
BUILD_PATH = SOURCE_PATH.joinpath("..", "..", "build").resolve()


def clean_ses_data(data: pd.DataFrame, role_model_data: pd.DataFrame) -> pd.DataFrame:
    """Clean up the SES data table.

    Args:
        data (pd.DataFrame): Substrate
        role_model_data (pd.DataFrame): Reference cleaned role model data

    Returns:
        pd.DataFrame: Cleaned copy of substrate
    """    
    data = data.copy()
    data = data.rename({
        'Unnamed: 0': 'id',
        'Low_SES': 'low_ses',
        'High_SES': 'high_ses',
        'Role_model_1': 'role_model_1',
        'Role_model_2': 'role_model_2',
        'Role_model_3': 'role_model_3',
        'Role_model_4': 'role_model_4',
        'Role_model_5': 'role_model_5',
    }, axis=1)
    data = data[~(data['id'].isna())]
    data = data.astype({'id': pd.Int64Dtype()})
    data = data.set_index('id')
    data = data[data['role_model_1'].isin(role_model_data.index)]
    data['low_ses'] = data['low_ses'] == 1.0
    data['high_ses'] = data['high_ses'] == 1.0
    data['ses'] = data['high_ses'].apply(lambda high_ses: 1.0 if high_ses else 0.0)
    data['role_model_count'] = 5 - data[[f'role_model_{n}' for n in range(1, 6)]].isna().sum(axis=1)
    return data


def produce_role_model_mention_table(ses_data: pd.DataFrame) -> pd.DataFrame:
    """Creates a table of all mentions of role models in the SES data.
    Each time a role model is mentioned in one of the 1..5th rank corresponds to one row.
    This rank and a role model significance, being the inverse amount of role models the same study participant mentioned.

    Args:
        ses_data (pd.DataFrame): Substrate (cleaned) ses data

    Returns:
        pd.DataFrame: A dataframe with a row for every time a role model is mentioned by a study participant.
    """    
    mention_data = pd.DataFrame(columns=['id', 'role_model', 'ses', 'rank'])
    for rank, column in [(n, f'role_model_{n}') for n in range(1, 6)]:
        rank_mention_data = ses_data[~(ses_data[column].isna())]
        rank_mention_data = rank_mention_data.reset_index()
        rank_mention_data = rank_mention_data.rename({column: 'role_model'}, axis=1)
        rank_mention_data['rank'] = rank
        rank_mention_data['significance'] = 1 / rank_mention_data['role_model_count']
        rank_mention_data = rank_mention_data[['id', 'role_model', 'ses', 'rank', 'significance']]
        mention_data = pd.concat([mention_data, rank_mention_data]).reset_index(drop=True)
    return mention_data


def produce_role_model_scores(mention_data: pd.DataFrame, rank_weights = [1, 1/2, 1/3, 1/4, 1/5]) -> pd.DataFrame:
    """Aggregates the mention data to the role model level,
    calculating counts, rank-weighted counts, and significance-weighted counts,
    as well as zero-centered average SES scores, average rank-weighted SES scores,
    average significance-weighted SES scores, and whether the role model had any
    low- and any high-SES mentions.

    Args:
        mention_data (pd.DataFrame): List of role model mentions.
        rank_weights (list, optional): How to weigh a mention in every rank from 1 to 5. Defaults to [1, 1/2, 1/3, 1/4, 1/5].

    Returns:
        pd.DataFrame: _description_
    """    
    mention_data = mention_data.copy()
    mention_data['zero_centered_ses'] = mention_data['ses'].map({1.0: 1.0, 0.0: -1.0})
    mention_data['weighted_rank'] = mention_data['rank'].apply(lambda rank: rank_weights[rank-1])
    mention_data['rank_weighted_ses'] = mention_data['zero_centered_ses'] * mention_data['weighted_rank']
    mention_data['significance_weighted_ses'] = mention_data['zero_centered_ses'] * mention_data['significance']

    role_models = mention_data['role_model'].unique().sort()
    score_data = pd.DataFrame(index=role_models)
    score_data['count'] = mention_data.groupby('role_model').count()['id']
    score_data['rank_weighted_count'] = mention_data.groupby('role_model')['weighted_rank'].sum()
    score_data['significance_weighted_count'] = mention_data.groupby('role_model')['significance'].sum()

    score_data['prevalent_ses'] = mention_data.groupby('role_model')['zero_centered_ses'].agg(pd.Series.mode).apply(lambda val: val if type(val) != np.ndarray else np.mean(val)).astype(float)
    score_data['average_ses'] = mention_data.groupby('role_model')['zero_centered_ses'].mean()
    score_data['rank_weighted_ses'] = mention_data.groupby('role_model')['rank_weighted_ses'].mean()
    score_data['significance_weighted_ses'] = mention_data.groupby('role_model')['significance_weighted_ses'].mean()

    score_data['low_ses_count'] = mention_data.groupby('role_model')['ses'].apply(lambda role_model_sess: role_model_sess.apply(lambda ses: 0 if ses==1.0 else 1).sum())
    score_data['low_ses'] = score_data['low_ses_count'] > 0
    score_data['high_ses_count'] = mention_data.groupby('role_model')['ses'].apply(lambda role_model_sess: role_model_sess.apply(lambda ses: 0 if ses==0.0 else 1).sum())
    score_data['high_ses'] = score_data['high_ses_count'] > 0

    return score_data


def filter_role_models(role_model_scores: pd.DataFrame, minimum_count: int, require_distinct_SES: bool) -> pd.DataFrame:
    """Filter role models by a minimum amount of mentions
    and by requiring unique SES of the role model mentioning questionnaire participants.

    Args:
        role_model_scores (pd.DataFrame): substrate
        minimum_count (int): Minimum count of mentions
        require_distinct_SES (bool): Whether to enforce distinct-SES mentions only

    Returns:
        pd.DataFrame: _description_
    """
    role_model_scores = role_model_scores.copy()
    if minimum_count is not None:
        role_model_scores = role_model_scores[role_model_scores['count'] >= minimum_count]
    if require_distinct_SES:
        role_model_scores = role_model_scores[role_model_scores['average_ses'].isin([1.0, -1.0])]
    return role_model_scores


def equilibrate_role_models(role_model_scores: pd.DataFrame, group_column='prevalent_ses') -> pd.DataFrame:
    """Equilibrate the distribution of role models by the prevalent_ses.
    The smallest amount of role models determines to how many the other ses' should be downsampled to.

    Args:
        role_model_scores (pd.DataFrame): substrate
        group_column (str): Column name to equilibrate by. Default: 'prevalent_ses'

    Returns:
        pd.DataFrame: substrate with equal amounts of role models for each prevalent_ses group
    """
    role_model_scores = role_model_scores.copy()
    minimum_count = role_model_scores.groupby(group_column).count()['count'].min()
    role_model_scores = role_model_scores.sample(frac=1.0, random_state=42)
    role_model_scores['_group_index'] = role_model_scores.groupby(group_column).cumcount()
    role_model_scores = role_model_scores[role_model_scores['_group_index'] < minimum_count]
    role_model_scores = role_model_scores.drop('_group_index', axis=1)
    return role_model_scores


@pytask.mark.produces(BUILD_PATH / 'role_models/role_model_data.pkl')  
def task_role_model_data(produces: Path):
    """Bring role model information into a good shape for joining with article data.

    Args:
        produces (Path): Output path
    """    
    role_model_data = pd.read_excel(ASSET_PATH / 'role_model_data.xlsx')
    role_model_data = role_model_data[['Star', 'Sex', 'Birth_year', 'Nationality', 'Profession_1']]
    role_model_data = role_model_data.rename({
        'Star': 'role_model',
        'Sex': 'sex',
        'Birth_year': 'birth_year',
        'Nationality': 'nationality',
        'Profession_1': 'profession',
    }, axis=1)

    role_model_data = role_model_data[~(role_model_data['role_model'].isna())]

    role_model_data['role_model'] = role_model_data['role_model'].astype(str).str.strip()
    role_model_data['sex'] = role_model_data['sex'].astype(pd.Int64Dtype())
    role_model_data['birth_year'] = role_model_data['birth_year'].astype(pd.Int64Dtype())
    role_model_data['nationality'] = role_model_data['nationality'].astype(str)
    role_model_data['profession'] = role_model_data['profession'].astype(str)

    role_model_data = role_model_data.sort_values(by=['role_model', 'sex', 'birth_year', 'nationality', 'profession'])
    role_model_data = role_model_data.drop_duplicates()
    role_model_data = role_model_data.set_index('role_model')
    role_model_data = role_model_data[~role_model_data.index.duplicated(keep='first')]
    role_model_data.to_pickle(produces)


@pytask.mark.depends_on(BUILD_PATH / 'role_models/role_model_data.pkl')
@pytask.mark.produces(BUILD_PATH / 'role_models/ses.pkl')
def task_ses(produces: Path):
    """Retreive cleaned SES data
    """    
    ses_data = pd.read_excel(ASSET_PATH / 'Role_models_by_SES_precleaned.xlsx')
    role_model_data = pd.read_pickle(BUILD_PATH / 'role_models/role_model_data.pkl')
    ses_data = clean_ses_data(ses_data, role_model_data)
    ses_data.to_pickle(produces)


@pytask.mark.depends_on(BUILD_PATH / 'role_models/ses.pkl')
@pytask.mark.produces(BUILD_PATH / 'role_models/ses_mentions.pkl')
def task_ses_mentions(produces: Path):
    """List all role model mentions
    """
    ses_data = pd.read_pickle(BUILD_PATH / 'role_models/ses.pkl')
    mention_data = produce_role_model_mention_table(ses_data)
    mention_data.to_pickle(produces)


@pytask.mark.depends_on(BUILD_PATH / 'role_models/ses_mentions.pkl')
@pytask.mark.produces(BUILD_PATH / 'role_models/ses_scores.pkl')
def task_ses_model_scores(produces: Path):
    """ Assign scores to role models
    """
    mention_data = pd.read_pickle(BUILD_PATH / 'role_models/ses_mentions.pkl')
    score_data = produce_role_model_scores(
        mention_data=mention_data,
        rank_weights=[1/n for n in range(1, 6)]
    )
    score_data.to_pickle(produces)


@pytask.mark.depends_on(BUILD_PATH / 'role_models/ses.pkl')
@pytask.mark.produces(BUILD_PATH / 'role_models/ses_scores_distinct.pkl')
def task_ses_model_filtering(produces: Path):
    """Filter out non-SES-unique/non-distinct role models.
    """    
    scores = pd.read_pickle(BUILD_PATH / 'role_models/ses_scores.pkl')
    filtered_scores = filter_role_models(
        scores,
        minimum_count=1,
        require_distinct_SES=True
    )
    filtered_scores.to_pickle(produces)


# @pytask.mark.depends_on(BUILD_PATH / 'role_models/ses_scores_distinct.pkl')
# @pytask.mark.produces(BUILD_PATH / 'role_models/ses_scores_balanced.pkl')
# def task_ses_model_balancing(produces: Path):
#     """Balance role models
#     such that an equal amount of low- and high-SES role models is present in the dataset.
#     """
#     filtered_scores = pd.read_pickle(BUILD_PATH / 'role_models/ses_scores_distinct.pkl')
#     equilibrated_scores = equilibrate_role_models(filtered_scores)
#     equilibrated_scores.to_pickle(produces)