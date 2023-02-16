# %%
from lib_zero_shot import *
articles, human_annotated, articles_distinct, human_annotated_distinct, category_columns = retrieve_data()


# %%
evaluate_category(articles, 'topic', category_columns, articles_per_SES=find_articles_per_SES(articles), human_annotated=human_annotated, is_distinct=False)
# %%
evaluate_category(articles_distinct, 'topic', category_columns, articles_per_SES=find_articles_per_SES(articles_distinct), human_annotated=human_annotated_distinct, is_distinct=True)
# %%
evaluate_category(articles, 'topic_l', category_columns, articles_per_SES=find_articles_per_SES(articles), human_annotated=human_annotated, is_distinct=False)
# %%
evaluate_category(articles_distinct, 'topic_l', category_columns, articles_per_SES=find_articles_per_SES(articles_distinct), human_annotated=human_annotated_distinct, is_distinct=True)

# %%
evaluate_category(articles_distinct, 'prosociality', category_columns, articles_per_SES=find_articles_per_SES(articles_distinct), human_annotated=human_annotated, is_distinct=True)


# %%
filtered_articles = filter_out_low_entropy_labels(articles, 0.3, category_columns)
evaluate_category(filtered_articles, 'topic_l', category_columns, articles_per_SES=find_articles_per_SES(filtered_articles, column='topic_l'), human_annotated=human_annotated, is_distinct=False)
# %%
human_annotated['topic'].count()
# %%
