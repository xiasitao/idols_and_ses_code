# %%
from lib_semantic_clustering import *
articles, human_annotated, articles_distinct, human_annotated_distinct, cluster_columns, cluster_topics, cluster_adjectives = retrieve_data()


#%% 
print_cluster_topics_words(cluster_adjectives, 20)


#%%
topics_distribution = find_topic_distributions(articles, cluster_columns)
topics_distribution_distinct = find_topic_distributions(articles_distinct, cluster_columns)


# %%
article_hypertopics = find_hypertopics(articles, hypertopic_table, hypertopics_columns)
hypertopics_distributions = find_topic_distributions(article_hypertopics, hypertopics_columns)
article_hypertopics_distinct = find_hypertopics(articles_distinct, hypertopic_table, hypertopics_columns)
hypertopics_distributions_distinct = find_topic_distributions(article_hypertopics_distinct, hypertopics_columns)


# %%
plot_human_annotation_confusion_matrix(article_hypertopics, human_annotated, 20)


# %%
plot_human_annotation_confusion_matrix(article_hypertopics_distinct, human_annotated_distinct, 20)


# %%
plot_accuracy_by_n(article_hypertopics, human_annotated)


# %%
plot_accuracy_by_n(article_hypertopics_distinct, human_annotated_distinct)


# %%
plot_hypertopic_distribution_by_n(hypertopics_distributions, hypertopics, find_articles_per_SES(articles))


# %%
plot_hypertopic_distribution_by_n(hypertopics_distributions_distinct, hypertopics, find_articles_per_SES(articles_distinct))


# %%
evaluate_for = 'cluster_20'
evaluate_hypertopics_for_n(hypertopics_distributions[evaluate_for], find_articles_per_SES(articles))


# %%
evaluate_hypertopics_for_n(hypertopics_distributions_distinct[evaluate_for], find_articles_per_SES(articles_distinct))


# %%
evaluate_hypertopics_for_n(topics_distribution[evaluate_for], find_articles_per_SES(articles))


# %%
evaluate_hypertopics_for_n(topics_distribution_distinct[evaluate_for], find_articles_per_SES(articles_distinct))


# %%
