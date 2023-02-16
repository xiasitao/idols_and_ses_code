# %%
from lib_topic_modelling import  *
articles, human_annotated, articles_distinct, human_annotated_distinct, topic_columns, topic_words = retrieve_data()

# Sanity checks
assert(articles.groupby('role_model').count()['content'].unique() == np.array([50]))


# %%
evaluate_topics_for_n(articles, topic_columns, 5, articles_per_SES=find_articles_per_SES(articles), is_distinct=False)


# %%
evaluate_topics_for_n(articles_distinct, topic_columns, 5, articles_per_SES=find_articles_per_SES(articles_distinct), is_distinct=True)


# %%
def hypertopic_crosscheck(hypertopic_table: "dict[int, list[str]]", topic_words, n_topics):
    for hypertopic in hypertopics:
        indices = [n for n, val in enumerate(hypertopic_table[n_topics]) if val==hypertopic]
        words = [topic_words[n_topics][i] for i in indices]
        print(hypertopic.upper())
        print(f'  {words}\n')
hypertopic_crosscheck(hypertopic_table, topic_words, 60)


# %%
article_hypertopics = find_hypertopics(articles, columns=hypertopics_columns, hypertopic_table=hypertopic_table)
hypertopics_distributions = find_topic_distributions(article_hypertopics, hypertopics_columns)
plot_hypertopic_distribution_by_n(hypertopics_distributions, hypertopics, articles_per_SES=find_articles_per_SES(articles))


# %% 
article_hypertopics_distinct = find_hypertopics(articles_distinct, columns=hypertopics_columns, hypertopic_table=hypertopic_table)
hypertopics_distributions_distinct = find_topic_distributions(article_hypertopics_distinct, hypertopics_columns)
plot_hypertopic_distribution_by_n(hypertopics_distributions_distinct, hypertopics, articles_per_SES=find_articles_per_SES(articles_distinct))


# %%
plot_human_annotation_confusion_matrix(article_hypertopics, human_annotated, 40)


# %%
plot_human_annotation_confusion_matrix(article_hypertopics_distinct, human_annotated_distinct, 40)


# %%
to_evaluate = 'topic_50'
plot_topic_distribution(hypertopics_distributions_distinct[to_evaluate])
print(chi2_contingency_test(hypertopics_distributions_distinct[to_evaluate]))
print(chi2_per_label_test(hypertopics_distributions_distinct[to_evaluate], find_articles_per_SES(articles_distinct)))


# %%
plot_accuracy_by_n(article_hypertopics, human_annotated)


# %%
plot_accuracy_by_n(article_hypertopics_distinct, human_annotated_distinct)


# %%
# Filtering by entropy
articles_filtered = filter_out_low_entropy_labels(articles, percentile=0.50, topic_columns=topic_columns)
article_hypertopics_filtered = find_hypertopics(articles_filtered, hypertopic_table, hypertopic_columns)
hypertopics_distributions_filtered = find_topic_distributions(article_hypertopics_filtered, hypertopic_columns)
plot_hypertopic_distribution_by_n(hypertopics_distributions_filtered, hypertopics, articles_per_SES=find_articles_per_SES(articles_filtered))
# %%
plot_accuracy_by_n(article_hypertopics_filtered, human_annotated)
# %%
to_evaluate = 'topic_15'
plot_topic_distribution(hypertopics_distributions_filtered[to_evaluate])
print(chi2_contingency_test(hypertopics_distributions_filtered[to_evaluate]))
print(chi2_per_label_test(hypertopics_distributions_filtered[to_evaluate], find_articles_per_SES(articles_filtered, to_evaluate)))


# %%
