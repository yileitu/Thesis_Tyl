# -*- coding: utf-8 -*-
import pandas as pd

df_processed = pd.read_csv('../../data/templated/tf/test_data.csv')
df_original = pd.read_csv(
	# '../../data/processed/mc/unfiltered/labeled/tf_labeled_answers.csv'
	'/Users/tuyilei/Desktop/Thesis/Thesis_Tyl/data/processed/tf/unfiltered/labeled/tf_labeled_answers.csv'
	# '/Users/tuyilei/Desktop/Thesis/Thesis_Tyl/data/processed/qa/without_chatgpt/sample/labeled/qa_labeled.csv'
	)

# merged_data = pd.merge(df_processed, df_original[['id', 'data_source', 'anomaly']], on='id', how='inner')
merged_data = pd.merge(df_processed, df_original[['id', 'anomaly']], on='id', how='inner')
anomalous_data = merged_data[merged_data['anomaly'] == 'Anomalous']

# data_source_counts = merged_data['data_source'].value_counts()
# data_source_proportions = merged_data['data_source'].value_counts(normalize=True) * 100
#
# anomaly_counts = anomalous_data['data_source'].value_counts()
# total_counts = merged_data['data_source'].value_counts()
# anomaly_proportions = anomaly_counts / total_counts
#
# print("Anomaly counts:", anomaly_counts)
# print("Total counts:", total_counts)
# print("Anomaly proportions:", anomaly_proportions)

total_anomalous_count = anomalous_data.shape[0]
total_entries_count = merged_data.shape[0]
overall_anomalous_proportion = total_anomalous_count / total_entries_count
print("\nOverall anomalous proportion:", overall_anomalous_proportion)

# # Create more descriptive output
# output = pd.DataFrame(
# 	{
# 		'Data Source'    : data_source_counts.index,
# 		'Counts'         : data_source_counts.values,
# 		'Proportions (%)': data_source_proportions.values
# 		}
# 	)
# total_counts = output['Counts'].sum()
# total_proportions = output['Proportions (%)'].sum()
# total_df = pd.DataFrame(
# 	{
# 		'Data Source'    : ['Total'],
# 		'Counts'         : [total_counts],
# 		'Proportions (%)': [total_proportions]
# 		}
# 	)
# output = pd.concat([output, total_df], ignore_index=True)
# output.to_csv('mc_test_stat.csv', index=False)
