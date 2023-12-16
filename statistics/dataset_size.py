# -*- coding: utf-8 -*-
import pandas as pd

df_processed = pd.read_csv('/Users/tuyilei/Desktop/Thesis/Thesis_Tyl/data/templated/mc/test_data.csv')
df_original = pd.read_csv(
	'/Users/tuyilei/Desktop/Thesis/Thesis_Tyl/data/processed/mc/unfiltered/labeled/mc_labeled_answers.csv'
	)

merged_data = pd.merge(df_processed, df_original[['id', 'data_source']], on='id', how='inner')
# Calculate counts and proportions for each 'data_source'
data_source_counts = merged_data['data_source'].value_counts()
data_source_proportions = merged_data['data_source'].value_counts(normalize=True) * 100

# Create more descriptive output
output = pd.DataFrame(
	{
		'Data Source'    : data_source_counts.index,
		'Counts'         : data_source_counts.values,
		'Proportions (%)': data_source_proportions.values
		}
	)
total_counts = output['Counts'].sum()
total_proportions = output['Proportions (%)'].sum()
total_df = pd.DataFrame(
	{
		'Data Source'    : ['Total'],
		'Counts'         : [total_counts],
		'Proportions (%)': [total_proportions]
		}
	)
output = pd.concat([output, total_df], ignore_index=True)
output.to_csv('mc_test_stat.csv', index=False)
