# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

TASK_NAME = 'mc'
FILTERED: bool = False

if FILTERED:
	stat_path = f'../../data/processed/{TASK_NAME}/filtered/{TASK_NAME}_filtered_stats.csv'
	majority_path = f'../../data/processed/{TASK_NAME}/filtered/{TASK_NAME}_filtered_majority.csv'
else:
	stat_path = f'../../data/processed/{TASK_NAME}/unfiltered/{TASK_NAME}_unfiltered_stats.csv'
	majority_path = f'../../data/processed/{TASK_NAME}/unfiltered/{TASK_NAME}_unfiltered_majority.csv'
stat_df = pd.read_csv(stat_path)
majority_df = pd.read_csv(majority_path)
new_majority_df = majority_df.reindex(columns=stat_df.columns)
combined_df = pd.concat([new_majority_df, stat_df], ignore_index=True)
num_llms = combined_df.shape[0]

# Set Seaborn style
sns.set(style="darkgrid")

# Initialize the matplotlib figure
plt.figure(figsize=(12, 6))

# Create a colorful bar plot
ax = sns.barplot(
	data=combined_df.melt(
		id_vars='llm_name', value_vars=['accuracy_excl_nan', 'accuracy_with_E_excl_nan', 'accuracy_without_E_excl_nan']
		),
	x='llm_name',
	hue='variable',
	y='value',
	palette=sns.color_palette("pastel")
	)

# Adding numbers on top of the bars
max_letter = majority_df['majority_letter'].iloc[0]
max_letter_with_e = majority_df['majority_letter_with_E'].iloc[0]
max_letter_without_e = majority_df['majority_letter_without_E'].iloc[0]
ANNOT_HEIGHT_OFFSET = 1

for idx, p in enumerate(ax.patches):

	if idx == 0:
		annot_text = f'Maj: {max_letter}\n{p.get_height():.2f}'
		annot_pos = (p.get_x() + p.get_width() / 2., p.get_height() + ANNOT_HEIGHT_OFFSET)
	elif idx == 1 * num_llms:
		annot_text = f'Maj: {max_letter_with_e}\n{p.get_height():.2f}'
		annot_pos = (p.get_x() + p.get_width() / 2., p.get_height() + ANNOT_HEIGHT_OFFSET)
	elif idx == 2 * num_llms:
		annot_text = f'Maj: {max_letter_without_e}\n{p.get_height():.2f}'
		annot_pos = (p.get_x() + p.get_width() / 2., p.get_height() + ANNOT_HEIGHT_OFFSET)
	else:
		annot_text = f'{p.get_height():.2f}'
		annot_pos = (p.get_x() + p.get_width() / 2., p.get_height())
	ax.annotate(
		annot_text,
		annot_pos,
		ha='center', va='center',
		xytext=(0, 5),
		textcoords='offset points'
		)

# Customize the plot appearance
if FILTERED:
	title = "Multiple Choices (MC) Accuracy Comparison Among LLMs (Filtered)"
else:
	title = "Multiple Choices (MC) Accuracy Comparison Among LLMs"
ax.set(
	xlabel='LLMs of Different Scales',
	ylabel='Accuracy (%)',
	title=title
	)
ax.grid(True)

# Custom legend
handles, labels = ax.get_legend_handles_labels()
legend_labels = ['Overall Accuracy (Excl. N/A)', 'Accuracy with 5 Options ABCDE (Excl. N/A)',
                 'Accuracy with 4 Options ABCD (Excl. N/A)']
ax.legend(handles=handles, title='Accuracy Type', labels=legend_labels)

sns.despine(left=True, bottom=True)

# Show plot
if FILTERED:
	save_fig_path = f'filtered/{TASK_NAME}_filtered_acc.svg'
else:
	save_fig_path = f'unfiltered/{TASK_NAME}_unfiltered_acc.svg'
plt.savefig(save_fig_path, bbox_inches='tight', pad_inches=0.1)
plt.show()
