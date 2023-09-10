# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

TASK_NAME = 'mc'
stat_df = pd.read_csv(f'../../data/processed/{TASK_NAME}/{TASK_NAME}_stats.csv')
majority_df = pd.read_csv(f'../../data/processed/{TASK_NAME}/{TASK_NAME}_majority.csv')
new_majority_df = majority_df.reindex(columns=stat_df.columns)
combined_df = pd.concat([new_majority_df, stat_df], ignore_index=True)
combined_df.to_csv(f'../../data/processed/{TASK_NAME}/{TASK_NAME}_combined.csv', index=False)
num_llms = combined_df.shape[0]

# Set Seaborn style
sns.set(style="darkgrid")

# Initialize the matplotlib figure
plt.figure(figsize=(15, 8))

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
ax.set(
	xlabel='LLMs of Different Scales',
	ylabel='Accuracy (%)',
	title='Multiple Choices (MC) Accuracy Comparison Among LLMs'
	)
ax.grid(True)

# Custom legend
handles, labels = ax.get_legend_handles_labels()
legend_labels = ['Overall Accuracy (Excl. N/A)', 'Accuracy with 5 Options ABCDE (Excl. N/A)',
                 'Accuracy with 4 Options ABCD (Excl. N/A)']
ax.legend(handles=handles, title='Accuracy Type', labels=legend_labels)

sns.despine(left=True, bottom=True)

# Show plot
plt.savefig(f'{TASK_NAME}_acc.svg', bbox_inches='tight', pad_inches=0.1)
plt.show()
