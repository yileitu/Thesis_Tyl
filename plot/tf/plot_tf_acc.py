# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

TASK_NAME = 'tf'
stat_df = pd.read_csv(f'../../data/processed/{TASK_NAME}/{TASK_NAME}_stats.csv')

# Set Seaborn style
sns.set(style="darkgrid")

# Initialize the matplotlib figure
plt.figure(figsize=(15, 8))

# Create a colorful bar plot
ax = sns.barplot(
	data=stat_df.melt(
		id_vars='llm_name', value_vars=['accuracy_excl_nan', 'accuracy_incl_nan']
		),
	x='llm_name',
	hue='variable',
	y='value',
	palette=sns.color_palette("pastel")
	)

# Customize the plot appearance
ax.set(
	xlabel='LLMs of Different Scales',
	ylabel='Accuracy (%)',
	title='True/False (TF) Questions Accuracy Comparison Among LLMs'
	)
ax.grid(True)

# Custom legend
handles, labels = ax.get_legend_handles_labels()
legend_labels = ['Accuracy (Excl. N/A)', 'Accuracy (Incl. N/A)']
ax.legend(handles=handles, title='Accuracy Type', labels=legend_labels)

sns.despine(left=True, bottom=True)

# Show plot
plt.savefig(f'{TASK_NAME}_acc.svg', bbox_inches='tight', pad_inches=0.1)
plt.show()
