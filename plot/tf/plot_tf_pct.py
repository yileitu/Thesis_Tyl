import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 1. 读取数据
TASK_NAME = 'tf'
FILTERED: bool = True
if FILTERED:
	stat_path = f'../../data/processed/{TASK_NAME}/filtered/{TASK_NAME}_filtered_stats.csv'
	gt_stat_path = f'../../data/processed/{TASK_NAME}/filtered/{TASK_NAME}_filtered_gt_stats.csv'
else:
	stat_path = f'../../data/processed/{TASK_NAME}/unfiltered/{TASK_NAME}_unfiltered_stats.csv'
	gt_stat_path = f'../../data/processed/{TASK_NAME}/unfiltered/{TASK_NAME}_unfiltered_gt_stats.csv'
stat_df = pd.read_csv(stat_path)
gt_stat_df = pd.read_csv(gt_stat_path)
print(gt_stat_df)

prop_gt_ture = gt_stat_df['prop_gt_True'].iloc[0]
prop_gt_false = gt_stat_df['prop_gt_False'].iloc[0]
gt = pd.DataFrame(
	{
		'llm_name'  : ["Ground Truth"],
		'prop_True' : [prop_gt_ture],
		'prop_False': [prop_gt_false],
		}
	)

gt = gt.reindex(columns=stat_df.columns)
stat_df = pd.concat([gt, stat_df], ignore_index=True)

columns = ["prop_True", "prop_False"]
stat_df[columns] *= 100
legend_names = ["True", "False"]

plt.figure(figsize=(12, 10))
colors = sns.color_palette("pastel", len(columns))
sns.set(style="darkgrid")

bottom_values = [0] * len(stat_df)
for idx, column in enumerate(columns):
	plt.bar(stat_df['llm_name'], stat_df[column], label=legend_names[idx], bottom=bottom_values, color=colors[idx])
	for bar_idx, (name, value) in enumerate(zip(stat_df['llm_name'], stat_df[column])):
		plt.text(
			bar_idx, bottom_values[bar_idx] + value / 2, f"{value:.2f}", ha='center', va='center',
			color='black'
			)
		bottom_values[bar_idx] += value

plt.legend(title="T/F", loc="upper left", bbox_to_anchor=(1, 1))
plt.ylim(0, 100)
if FILTERED:
	title = "True/False (TF) Response Percentages Among LLMs (Filtered)"
else:
	title = "True/False (TF) Response Percentages Among LLMs (Unfiltered)"
plt.title(title)
plt.xlabel("LLMs of Different Scales")
plt.ylabel("T/F Percentages (%)")
plt.tight_layout()
if FILTERED:
	save_fig_path = f'filtered/{TASK_NAME}_filtered_tf_pct.svg'
else:
	save_fig_path = f'unfiltered/{TASK_NAME}_unfiltered_tf_pct.svg'
plt.savefig(save_fig_path, bbox_inches='tight', pad_inches=0.1)
plt.show()
