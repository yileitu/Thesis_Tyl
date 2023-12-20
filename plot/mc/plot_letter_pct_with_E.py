import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 1. 读取数据
TASK_NAME = 'mc'
FILTERED: bool = False
if FILTERED:
	stat_path = f'../../data/processed/{TASK_NAME}/filtered/{TASK_NAME}_filtered_stats.csv'
	gt_stat_path = f'../../data/processed/{TASK_NAME}/filtered/{TASK_NAME}_filtered_gt_stats.csv'
else:
	stat_path = f'../../data/processed/{TASK_NAME}/unfiltered/{TASK_NAME}_unfiltered_stats.csv'
	gt_stat_path = f'../../data/processed/{TASK_NAME}/unfiltered/{TASK_NAME}_unfiltered_gt_stats.csv'
stat_df = pd.read_csv(stat_path)
gt_stat_df = pd.read_csv(gt_stat_path)

prop_gt_A = gt_stat_df['prop_gt_A'].iloc[0]
prop_gt_B = gt_stat_df['prop_gt_B'].iloc[0]
prop_gt_C = gt_stat_df['prop_gt_C'].iloc[0]
prop_gt_D = gt_stat_df['prop_gt_D'].iloc[0]
prop_gt_E = gt_stat_df['prop_gt_E'].iloc[0]
gt = pd.DataFrame(
	{
		'llm_name'     : ["Ground Truth"],
		'prop_A_with_E': [prop_gt_A],
		'prop_B_with_E': [prop_gt_B],
		'prop_C_with_E': [prop_gt_C],
		'prop_D_with_E': [prop_gt_D],
		'prop_E_with_E': [prop_gt_E]
		}
	)

gt = gt.reindex(columns=stat_df.columns)
stat_df = pd.concat([gt, stat_df], ignore_index=True)

columns = ["prop_A_with_E", "prop_B_with_E", "prop_C_with_E", "prop_D_with_E", "prop_E_with_E"]
stat_df[columns] *= 100
legend_names = ["A", "B", "C", "D", "E"]  # 新的图例名称

plt.figure(figsize=(6, 5))
colors = sns.color_palette("pastel", len(columns))
sns.set(style="darkgrid")

bottom_values = [0] * len(stat_df)
for idx, column in enumerate(columns):
	plt.bar(stat_df['llm_name'], stat_df[column], label=legend_names[idx], bottom=bottom_values, color=colors[idx])
	for bar_idx, (name, value) in enumerate(zip(stat_df['llm_name'], stat_df[column])):
		if value > 0:  # 只为非零部分添加标签
			plt.text(
				bar_idx, bottom_values[bar_idx] + value / 2, f"{value:.2f}", ha='center', va='center',
				color='black', fontsize=8
				)  # 由于pastel颜色较浅，所以我建议使用黑色标签
		bottom_values[bar_idx] += value

plt.legend(title="Option\nLetters", loc="upper left", bbox_to_anchor=(1, 1))
plt.ylim(0, 100)
if FILTERED:
	title = "Multiple Choices (MC) with Options ABCDE Response Percentages Among LLMs (Filtered)"
else:
	title = "Multiple Choices (MC) with Options ABCDE Response Percentages Among LLMs"

plt.xticks(rotation=15)
plt.title(title)
plt.xlabel("LLMs of Different Scales")
plt.ylabel("Option Percentages (%)")
plt.tight_layout()
if FILTERED:
	save_fig_path = f'filtered/{TASK_NAME}_filtered_letter_pct_with_E.svg'
else:
	save_fig_path = f'unfiltered/{TASK_NAME}_unfiltered_letter_pct_with_E.svg'
plt.savefig(save_fig_path, bbox_inches='tight', pad_inches=0.1)
plt.show()
