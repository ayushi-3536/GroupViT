import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# # Read the mIoU values for two runs from CSV files
# run1_df = pd.read_csv('/misc/lmbraid21/sharmaa/outputs/gs2_nounphrase_5_grouping/PascalVOCDataset_per_class_iou.csv')

# run2_df = pd.read_csv('/misc/lmbraid21/sharmaa/outputs/original_checkpoint_results/PascalVOCDataset_per_class_iou.csv')

run1_df = pd.read_csv('/misc/lmbraid21/sharmaa/outputs/gs2_nounphrase_5_grouping/COCOObjectDataset_per_class_iou.csv')

run2_df = pd.read_csv('/misc/lmbraid21/sharmaa/outputs/original_checkpoint_results/COCOObjectDataset_per_class_iou.csv')

# Merge the two dataframes based on the class name column
merged_df = pd.merge(run1_df, run2_df, on='Class Name', suffixes=('_run1', '_run2'))

# Calculate the absolute differences between the two runs
merged_df['abs_diff'] = merged_df['IoU Score_run1'] - merged_df['IoU Score_run2']
print("merged df", merged_df)

# Sort the classes in descending order based on the absolute differences
sorted_classes = merged_df.sort_values('abs_diff', ascending=False)['Class Name'].values

# Get the top 5 classes with the maximum differences
top_classes = sorted_classes[:5]

# Get the bottom 5 classes with the minimum differences
bottom_classes = sorted_classes[-5:]
miou_diffs = pd.DataFrame({'Class Name': merged_df['Class Name'],
                           'mIoU Difference':merged_df['abs_diff']})
sns.set_style('whitegrid')
sns.set(rc={'figure.figsize':(14,6)})
ax = sns.barplot(x='Class Name', y='mIoU Difference', data=miou_diffs,
            color='red', saturation=.5, errwidth=0)
ax.tick_params(axis='x', which='major', labelsize=8)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
plt.subplots_adjust(left=0.03, bottom=0.2, right=0.999999, top=0.93)

plt.savefig('miou_diffs_coco.png', dpi=1000, bbox_inches='tight')
# Create a horizontal bar chart to plot the mIoU differences
