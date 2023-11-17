
import matplotlib.pyplot as plt
import numpy as np

# Sample data
baseline = {
    "1": {"acc": "0.9437", "auc": "0.9407", "sensitivity": "0.5877"},
    "2": {"acc": "0.9297", "auc": "0.8788", "sensitivity": "0.5060"},
    "3": {"acc": "0.9513", "auc": "0.9305", "sensitivity": "0.5566"},
    "4": {"acc": "0.9450", "auc": "0.9401", "sensitivity": "0.4548"}
}

ours = {
    "1": {"Best_epoch": {"Auc": 0.9726, "Acc": 0.9572, "Recall": 1.0}},
    "2": {"Best_epoch": {"Auc": 0.9693, "Acc": 0.9448, "Recall": 1.0}},
    "3": {"Best_epoch": {"Auc": 0.9497, "Acc": 0.9513, "Recall": 1.0}},
    "4": {"Best_epoch": {"Auc": 0.9811, "Acc": 0.9498, "Recall": 1.0}}
}

# Function to calculate mean of metrics
def calculate_mean(data, metric):
    values = [float(data[key][metric]) for key in data if key.isdigit()]
    return np.mean(values)

# Calculate mean metrics for baseline and ours
baseline_acc_mean = calculate_mean(baseline, 'acc')
baseline_auc_mean = calculate_mean(baseline, 'auc')
baseline_recall_mean = calculate_mean(baseline, 'sensitivity')

ours_data = {k: v['Best_epoch'] for k, v in ours.items() if k.isdigit()}
ours_acc_mean = calculate_mean(ours_data, 'Acc')
ours_auc_mean = calculate_mean(ours_data, 'Auc')
ours_recall_mean = calculate_mean(ours_data, 'Recall')

# Metrics for plotting
metrics = ['Accuracy', 'AUC', 'Recall']
baseline_means = [baseline_acc_mean, baseline_auc_mean, baseline_recall_mean]
ours_means = [ours_acc_mean, ours_auc_mean, ours_recall_mean]

# Colors
ours_colors = [(58/255, 27/255, 25/255), (199/255, 160/255, 133/255), (201/255, 71/255, 55/255)]
baseline_colors = [(194/255, 207/255, 162/255), (164/255, 121/255, 158/255), (122/255, 102/255, 114/255)]

# Plotting
x_ours = np.arange(len(metrics))  # Positions for the 'ours' metrics
x_baseline = np.arange(len(metrics)) + len(metrics) + 1  # Positions for the 'baseline' metrics

# Plotting
fig, ax = plt.subplots()
for i in range(len(metrics)):
    ax.bar(x_ours[i], ours_means[i],1, label='Ours' if i == 0 else "", color=ours_colors[i])
    ax.bar(x_baseline[i], baseline_means[i],1, label='Baseline' if i == 0 else "", color=baseline_colors[i])

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Scores')
ax.set_title('Comparison of Metrics')
ax.set_xticks(np.concatenate([x_ours, x_baseline]))
ax.set_xticklabels(metrics * 2)  # Repeat the labels for both methods

# Place legend outside the plot
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

# Attach a text label above each bar in *rects*, displaying its height (as before)
def autolabel(x, heights, colors):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for i, height in enumerate(heights):
        ax.annotate(f'{height:.4f}',
                    xy=(x[i], height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', color=colors[i])

autolabel(x_ours, ours_means, ours_colors)
autolabel(x_baseline, baseline_means, baseline_colors)

fig.tight_layout()

plt.savefig('./comparison_chart_colored.png', bbox_inches='tight')
plt.show()
