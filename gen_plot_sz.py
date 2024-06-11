import json
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.font_manager import FontProperties

# Load JSON data from file
with open('./experiments/shen_record.json', 'r') as file:
    data = json.load(file)

# Load custom font
font_path = './arial.ttf'
custom_font = FontProperties(fname=font_path, size=30)

# Determine the highest accuracy for each model
model_accuracies = {}
for model, params in data.items():
    accuracies = [float(details['Accuracy']) for details in params.values()]
    highest_accuracy = max(accuracies)
    model_accuracies[model] = highest_accuracy

# Colors for the bars (e.g., Tableau 10)
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

# Create a plot with specified figure size
fig, ax = plt.subplots(figsize=(14, 8))
models = list(model_accuracies.keys())
accuracies = [model_accuracies[model] for model in models]
normalized_accuracies = [(acc - 0.9) / 0.1 for acc in accuracies]

# Bar plot with custom colors
bars = ax.bar(models, normalized_accuracies, color=colors[:len(models)])

# Add value labels above bars
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval * 0.1 + 0.9:.4f}', va='bottom', ha='center', fontproperties=custom_font)

# Setting labels with 45 degree tilt and custom font
plt.xticks(rotation=45, fontproperties=custom_font)
ax.set_ylim([0, 1])  # Set y-axis to show normalized range
ax.set_ylabel('Accuracy', fontproperties=custom_font)
ax.set_title('Highest Model Accuracies', fontproperties=custom_font)

# Customize y-axis ticks to reflect actual data values
yticks = np.linspace(0, 1, 11)  # Creates 11 ticks from 0 to 1
ylabels = [f'{0.9 + ytick * 0.1:.4f}' for ytick in yticks]
ax.set_yticks(yticks)  # Set the y-tick positions
ax.set_yticklabels(ylabels, fontproperties=custom_font)  # Set the y-tick labels

# Ensure the output directory exists
os.makedirs('./experiments', exist_ok=True)

# Save the figure
plt.savefig('./experiments/shen.png', dpi=300, bbox_inches='tight')
plt.close()
