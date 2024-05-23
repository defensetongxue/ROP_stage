import json
import matplotlib.pyplot as plt
import os

# Paths to JSON files
crop_method_path = './experiments/0507.json'
overall_method_path = './experiments/0508.json'
# Load data from JSON files
with open(crop_method_path, 'r') as f:
    crop_data = json.load(f)
with open(overall_method_path, 'r') as f:
    overall_data = json.load(f)

# Function to extract and sort metrics from the JSON data
def extract_auc(data):
    epochs = sorted(map(int, data.keys()))
    auc = [float(data[str(epoch)]["auc"]) for epoch in epochs]
    return epochs, auc

# Extract AUC from both methods
epochs_crop, auc_crop = extract_auc(crop_data)
epochs_overall, auc_overall = extract_auc(overall_data)

# Ensure output directory exists
output_dir = './experiments'
os.makedirs(output_dir, exist_ok=True)

# Create the AUC plot
fig, ax = plt.subplots(figsize=(12, 8))

# Plot AUC
ax.plot(epochs_crop, auc_crop, label='Crop Method', marker='o', color='blue')
ax.plot(epochs_overall, auc_overall, label='Overall Method', marker='s', color='orange')
ax.set_xlabel('Epoch')
ax.set_ylabel('AUC')
ax.set_title('Comparison of AUC between Crop and Overall Methods')
ax.legend()
ax.grid(True)

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'compare.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"Comparison figure saved to {os.path.join(output_dir, 'compare.png')}")
