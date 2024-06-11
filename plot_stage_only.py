import json,os
import matplotlib.pyplot as plt
with open('./experiments/record.json') as f:
    data_record=json.load(f)["efficientnet_b7"]
# Assuming `data_record` is already loaded as described
def reformat_data(data_record):
    metrics = {
        "patch_level_acc": ("patch", "Accuracy"),
        "patch_level_auc": ("patch", "AUC"),
        "auc": ("all", "AUC"),
        "acc": ("all", "Accuracy"),
        "1_recall": ("all", "1_recall"),
        "2_recall": ("all", "2_recall"),
        "3_recall": ("all", "3_recall")
    }

    structured_data = {metric: {} for metric in metrics}

    for key, values in data_record.items():
        params = key.split('_')
        param_dict = {"ridge_seg_number": params[0], "sample_distance": params[1], "patch_size": params[2]}
        
        for metric, (category, name) in metrics.items():
            if category in values and name in values[category]["clr_1"]:
                value = float(values[category]["clr_1"][name])
                
                for parameter_name in ["ridge_seg_number", "sample_distance", "patch_size"]:
                    parameter_value = param_dict[parameter_name]
                    if parameter_name not in structured_data[metric]:
                        structured_data[metric][parameter_name] = {}
                    structured_data[metric][parameter_name][parameter_value] = value

    for metric in structured_data:
        for parameter in structured_data[metric]:
            sorted_items = sorted(structured_data[metric][parameter].items(), key=lambda x: float(x[0]) if x[0].isdigit() else -1)
            structured_data[metric][parameter] = dict(sorted_items)

    return structured_data
structured_data = reformat_data(data_record)
with open('./experiments/save_result.json','w') as f:
    json.dump(structured_data,f)
def ensure_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def plot_and_save_metrics(structured_data, output_dir='./experiments/plot'):
    ensure_directory(output_dir)

    for metric, data in structured_data.items():
        for parameter_name, values in data.items():
            plt.figure(figsize=(10, 6))  # Specify a figure size
            # Sorting data to maintain sequence
            sorted_data = sorted(values.items(), key=lambda item: float(item[0]))
            x, y = zip(*sorted_data)  # Ensure data is in the correct sequence

            plt.plot(x, y, marker='o', linestyle='-', color='b', label=f'{metric} ({parameter_name})')
            plt.title(f'{metric} by {parameter_name}')
            plt.xlabel(parameter_name)
            plt.ylabel('Value')
            plt.grid(True)
            plt.legend(loc='best')  # Include a legend to identify the plotted lines
            plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility

            filename = f"{parameter_name}_{metric}.png"
            file_path = os.path.join(output_dir, filename)
            plt.savefig(file_path, bbox_inches='tight')  # Save the plot with tight bounding box to include all labels
            plt.close()
            print(f"Saved plot to {file_path}")
plot_and_save_metrics(structured_data)