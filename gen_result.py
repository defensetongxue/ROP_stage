# import json
# import numpy as np
# import os

# def mean_std(values):
#     """Calculate mean and standard deviation formatted for LaTeX."""
#     mean = np.mean(values)
#     std = np.std(values)
#     return f"${mean:.4f} \\pm {std:.4f}$"

# # Load JSON data
# src_json_file = './experiments/different_model.json'
# save_path = './experiments/fs_table_transposed.txt'

# with open(src_json_file, 'r') as f:
#     data = json.load(f)

# # Collect data in a structured way: {metric: {model: value, ...}, ...}
# metrics_data = {}

# # Initialize structure for each model
# models = set()
# for model_name, parameters in data.items():
#     models.add(model_name)
#     for param_name, results in parameters.items():
#         for result_type, splits in results.items():  # 'patch' or 'all'
#             prefix = 'Patch Level' if result_type == 'patch' else 'Image Level'
#             for split_name, metrics in splits.items():
#                 for metric, value in metrics.items():
#                     metric_label = f"{prefix} {metric.replace('_', ' ').capitalize()}"
#                     if metric_label not in metrics_data:
#                         metrics_data[metric_label] = {}
#                     if model_name not in metrics_data[metric_label]:
#                         metrics_data[metric_label][model_name] = []
#                     metrics_data[metric_label][model_name].append(float(value))

# # Start building the LaTeX table
# table_content = "\\begin{table}[ht]\n"
# table_content += "\\centering\n"
# table_content += "\\caption{Summary of Model Performance Across Different Splits}\n"
# table_content += "\\label{tab:model_performance_transposed}\n"
# table_content += "\\begin{tabular}{l" + "c" * len(models) + "}\n"  # One column for each model
# table_content += "\\toprule\n"
# table_content += "Metric & " + " & ".join(models) + " \\\\\n"  # Header row
# table_content += "\\midrule\n"

# # Fill in the rows for each metric
# for metric, model_values in metrics_data.items():
#     row_values = [mean_std(model_values[model]) if model in model_values else 'N/A' for model in models]
#     table_content += metric + " & " + " & ".join(row_values) + " \\\\\n"

# # Close the LaTeX table structure
# table_content += "\\bottomrule\n"
# table_content += "\\end{tabular}\n"
# table_content += "\\end{table}\n"

# # Save the LaTeX table content to a file
# with open(save_path, 'w') as f:
#     f.write(table_content)

# print(f"LaTeX table saved to {save_path}")
