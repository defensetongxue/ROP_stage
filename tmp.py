import os
import json
import numpy as np

# 加载记录文件
record_path = './final.json'
if os.path.exists(record_path):
    with open(record_path, 'r') as file:
        record = json.load(file)
else:
    raise FileNotFoundError(f"The record file {record_path} does not exist.")

# 初始化存储统计结果的字典
metrics_summary = {
    'recall': {"Class 1": [], "Class 2": [], "Class 3": []},
    'auc': {"Class 1": [], "Class 2": [], "Class 3": []},
    'f1': {"Class 1": [], "Class 2": [], "Class 3": []},
    'accuracy': {"Class 1": [], "Class 2": [], "Class 3": []}
}

# 收集所有记录
for split_name, metrics in record.items():
    for metric_name, classes in metrics.items():
        for class_name, value in classes.items():
            metrics_summary[metric_name][class_name].append(value)

# 计算均值和标准差
summary_stats = {
    'recall': {},
    'auc': {},
    'f1': {},
    'accuracy': {}
}

for metric_name, classes in metrics_summary.items():
    for class_name, values in classes.items():
        values_array = np.array(values)
        mean_value = np.mean(values_array)
        std_value = np.std(values_array)
        summary_stats[metric_name][class_name] = [round(mean_value, 4), round(std_value, 4)]

# 打印或保存统计结果
print(json.dumps(summary_stats, indent=4))

# 保存统计结果到文件
summary_path = './summary.json'
with open(summary_path, 'w') as file:
    json.dump(summary_stats, file, indent=4)

print(f"Summary statistics have been saved to {summary_path}")
