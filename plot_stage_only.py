import json
import matplotlib.pyplot as plt
import os

def decode_key(key):
    ridge_seg_number, sample_distance, sample_low_threshold, patch_size = key.split('_')
    return {
        'ridge_seg_number': int(ridge_seg_number),
        'sample_distance': int(sample_distance),
        'sample_low_threshold': int(sample_low_threshold) / 100,
        'patch_size': int(patch_size)
    }

def load_data(record_path):
    with open(record_path, 'r') as f:
        return json.load(f)

def verify_splits(record, required_splits):
    for key, data in record.items():
        missing_splits = [split for split in required_splits if split not in data]
        if missing_splits:
            raise ValueError(f"Missing splits {missing_splits} for parameters {decode_key(key)}")

def calculate_mean_results(record):
    for key, data in record.items():
        acc_values = [data[split]['acc'] for split in data]
        auc_values = [data[split]['auc'] for split in data]
        record[key]['result'] = {
            'mean_acc': sum(acc_values) / len(acc_values),
            'mean_auc': sum(auc_values) / len(auc_values)
        }

def plot_results(record, param_name):
    os.makedirs(f"./experiments/{param_name}", exist_ok=True)
    param_to_key = {decode_key(key)[param_name]: key for key in record}
    x_values = sorted(param_to_key.keys())
    acc_values = [record[param_to_key[x_val]]['result']['mean_acc'] for x_val in x_values]
    auc_values = [record[param_to_key[x_val]]['result']['mean_auc'] for x_val in x_values]

    # Plotting Accuracy
    plt.figure()
    plt.plot(x_values, acc_values, label='Accuracy')
    plt.xlabel(param_name)
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy vs {param_name}')
    plt.xticks(x_values)  # Set the x-ticks to the x_values only
    plt.savefig(f"./experiments/{param_name}/acc.png")
    plt.close()

    # Plotting AUC
    plt.figure()
    plt.plot(x_values, auc_values, label='AUC')
    plt.xlabel(param_name)
    plt.ylabel('AUC')
    plt.title(f'AUC vs {param_name}')
    plt.xticks(x_values)  # Set the x-ticks to the x_values only
    plt.savefig(f"./experiments/{param_name}/auc.png")
    plt.close()


if __name__ == '__main__':
    record_path = './experiments/record_stage.json'
    record = load_data(record_path)
    required_splits = ['1', '2', '3', '4']
    verify_splits(record, required_splits)
    calculate_mean_results(record)

    # for param in ['ridge_seg_number', 'sample_distance', 'sample_low_threshold', 'patch_size']:
    for param in ['patch_size']:
        plot_results(record, param)
