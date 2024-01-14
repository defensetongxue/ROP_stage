import os
import json
from configs import get_config

if __name__ == '__main__':
    args = get_config()
    with open(os.path.join(args.data_path, 'annotations.json'), 'r') as f:
        data_dict = json.load(f)

    # Iterate through data_dict and remove 'ridge_seg' for jpg images
    for image_name, data in list(data_dict.items()):  # Using list() to avoid RuntimeError due to dictionary size change
        if 'ridge_seg' in data and "ridge_seg_path" in data['ridge_seg']:
            ridge_seg_path = data['ridge_seg']['ridge_seg_path']
            if ridge_seg_path is not None and ridge_seg_path.endswith('.jpg'):
                del data_dict[image_name]['ridge_seg']

    # Save the updated data_dict back to the original JSON file
    with open(os.path.join(args.data_path, 'annotations.json'), 'w') as f:
        json.dump(data_dict, f, indent=4)
