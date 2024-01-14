import os
import json
from configs import get_config

if __name__ == '__main__':
    args = get_config()
    
    # Load annotations data
    with open(os.path.join(args.data_path, 'annotations.json'), 'r') as f:
        data_dict = json.load(f)

    # Create the stage_split directory if it doesn't exist
    os.makedirs('./stage_split', exist_ok=True)

    # Determine the split files to process
    get_all_split = True
    if get_all_split:
        split_file_list = [os.path.join(args.data_path, 'split', f"clr_{i}.json") for i in ['1', '2', '3', '4']]
    else:
        split_file_list = [os.path.join(args.data_path, 'split', f'clr_{args.split_name}.json')]

    # Process each split file
    for json_file in split_file_list:
        with open(json_file, 'r') as f:
            split_file = json.load(f)

        # Create a new split dictionary
        new_split = {}
        for split in split_file:
            new_split[split] = [image_name for image_name in split_file[split] 
                                if data_dict[image_name]['stage'] > 0 
                                and data_dict[image_name]['ridge_seg']['max_val'] >= 0.5]

        # Save the new split to a file
        save_path = os.path.join('./stage_split', os.path.basename(json_file))
        with open(save_path, 'w') as f:
            json.dump(new_split, f, indent=4)

    print("Split files saved successfully.")

    