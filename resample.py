import os
import json
from PIL import Image
from torchvision import transforms
import torch
from configs import get_config
from util.tools import k_max_values_and_indices

if __name__ == '__main__':
    args = get_config()
    with open(os.path.join(args.data_path, 'annotations.json'), 'r') as f:
        data_dict = json.load(f)

    for image_name, data in data_dict.items():
        if 'ridge_seg' in data and "ridge_seg_path" in data['ridge_seg'] and data['ridge_seg']['ridge_seg_path'] is not None:
            ridge_seg_path = data['ridge_seg']['ridge_seg_path']

            # Load the image and transform it into a torch tensor
            image = Image.open(ridge_seg_path).convert('L')  # Convert to grayscale if needed
            transform = transforms.Compose([
                transforms.ToTensor(),
                lambda x: x / 255
            ])
            output_img = transform(image).squeeze()

            # Process the image with k_max_values_and_indices
            maxval, pred_point = k_max_values_and_indices(output_img, args.ridge_seg_number, r=args.sample_distance, threshold=0.3)

            # Prepare the lists for updating the dictionary
            value_list = [round(float(value), 2) for value in maxval]
            point_list = [[int(x), int(y)] for y, x in pred_point]

            # Update data_dict
            data_dict[image_name]['ridge_seg'].update({
                "value_list": value_list,
                "point_list": point_list,
                "sample_number":args.ridge_seg_number,
                "sample_interval":args.sample_distance
            })

    # Optionally, save the updated data_dict back to file
    with open(os.path.join(args.data_path, 'annotations_updated.json'), 'w') as f:
    # with open('./annotations_updated.json', 'w') as f:
        json.dump(data_dict, f, indent=4)
