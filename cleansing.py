import os,json
from util.tools  import crop_patches
from configs  import get_config
if __name__ == '__main__':
    args=get_config()
    with open(os.path.join(args.data_path,'annotations.json'),'r') as f:
        data_dict=json.load(f)
    os.makedirs(os.path.join(args.data_path,'stage_sentence'),exist_ok=True)
    for image_name in data_dict:
        data=data_dict[image_name]
        if 'ridge_seg'in data:
            save_path=os.path.join(args.data_path,'stage_sentence',image_name)
            crop_patches(
                image_path=data['image_path'],
                point_list=data['ridge_seg']['point_list'],
                resize_height=224,
                word_number=5,
                patch_size=16,
                save_path=save_path
            )
            data_dict[image_name]['stage_sentence_path']=save_path
    with open(os.path.join(args.data_path,'annotations.json'),'r') as f:
        json.dump(data_dict,f)