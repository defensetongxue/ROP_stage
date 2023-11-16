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
            save_path=os.path.join(args.data_path,'stage_sentence',image_name[:-4])
            os.makedirs(save_path,exist_ok=True)
            crop_patches(
                image_path=data['image_path'],
                point_list=data['ridge_seg']['point_list'],
                word_number=5,
                patch_size=112,
                save_path=save_path
            )
            data_dict[image_name]['stage_sentence_path']=save_path
    with open(os.path.join(args.data_path,'annotations.json'),'w') as f:
        json.dump(data_dict,f)