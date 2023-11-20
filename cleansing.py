import os,json
from util.tools  import crop_patches
from configs  import get_config
if __name__ == '__main__':
    args=get_config()
    with open(os.path.join(args.data_path,'annotations.json'),'r') as f:
        data_dict=json.load(f)
    os.makedirs(os.path.join(args.data_path,'stage_sentence'),exist_ok=True)
    os.system(f"rm -rf {os.path.join(args.data_path,'stage_sentence')}/*")
    for image_name in data_dict:
        data=data_dict[image_name]
        if 'ridge_seg'in data:
            save_path=os.path.join(args.data_path,'stage_sentence',image_name[:-4])
            os.makedirs(save_path,exist_ok=True)
            ridge_diffusion_path=data['ridge_diffusion_path'] if 'ridge_diffusion_path' in data else None
            abnormal_cooridinate=data['ridge']["vessel_abnormal_coordinate"] if \
            'ridge' in data  and \
            data['ridge']["vessel_abnormal_number"]>0 \
                else None
            stage_list=crop_patches(
                image_path=data['image_path'],
                point_list=data['ridge_seg']['point_list'],
                value_list=data['ridge_seg']['value_list'],
                ridge_diffusion_path=ridge_diffusion_path,
                abnormal_cooridinate=abnormal_cooridinate,
                stage=data['stage'],
                word_number=args.word_size,
                patch_size=args.patch_size,
                save_path=save_path
            )
            data_dict[image_name]['stage_sentence_stagelist']=stage_list
            data_dict[image_name]['stage_sentence_path']=save_path
    with open(os.path.join(args.data_path,'annotations.json'),'w') as f:
        json.dump(data_dict,f)