import os,json
from util.tools  import crop_patches,sample_patch
from configs  import get_config
from PIL import Image
import numpy as np
if __name__ == '__main__':
    args=get_config()
    with open(os.path.join(args.data_path,'annotations.json'),'r') as f:
        data_dict=json.load(f)
    os.makedirs(os.path.join(args.data_path,'stage_sentence'),exist_ok=True)
    os.system(f"rm -rf {os.path.join(args.data_path,'stage_sentence')}/*")
    
    unbalance_sample={
        1:1,
        2:1,
        3:0.5
    }
    
    for image_name in data_dict:
        data=data_dict[image_name]
        if 'stage_sentence_path' in data:
            del data_dict[image_name]['stage_sentence_path']
        if data['stage']<=0:
            continue
        if 'ridge_diffusion_path' in data:
            sample_dense=int(args.sample_dense*unbalance_sample[data['stage']])
            if data['stage'] ==3:
                max_number=100 
                # build abnormal_mask
                abnormal_mask=np.zeros((1200,1600))
                if not len(data['ridge']["vessel_abnormal_coordinate"])>0:
                    print(image_name)
                    continue
                for x,y in data['ridge']["vessel_abnormal_coordinate"]:
                    abnormal_mask[int(y),int(x)]=1
                
            else:
                max_number=args.max_sample_number
                abnormal_mask=None
            sample_list= sample_patch(data['ridge_diffusion_path'],sample_dense,max_number)
            img=Image.open(data['image_path']).convert("RGB")
            
            save_dir=os.path.join(args.data_path,'stage_sentence',image_name[:-4])
            os.makedirs(save_dir,exist_ok=True)
            stage_list=[]
            
            rm_idx=[]
            enhanced=False
            enhanced_distance=50
            for cnt,(x,y) in enumerate(sample_list):
                patch_stage,_=crop_patches(img,args.patch_size,x,y,abnormal_mask,data['stage'],
                            save_dir=save_dir,image_cnt=str(cnt))
                if patch_stage==-1:
                    rm_idx.append(cnt)
                    continue
                stage_list.append(patch_stage)
                if enhanced:
                    # Generate coordinates for near 8 points around (x, y)
                    offsets = [-enhanced_distance, 0, enhanced_distance]
                    near_points = [(x + dx, y + dy) for dx in offsets for dy in offsets if not (dx == 0 and dy == 0)]

                    for enhanced_cnt, (x_extra, y_extra) in enumerate(near_points):
                        unique_cnt = 100 * cnt + enhanced_cnt  # Unique identifier for each patch
                        patch_stage, _ = crop_patches(img, args.patch_size, x_extra, y_extra, abnormal_mask, data['stage'],
                                                      save_dir=save_dir, image_cnt=str(unique_cnt))
            

            if len(rm_idx) > 0:
                sample_list = [item for cnt, item in enumerate(sample_list) if cnt not in rm_idx]

            data_dict[image_name]['stage_sentence_path']=save_dir
            data_dict[image_name]['stage_sample']=sample_list
            data_dict[image_name]['stage_list']=stage_list


        if data['stage']<=0:
            continue
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