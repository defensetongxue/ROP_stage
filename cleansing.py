import os,json
from util.tools  import crop_patches
from configs  import get_config
if __name__ == '__main__':
    args=get_config()
    with open(os.path.join(args.data_path,'annotations.json'),'r') as f:
        data_dict=json.load(f)
    os.makedirs(os.path.join(args.data_path,'stage_sentence'),exist_ok=True)
    os.system(f"rm -rf {os.path.join(args.data_path,'stage_sentence')}/*")
    
    max_sample_number=args.max_sample_number
    unbalance_sample={
        1:1,
        2:1,
        3:0.5
    }
    
    for image_name in data_dict:
        data=data_dict[image_name]
        if data['stage']<=0:
            continue
        if 'ridge_diffusion_path' in data:
            sample_dense=int(args.sample_dense*unbalance_sample[data['stage']])
            
    with open(os.path.join(args.data_path,'annotations.json'),'w') as f:
        json.dump(data_dict,f)