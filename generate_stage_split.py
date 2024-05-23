import json,os
data_path='../autodl-tmp/dataset_ROP'
with open(os.path.join(data_path,'annotations.json')) as f:
    data_dict=json.load(f)
with open(os.path.join(data_path,'split','clr_1.json')) as f:
    split_list=json.load(f)
stage_split={}
for split in split_list:
    stage_split[split]=[]
    for image_name in split_list[split]:
        data=data_dict[image_name]
        if split !='test':
            if 'stage_sentence_path' not in data or data['stage']==0:
                continue
        else:
            if data['stage']==0 or data['ridge_seg']['max_val']<0.5:
                continue
            
        stage_split[split].append(image_name)
    print(len(stage_split[split]))
with open(os.path.join(data_path,'split','stage_1.json'),'w') as f:
    json.dump(stage_split,f)