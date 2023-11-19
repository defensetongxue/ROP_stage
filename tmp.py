import json,os
data_path='../autodl-tmp/dataset_ROP'

with open(os.path.join(data_path,'annotations.json'),'r') as f:
    data_dict=json.load(f)

for split_name in ["1","2","3","4"]:
    with open(os.path.join(data_path,'split',f'{split_name}.json'),'r') as f:
        split_list=json.load(f)
    for split  in ['train','val','test']:
        condition_cnt={i:0 for i in range(4)}
        for image_name in split_list[split]:
            condition_cnt[
                data_dict[image_name]['stage']
            ]+=1
        print(f"Split_name: {split_name}, {split}, {condition_cnt}")