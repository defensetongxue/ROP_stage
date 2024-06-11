import os,json
from PIL import Image, ImageOps, ImageDraw, ImageFont
def crop_patches(img,patch_size,x,y,max_weight=1599,max_height=1199, save_path=None):
    '''
    keep the size as conv ridge segmentation model
    '''
    left  = int(x - patch_size // 2)
    upper = int(y - patch_size // 2)
    right = int(x + patch_size // 2)
    lower = int(y + patch_size // 2)
    
    # Pad if necessary
    padding = [max(0, -left), max(0, -upper), max(0, right - max_weight), max(0, lower - max_height)]
    left=max(0, left)
    upper=max(0,upper)
    right=min(max_weight, right)
    lower= min(max_height, lower)
    # Crop the patch
    patch = img.crop((left, upper, right, lower))
    patch = ImageOps.expand(patch, tuple(padding), fill=255) 
    
    res = Image.new('RGB', (patch_size, patch_size), (0, 0, 0))
    # Create a circular mask for the inscribed circle
    mask = Image.new('L', (patch_size, patch_size), 0)
    draw = ImageDraw.Draw(mask)
    radius = patch_size // 2
    center = (radius, radius)
    draw.ellipse((center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius), fill=255)
    res.paste(patch, (0, 0), mask=mask)
    
    
    if save_path:
        res.save(save_path)
        
data_path='../autodl-tmp/dataset_ROP'
with open(os.path.join(data_path,'split','stage_1.json'),'r') as f:
    stage_split=json.load(f)['test']
with open(os.path.join(data_path, 'annotations.json'), 'r') as f:
    data_dict = json.load(f)
os.makedirs(os.path.join(data_path,'patch_clr_1','images'),exist_ok=True)
patch_cnt=0
record_dict=[]
for image_name in stage_split:
    data=data_dict[image_name]
    for (x,y),val in zip(data['ridge_seg']['point_list'],data['ridge_seg']["value_list"]):
        img=Image.open(data["image_path"]).convert("RGB")
        save_path=os.path.join(data_path,'patch_clr_1','images',f"{patch_cnt}.jpg")
        patch_cnt+=1
        crop_patches(img,400,x,y,save_path=save_path)
        record_dict.append(
            {
                "image_name":image_name,
                "patch_path":save_path
            }
        )
        if val<0.45:
            break
print(patch_cnt)
with open(os.path.join(data_path,'patch_clr_1','annotations.json'),'w') as f:
    json.dump(record_dict,f)