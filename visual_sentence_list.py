import json,os
from PIL import Image, ImageDraw, ImageFont
data_path='../autodl-tmp/dataset_ROP'

with open(os.path.join(data_path,'annotations.json'),'r') as f:
    data_dict=json.load(f)
with open(os.path.join(data_path,'split','2.json'),'r') as f:
    split_list=json.load(f)['train']


def visual_stage3(image_path, point_list, stage_list, save_path, patch_size=112,orignal_annotation=[]):
    img = Image.open(image_path).convert('RGB').resize((800, 600))
    draw = ImageDraw.Draw(img)
    
    # Load Arial font
    font_path = './arial.ttf'
    font_size = 25  # Set your desired font size
    font = ImageFont.truetype(font_path, font_size)
    for cnt, (y, x) in enumerate(point_list):  # x is for width, y is for height
        stage = stage_list[cnt]
        left, upper = x - patch_size // 2, y - patch_size // 2
        right, lower = x + patch_size // 2, y + patch_size // 2

        if stage == 3:
            # Draw red box
            draw.rectangle([left, upper, right, lower], outline="red", width=2)
        elif stage == 2:
            # Draw yellow box
            draw.rectangle([left, upper, right, lower], outline="yellow", width=2)
        elif stage == 1:
            # Draw green box
            draw.rectangle([left, upper, right, lower], outline="green", width=2)
        else:
            # Draw point and number if stage is 0
            draw.point((x, y), fill="blue")
            draw.text((x, y), str(cnt), fill="blue", font=font)

    # Ensure the save path directory exists
    assert len(orignal_annotation)>0
    for x,y in orignal_annotation:
        x=int(x/2)
        y=int(y/2)
        draw.text((x, y), 'x', fill="blue", font=font)
    # Save the image
    img.save(save_path)

# Example usage
for image_name in data_dict:

    data = data_dict[image_name]
    if data['stage']<3:
        continue
    point_list = data['ridge_seg']['point_list']
    stage_list = data['stage_sentence_stagelist']
    save_path = './experiments/stage3/'+image_name
    image_path = data['image_path']
    visual_stage3(
        image_path=image_path,
        point_list=point_list,stage_list= stage_list,save_path= save_path,orignal_annotation= data['ridge']["vessel_abnormal_coordinate"])