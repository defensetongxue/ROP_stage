from PIL import Image, ImageOps, ImageDraw, ImageFont
import os,json
import numpy as np
from shutil import copy
with open('../autodl-tmp/dataset_ROP/annotations.json','r') as f:
    data_dict=json.load(f)
cnt=0
for image_name in data_dict:
    data=data_dict[image_name]
    if data['stage']==1:
        cnt+=1
print(cnt)