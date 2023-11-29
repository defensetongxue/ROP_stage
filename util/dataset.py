
import os
from torchvision import  transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from PIL import Image
import json
import torch
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
class CustomDataset(Dataset):
    def __init__(self, split,data_path,split_name,img_resize=299):
        with open(os.path.join(data_path,'split',f'{split_name}.json'), 'r') as f:
            self.split_list=json.load(f)[split]
        with open(os.path.join(data_path,'annotations.json'),'r') as f:
            self.data_dict=json.load(f)
        self.img_resize=transforms.Resize((img_resize,img_resize))
        self.enhance_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            Fix_RandomRotation(),
        ])
        
        self.patch_enhance= transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation((0, 360)),
        ])
        self.split = split
        self.img_norm=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_DEFAULT_MEAN,std=IMAGENET_DEFAULT_STD)])
    def __getitem__(self, idx):
        image_name = self.split_list[idx]
        data=self.data_dict[image_name]
             
        # buid patch image
        word_list=os.listdir(data['stage_sentence_path'])
        patch_list=[]
        for image_cnt in word_list:
            patch = Image.open(os.path.join(data['stage_sentence_path'],image_cnt)).convert('RGB')
            if self.split=='train':
                patch=self.patch_enhance(patch)
            patch= self.img_norm(patch)
            patch_list.append(patch)
        patches=torch.stack(patch_list,dim=0)

        # build label
        stage_list=torch.tensor(data['stage_sentence_stagelist']).long()
        if self.split == 'train':
            return patches,stage_list,image_name
        else:
            return  patches,data['stage'],image_name


    def __len__(self):
        return len(self.split_list)
    
class Fix_RandomRotation(object):

    def __init__(self, degrees=360, resample=False, expand=False, center=None):
        self.degrees = degrees
        self.resample = resample
        self.expand = expand
        self.center = center

    @staticmethod
    def get_params():
        p = torch.rand(1)

        if p >= 0 and p < 0.25:
            angle = -180
        elif p >= 0.25 and p < 0.5:
            angle = -90
        elif p >= 0.5 and p < 0.75:
            angle = 90
        else:
            angle = 0
        return angle

    def __call__(self, img):
        angle = self.get_params()
        return F.rotate(img, angle, F.InterpolationMode.NEAREST , self.expand, self.center)

    def __repr__(self):
        format_string = self.__class__.__name__ + \
            '(degrees={0}'.format(self.degrees)
        format_string += ', resample={0}'.format(self.resample)
        format_string += ', expand={0}'.format(self.expand)
        if self.center is not None:
            format_string += ', center={0}'.format(self.center)
        format_string += ')'
        return format_string