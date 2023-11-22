
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
    def __init__(self, split,data_path,split_name):
        with open(os.path.join(data_path,'split',f'{split_name}.json'), 'r') as f:
            self.split_list=json.load(f)[split]
        with open(os.path.join(data_path,'annotations.json'),'r') as f:
            self.data_dict=json.load(f)
        # new=[]
        # for image_name in self.split_list:
        #     if 'ridge' in self.data_dict[image_name]:
        #         new.append(image_name)
        # self.split_list=new
        self.enhance_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            Fix_RandomRotation(),
        ])
        self.split = split
        self.img_transforms=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_DEFAULT_MEAN,std=IMAGENET_DEFAULT_STD)])
                # mean=[0.4623, 0.3856, 0.2822],std=[0.2527, 0.1889, 0.1334])])
    def __getitem__(self, idx):
        image_name = self.split_list[idx]
        data=self.data_dict[image_name]
        word_list=os.listdir(data['stage_sentence_path'])
        image_list=[]
        for image_cnt in word_list:
            img = Image.open(os.path.join(data['stage_sentence_path'],image_cnt)).convert('RGB')
            if self.split=='train':
                img=self.enhance_transforms(img)
            img = self.img_transforms(img)
            image_list.append(img)
        patches=torch.stack(image_list,dim=0)
        val_tensor=torch.tensor(data['ridge_seg']["value_list"])
        # Convert mask and pos_embed to tensor
        stage_label=data['stage']-1
        stage_list=torch.tensor(data['stage_sentence_stagelist']).long()
        return( patches,val_tensor),(data['stage'],stage_list),image_name
        # return( patches,val_tensor),stage_label,image_name


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