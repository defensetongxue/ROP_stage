
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
    def __init__(self, split,data_path,split_name,img_resize=256):
        with open(os.path.join(data_path,'split',f'{split_name}.json'), 'r') as f:
            ori_split_list=json.load(f)[split]
        with open(os.path.join(data_path,'annotations.json'),'r') as f:
            self.data_dict=json.load(f)
        self.split_list=[]
        for image_name in ori_split_list:
            data=self.data_dict[image_name]
            # buid patch image
            if self.data_dict[image_name]['stage']==0:
                continue
            if not 'stage_sentence_path' in data:
                continue
            word_list=os.listdir(data['stage_sentence_path'])
            for image_cnt in word_list:
                self.split_list.append((os.path.join(data['stage_sentence_path'],image_cnt),image_name))
        
        self.patch_enhance= transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation((0, 360)),
        ])
        self.img_resize=transforms.Resize((img_resize,img_resize))
        self.split = split
        self.img_norm=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_DEFAULT_MEAN,std=IMAGENET_DEFAULT_STD)])
    def __getitem__(self, idx):
        image_path,image_name = self.split_list[idx]
        img=Image.open(image_path).convert("RGB")
        img=self.img_resize(img)
        if self.split=='train':
            img=self.patch_enhance(img)
        img=self.img_norm(img)
        
        # save_path=os.path.join(save_dir,f"{str(patch_stage)}_{image_cnt}.jpg")
        label = int(os.path.basename(image_path)[0])-1
        return img,label,image_name


    def __len__(self):
        return len(self.split_list)
    
class BaselineDataset(Dataset):
    def __init__(self, split,data_path,split_name,img_resize=256):
        with open(os.path.join(data_path,'split',f'{split_name}.json'), 'r') as f:
            ori_split_list=json.load(f)[split]
        with open(os.path.join(data_path,'annotations.json'),'r') as f:
            self.data_dict=json.load(f)
        self.split_list=[]
        for image_name in ori_split_list:
            data=self.data_dict[image_name]
            # buid patch image
            if self.data_dict[image_name]['stage']==0:
                continue
            if not 'stage_sentence_path' in data:
                continue
            self.split_list.append(image_name)
        if split=='test':
            with open(f'./stage_split/clr_{split_name}.json') as f:
                self.split_list=json.load(f)['test']
        self.patch_enhance= transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            # transforms.RandomRotation((0, 360)),
            Fix_RandomRotation(),
        ])
        self.img_resize=transforms.Resize((img_resize,img_resize))
        self.split = split
        self.img_norm=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_DEFAULT_MEAN,std=IMAGENET_DEFAULT_STD)])
    def __getitem__(self, idx):
        image_name = self.split_list[idx]
        data=self.data_dict[image_name]
        img=Image.open(data["image_path"]).convert("RGB")
        img=self.img_resize(img)
        if self.split=='train':
            img=self.patch_enhance(img)
        img=self.img_norm(img)
        label=data['stage']-1
        assert label<3
        # save_path=os.path.join(save_dir,f"{str(patch_stage)}_{image_cnt}.jpg")
        return img,label,image_name


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
    
class TestPatchDataset(Dataset):
    def __init__(self, record_path,img_resize=256):
        with open(record_path, 'r') as f:
            self.ori_split_list=json.load(f)
      
        
        self.img_norm=transforms.Compose([
            transforms.Resize((img_resize,img_resize)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_DEFAULT_MEAN,std=IMAGENET_DEFAULT_STD)])
    def __getitem__(self, idx):
        patch_record = self.ori_split_list[idx]
        img=Image.open(patch_record['patch_path']).convert("RGB")
        img=self.img_norm(img)
        
        return img,patch_record['image_name']


    def __len__(self):
        return len(self.ori_split_list)