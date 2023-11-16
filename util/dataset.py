
import os
from torchvision import  transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import Dataset
from PIL import Image
import json
class CustomDataset(Dataset):
    def __init__(self, split,data_path,split_name):
        with open(os.path.join(data_path,'split',f'{split_name}.json'), 'r') as f:
            self.split_list=json.load(f)[split]
        with open(os.path.join(data_path,'annotations.json'),'r') as f:
            self.data_dict=json.load(f)
        
        self.split = split
        self.img_transforms=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_DEFAULT_MEAN,
                std=IMAGENET_DEFAULT_STD)])
    def __getitem__(self, idx):
        data = self.split_list[idx]
        img = Image.open(data['stage_sentence_path']).convert('RGB')
        
        img = self.preprocess(img)
            
        # Convert mask and pos_embed to tensor
        img = self.img_transformss(img)
        
        return img,data['stage']


    def __len__(self):
        return len(self.split_list)