import torch
from PIL import Image
from  models import build_model
import os,json
import numpy as np
from util.functions import to_device
from  util.metric import calculate_recall
from configs import get_config
from sklearn.metrics import accuracy_score, roc_auc_score
from util.tools import visual_sentences,crop_patches
from shutil import copy
from torchvision import transforms
from util.dataset import IMAGENET_DEFAULT_MEAN,IMAGENET_DEFAULT_STD
# Initialize the folder
os.makedirs("checkpoints",exist_ok=True)
os.makedirs("experiments",exist_ok=True)
torch.manual_seed(0)
np.random.seed(0)
# Parse arguments
args = get_config()

os.makedirs(args.save_dir,exist_ok=True)
print("Saveing the model in {}".format(args.save_dir))
# Create the model and criterion
model= build_model(num_classes=args.configs["num_classes"])# as we are loading the exite
# model.load_pretrained(pretrained_path=args.configs["pretrained_path"])

# Set up the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(f"using {device} for training")

save_model_name=args.split_name+args.configs['save_name']
print(os.path.join(args.save_dir, save_model_name))
model.load_state_dict(
    torch.load(os.path.join(args.save_dir, save_model_name)))
model.eval()

all_predictions = []
all_targets = []
probs_list = []
with open(os.path.join(args.data_path,'annotations.json'),'r') as f:
    data_dict=json.load(f)
with open(os.path.join(args.data_path,'split',f"{args.split_name}.json"),'r') as f:
    split_all_list=json.load(f)['test']
split_list=[]
for image_name in split_all_list:
    if data_dict[image_name]['stage']>0:
        split_list.append(image_name)

os.makedirs("./experiments/visual/",exist_ok=True)
os.system(f"rm -rf ./experiments/visual/*")
for i in ["1","2","3"]:
    os.makedirs("./experiments/visual/"+i,exist_ok=True)
img_norm=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_DEFAULT_MEAN,std=IMAGENET_DEFAULT_STD)])
probs_list=[]
labels_list=[]
pred_list=[]   
with torch.no_grad():
    for image_name in split_list:
        data=data_dict[image_name]
        label=int(data['stage'])-1
        img=Image.open(data["image_path"]).convert("RGB")
        inputs=[]
        for x,y in data['ridge_seg']['point_list']:
            _,patch=crop_patches(img,args.patch_size,x*4,y*4,
                                 abnormal_mask=None,stage=0,save_dir=None)
            patch=img_norm(patch)
            inputs.append(patch.unsqueeze(0))
        inputs=torch.cat(inputs,dim=0)
        
        outputs=model(inputs.to(device))
        probs = torch.softmax(outputs.cpu(), axis=1)
        # output shape is bc,num_class
        # get pred for each patch
        pred_labels = torch.argmax(probs, dim=1)
        # get the max predict  label for this batch ( as  bc_pred)
        bc_pred= int(torch.max(pred_labels))
        # select the patch whose preds_label is equal to bc_pred
        matching_indices = torch.where(pred_labels == bc_pred)[0]
        selected_probs = probs[matching_indices]
        # mean these selectes patches probs as bc_porb
        bc_prob = torch.mean(selected_probs, dim=0)
        
        probs_list.append(bc_prob)
        labels_list.append(label)
        pred_list.append(bc_pred)
        if label!=bc_pred:
            # Get top k firmest predictions for bc_pred class
            top_k = min(args.k,matching_indices.shape[0])  # Assuming args.k is defined and valid
            class_probs = probs[:, bc_pred]  # Extract probabilities for bc_pred class
            top_k_values, top_k_indices = torch.topk(class_probs, k=top_k)
            print(matching_indices.shape[0])
            visual_point=[]
            visual_confidence=[]
            for val,idx in zip(top_k_values,top_k_indices):
                x,y=data['ridge_seg']['point_list'][idx]
                visual_point.append([x*4,y*4])
                visual_confidence.append(round(float(val),2))
            visual_sentences(
                data_dict[image_name]['image_path'],
                points=visual_point,
                patch_size=224,
                text=f"label: {label+1}",
                confidences=visual_confidence,
                label=bc_pred+1,
                save_path=os.path.join('./experiments/visual/',str(label+1),image_name)
            )