import torch
from PIL import Image
from  models import build_model
import os,json
import numpy as np
from configs import get_config
from models import build_model
from sklearn.metrics import accuracy_score, roc_auc_score,recall_score
from util.tools import visual_sentences,crop_patches
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
model= build_model(args.configs['model'])# as we are loading the exite
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
with open(os.path.join(args.data_path,'split',f"clr_{args.split_name}.json"),'r') as f:
    split_all_list=json.load(f)['test']
split_list=[]
for image_name in split_all_list:
    # if data_dict[image_name]['stage']>0:
    #     split_list.append(image_name)
    split_list.append(image_name)
os.makedirs("./experiments/mistake/",exist_ok=True)
os.system(f"rm -rf ./experiments/mistake/*")
for i in ["0","1","2","3"]:
    os.makedirs("./experiments/mistake/"+i,exist_ok=True)
os.makedirs("./experiments/right/",exist_ok=True)
os.system(f"rm -rf ./experiments/right/*")
for i in ["0","1","2","3"]:
    os.makedirs("./experiments/right/"+i,exist_ok=True)
img_norm=transforms.Compose([
            transforms.Resize((args.resize,args.resize)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_DEFAULT_MEAN,std=IMAGENET_DEFAULT_STD)])
probs_list=[]
labels_list=[]
pred_list=[]   

model_prediction_path = './model_prediction.json'
if os.path.exists(model_prediction_path):
    with open(model_prediction_path, 'r') as f:
        model_prediction = json.load(f)
else:
    model_prediction = {}
visual_mistake=False
visual_patch_size= 200
save_visual_global =True
global_path=os.path.join(args.data_path,'visual_stage')
os.makedirs(global_path,exist_ok=True)
with torch.no_grad():
    for image_name in split_list:
        data=data_dict[image_name]
        label=int(data['stage'])
        img=Image.open(data["image_path"]).convert("RGB")
        inputs=[]
        if data['ridge_seg']["max_val"]<0.5:
            bc_prob=np.zeros((1,4),dtype=float)
            bc_prob[0,0]=1.
            bc_pred=0
        else:
            sample_visual=[]
            for (x,y),val in zip(data['ridge_seg']['point_list'],data['ridge_seg']["value_list"]):
                
                sample_visual.append([x,y])
                _,patch=crop_patches(img,args.patch_size,x,y,
                                     abnormal_mask=None,stage=0,save_dir=None)
                patch=img_norm(patch)
                inputs.append(patch.unsqueeze(0))
                if val<args.sample_low_threshold:
                    break
            if len(inputs)<=0:
                print(f"{image_name} do not have enougph data, value list is ",data['ridge_seg']["value_list"] )
                continue
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
            bc_prob[-1]=1-torch.sum(bc_prob[:-1])
            bc_prob=bc_prob.unsqueeze(0).numpy()
            bc_prob = np.insert(bc_prob, 0, 0, axis=1)
            
            bc_pred+=1
            # visual the mismatch version
            if save_visual_global or visual_mistake:
                # Get top k firmest predictions for bc_pred class
                top_k = min(args.k,matching_indices.shape[0])  # Assuming args.k is defined and valid
                class_probs = probs[:, bc_pred-1]  # Extract probabilities for bc_pred class
                top_k_values, top_k_indices = torch.topk(class_probs, k=top_k)
                visual_point=[]
                visual_confidence=[]
                for val,idx in zip(top_k_values,top_k_indices):
                    x,y=data['ridge_seg']['point_list'][idx]
                    visual_point.append([int(x),int(y)])
                    visual_confidence.append(round(float(val),2))
                    
                if visual_mistake:
                    if label!=bc_pred:
                        save_path=os.path.join('./experiments/mistake/',str(label),image_name)
                    else:
                        continue # mistake only
                        save_path=os.path.join('./experiments/right/',str(label),image_name)
                    
                    visual_sentences(
                    data_dict[image_name]['image_path'],
                    points=visual_point,
                    patch_size=visual_patch_size,
                    text=f"label: {label}",
                    confidences=visual_confidence,
                    label=bc_pred,
                    save_path=save_path,
                    sample_visual=sample_visual
                    )
                if save_visual_global:
                    save_global_path=os.path.join(global_path,image_name)
                    visual_sentences(
                    data_dict[image_name]['image_path'],
                    points=visual_point,
                    patch_size=visual_patch_size,
                    # text=f"label: {label}",
                    confidences=visual_confidence,
                    label=bc_pred,
                    save_path=save_global_path,
                    sample_visual=sample_visual
                    )
                    data_dict[image_name]['visual_stage_path']=save_global_path
        probs_list.extend(bc_prob)
        labels_list.append(label)
        pred_list.append(bc_pred)
        model_prediction[image_name]=bc_pred
probs_list=np.vstack(probs_list)
pred_labels=np.array(pred_list)
labels_list=np.array(labels_list)

accuracy=accuracy_score(labels_list,pred_list)
auc=roc_auc_score(labels_list,probs_list, multi_class='ovo')
print(f"acc: {accuracy:.4f}, auc: {auc:.4f}")

# with open(model_prediction_path, 'w') as f:
#     json.dump(model_prediction,f)
with open(os.path.join(args.data_path,'annotations.json'),'w') as f:
    json.dump(data_dict,f)

# Assuming probs_list has shape (num_samples, num_classes) and pred_labels, labels_list are 1D arrays
num_classes = probs_list.shape[1]
recall_per_class = np.zeros(num_classes)

# Calculate recall for each class
for i in range(num_classes):
    true_class = labels_list == i
    predicted_class = pred_labels == i
    recall_per_class[i] = recall_score(true_class, predicted_class)

# Calculate recall for positive classes (classes > 0)
true_positive = labels_list > 0
predicted_positive = pred_labels > 0
recall_positive = recall_score(true_positive, predicted_positive)

# Print recall for each class and positive recall
for i, recall in enumerate(recall_per_class):
    print(f"Recall for class {i}: {recall:.4f}")
print(f"Recall for positive classes: {recall_positive:.4f}")