import os
import json
from PIL import Image
import torch
import numpy as np
from configs import get_config
from models import build_model
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score
from util.tools import visual_sentences, crop_patches
from torchvision import transforms
from util.dataset import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
args=get_config()
# Initialize the folders
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("experiments", exist_ok=True)
os.makedirs(args.save_dir, exist_ok=True)
os.makedirs("./experiments/stage_only/", exist_ok=True)
os.system(f"rm -rf ./experiments/stage_only/*")
for i in ["1","2","3"]:
    os.makedirs("./experiments/stage_only/"+i,exist_ok=True)
    
# Set seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)

# Parse arguments
args = get_config()
print(f"Saving the model in {args.save_dir}")

# Create the model
model = build_model(args.configs['model'])
# model.load_pretrained(pretrained_path=args.configs["pretrained_path"])

# Set up the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(f"Using {device} for training")

# Load the model
save_model_name = args.split_name + args.configs['save_name']
model.load_state_dict(torch.load(os.path.join(args.save_dir, save_model_name)))
model.eval()

# Load annotations and split list
with open(os.path.join(args.data_path, 'annotations.json'), 'r') as f:
    data_dict = json.load(f)
with open(os.path.join('./stage_split', f"clr_{args.split_name}.json"), 'r') as f:
    split_all_list = json.load(f)['test']

# Filter images
split_list = [image_name for image_name in split_all_list if data_dict[image_name]['stage'] > 0]

# Image normalization
img_norm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
])

# Processing images
probs_list, labels_list, pred_list = [], [], []
visual = False

with torch.no_grad():
    for image_name in split_list:
        data=data_dict[image_name]
        label=int(data['stage'])-1
        img=Image.open(data["image_path"]).convert("RGB")
        inputs=[]
        if 'point_list' not in data['ridge_seg']:
            continue
        
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
            print(inputs)
            print(image_name)
            print(data['ridge_seg']["value_list"])
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
        assert label<=2
        # visual the mismatch version
        if label!=bc_pred and visual:
            # Get top k firmest predictions for bc_pred class
            top_k = min(args.k,matching_indices.shape[0])  # Assuming args.k is defined and valid
            class_probs = probs[:, bc_pred]  # Extract probabilities for bc_pred class
            top_k_values, top_k_indices = torch.topk(class_probs, k=top_k)
            visual_point=[]
            visual_confidence=[]
            for val,idx in zip(top_k_values,top_k_indices):
                x,y=data['ridge_seg']['point_list'][idx]
                visual_point.append([int(x),int(y)])
                visual_confidence.append(round(float(val),2))
            
            visual_sentences(
                data_dict[image_name]['image_path'],
                points=visual_point,
                patch_size=args.patch_size,
                text=f"label: {label}",
                confidences=visual_confidence,
                label=bc_pred+1,
                save_path=os.path.join('./experiments/stage_only/',str(label+1),image_name),
                sample_visual=sample_visual
            )
        probs_list.extend(bc_prob)
        labels_list.append(label)
        pred_list.append(bc_pred)
        
probs_list=np.vstack(probs_list)
pred_labels=np.array(pred_list)
labels_list=np.array(labels_list)

# Performance metrics
accuracy = accuracy_score(labels_list, pred_list)
auc = roc_auc_score(labels_list, probs_list, multi_class='ovo')

# Calculate recall for each class
num_classes = probs_list.shape[1]
recall_per_class = np.zeros(num_classes)
for i in range(num_classes):
    true_class = labels_list == i
    predicted_class = pred_labels == i
    recall_per_class[i] = recall_score(true_class, predicted_class)

print(f"acc: {accuracy:.4f}, auc: {auc:.4f}")
for i, recall in enumerate(recall_per_class):
    print(f"Recall for class {i}: {recall:.4f}")


# Record results
record_path = './experiments/record_stage.json'
key = f"{args.ridge_seg_number}_{args.sample_distance}_{int(100 * args.sample_low_threshold)}"
content = {
    "auc": auc,
    "acc": accuracy,
    "recall_per_class": recall_per_class.tolist()  # Convert numpy array to list for JSON serialization
}

# Check if record file exists and load it, otherwise create a new record
if not os.path.exists(record_path):
    record = {}
else:
    with open(record_path, 'r') as f:
        record = json.load(f)

# Update the record with new results
record[key] = content

# Save the updated record
with open(record_path, 'w') as f:
    json.dump(record, f, indent=4)

print("Results saved in record_stage.json.")
