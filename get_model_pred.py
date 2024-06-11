import torch
from PIL import Image
from models import build_model
import os
import json
import numpy as np
from configs import get_config
from models import build_model
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score
from util.tools import visual_sentences, crop_patches
from torchvision import transforms
from util.dataset import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
# Initialize the folder
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("experiments", exist_ok=True)
torch.manual_seed(0)
np.random.seed(0)
# Parse arguments
args = get_config()

os.makedirs(args.save_dir, exist_ok=True)
print("Saveing the model in {}".format(args.save_dir))
# Create the model and criterion
model = build_model(args.configs['model'])  # as we are loading the exite
# model.load_pretrained(pretrained_path=args.configs["pretrained_path"])

# Set up the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(f"using {device} for training")

save_model_name = args.split_name+args.configs['save_name']
print(os.path.join(args.save_dir, save_model_name))
model.load_state_dict(
    torch.load(os.path.join(args.save_dir, save_model_name)))
model.eval()

all_predictions = []
all_targets = []
probs_list = []
with open(os.path.join(args.data_path, 'annotations.json'), 'r') as f:
    data_dict = json.load(f)
with open(os.path.join(args.data_path, 'split', f"clr_{args.split_name}.json"), 'r') as f:
    split_all_list = json.load(f)['test']
split_list = []
for image_name in split_all_list:
    split_list.append(image_name)
img_norm = transforms.Compose([
    transforms.Resize((args.resize, args.resize)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)])
probs_list = []
labels_list = []
pred_list = []

model_prediction_path = './model_prediction.json'
if os.path.exists(model_prediction_path):
    with open(model_prediction_path, 'r') as f:
        model_prediction = json.load(f)
else:
    model_prediction = {}
visual_mistake = False
visual_patch_size = 200
save_visual_global = True
global_path = os.path.join(args.data_path, 'visual_stage')
os.makedirs(global_path, exist_ok=True)
with torch.no_grad():
    for image_name in split_list:
        data = data_dict[image_name]
        label = int(data['stage'])
        img = Image.open(data["image_path"]).convert("RGB")
        inputs = []
        if data['ridge_seg']["max_val"] < 0.5:
            bc_prob = np.zeros((1, 4), dtype=float)
            bc_prob[0, 0] = 1.
            bc_pred = 0
        else:
            sample_visual = []
            for (x, y), val in zip(data['ridge_seg']['point_list'], data['ridge_seg']["value_list"]):

                sample_visual.append([x, y])
                _, patch = crop_patches(img, args.patch_size, x, y,
                                        abnormal_mask=None, stage=0, save_dir=None)
                patch = img_norm(patch)
                inputs.append(patch.unsqueeze(0))
                if val < args.sample_low_threshold:
                    break
            if len(inputs) <= 0:
                print(f"{image_name} do not have enougph data, value list is ",
                      data['ridge_seg']["value_list"])
                continue
            inputs = torch.cat(inputs, dim=0)

            outputs = model(inputs.to(device))
            probs = torch.softmax(outputs.cpu(), axis=1)
            # output shape is bc,num_class
            # get pred for each patch
            pred_labels = torch.argmax(probs, dim=1)
            # get the max predict  label for this batch ( as  bc_pred)
            bc_pred = int(torch.max(pred_labels))
            # select the patch whose preds_label is equal to bc_pred
            matching_indices = torch.where(pred_labels == bc_pred)[0]
            selected_probs = probs[matching_indices]
            # mean these selectes patches probs as bc_porb
            bc_prob = torch.mean(selected_probs, dim=0)
            bc_prob[-1] = 1-torch.sum(bc_prob[:-1])
            bc_prob = bc_prob.unsqueeze(0).numpy()
            bc_prob = np.insert(bc_prob, 0, 0, axis=1)

            bc_pred += 1
        model_prediction[image_name]=bc_pred
        probs_list.extend(bc_prob)
        labels_list.append(label)
        pred_list.append(bc_pred)
        model_prediction[image_name] = bc_pred
probs_list = np.vstack(probs_list)
pred_labels = np.array(pred_list)
labels_list = np.array(labels_list)

with open(model_prediction_path, 'w') as f:
    json.dump(model_prediction,f)