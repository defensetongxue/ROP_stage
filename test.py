import torch
from torch.utils.data import DataLoader
from util.dataset import CustomDataset
from  models import build_model
import os,json
import numpy as np
from util.functions import to_device
from  util.metric import calculate_recall
from configs import get_config
from sklearn.metrics import accuracy_score, roc_auc_score
from util.tools import visual_sentence
from shutil import copy
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
model.load_pretrained(pretrained_path=args.configs["pretrained_path"])

# Set up the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(f"using {device} for training")
model.eval()


# Load the datasets
test_dataset=CustomDataset(split='test',data_path=args.data_path,split_name=args.split_name)
# Create the data loaders
test_loader=  DataLoader(test_dataset,
                        batch_size=args.configs['train']['batch_size'],
                        shuffle=True, num_workers=args.configs['num_works'])

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
cnt={}
os.makedirs("./experiments/visual/",exist_ok=True)
for i in ["0","1","2","3"]:
    os.makedirs("./experiments/visual/"+i,exist_ok=True)
    for j in ["True","False"]:
        os.makedirs("./experiments/visual/"+i+'/'+j,exist_ok=True)
cnt=50
with torch.no_grad():
    for inputs, targets, image_names in test_loader:
        inputs = to_device(inputs, device)

        outputs,att_order = model._visual(inputs)
        att_order=att_order.detach().cpu()
        # Convert model outputs to probabilities using softmax
        probs = torch.softmax(outputs.cpu(), axis=1).numpy()

        # Use argmax to get predictions from probabilities
        predictions = np.argmax(probs, axis=1)
        for preds,prob,orders,image_name,label in zip(predictions,probs,att_order,image_names,targets[0]):
            label=int(label)
            
            if label==0:
                if preds ==0 :
                    if cnt<=0:
                        continue
                    cnt-=1
                    copy(data_dict[image_name]['image_path'],
                        os.path.join('./experiments/visual/',str(label),"True",image_name)
                    )
                    # print(os.path.join('./experiments/visual/',str(label),"True",image_name))
                    
                else:
                    select=int(orders[0])
                    confidence=prob[preds]
                    point=data_dict[image_name]['ridge_seg']['point_list'][select]
                    visual_sentence(
                        data_dict[image_name]['image_path'],
                        x=point[1],y=point[0],
                        patch_size=112,
                        # text=f"label: {label}",
                        confidence=confidence,
                        label=preds,
                        save_path=os.path.join('./experiments/visual/',str(label),"False",image_name)
                    )
                    raise
                    # print(os.path.join('./experiments/visual/',str(label),"False",image_name))
                continue
            # label  != 0
            select=int(orders[0])
            confidence=prob[preds]
            # print(confidence)
            point=data_dict[image_name]['ridge_seg']['point_list'][select]
            if preds==label:
                suc="True"
            else:
                suc="False"
            visual_sentence(
                data_dict[image_name]['image_path'],
                x=point[1],y=point[0],
                patch_size=112,
                # text=f"label: {label}",
                confidence=confidence,
                label=preds,
                save_path=os.path.join('./experiments/visual/',str(label),suc,image_name)
            )
            # print(os.path.join('./experiments/visual/',str(label),suc,image_name))
#         all_predictions.extend(predictions)
#         all_targets.extend(targets[0].numpy())
#         probs_list.extend(probs)
# # Convert all predictions and targets into numpy arrays
# all_predictions = np.array(all_predictions)
# all_targets = np.array(all_targets)
# probs = np.vstack(probs_list)
# # Calculate accuracy
# accuracy = accuracy_score(all_targets, all_predictions)

# max_label = all_targets.max()
# if max_label == 1:  # Binary classification
#     auc = roc_auc_score(all_targets, probs[:, 1])  # Assuming the second column is the probability of class 1
# else:  # Multi-class classification
#     auc = roc_auc_score(all_targets, probs, multi_class='ovr')
# recall=calculate_recall(all_targets,all_predictions)
# print(" Acc: {:.4f} | Auc: {:.4f} | REcall: {:.4f}".format(accuracy, auc, recall))
