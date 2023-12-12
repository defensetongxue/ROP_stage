import torch
from torch.utils.data import DataLoader
from util.dataset import CustomDataset
from  models import build_model
import os,json
import numpy as np
from util.metric import Metrics
from util.functions import train_epoch,val_epoch,get_optimizer,lr_sche
from configs import get_config
# Initialize the folder
os.makedirs("checkpoints",exist_ok=True)
os.makedirs("experiments",exist_ok=True)
torch.manual_seed(0)
np.random.seed(0)
# Parse arguments
args = get_config()

args.configs['train']['wd']=args.wd
args.configs["lr_strategy"]['lr']=args.lr

os.makedirs(args.save_dir,exist_ok=True)
print("Saveing the model in {}".format(args.save_dir))
# Create the model and criterion
model= build_model(num_classes=args.configs["num_classes"])# as we are loading the exite

# Set up the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(f"using {device} for training")

# early stopping
early_stop_counter = 0


# Creatr optimizer
model.train()
# Creatr optimizer
optimizer = get_optimizer(args.configs, model)
lr_scheduler=lr_sche(config=args.configs["lr_strategy"])
last_epoch = args.configs['train']['begin_epoch']

# Load the datasets
train_dataset=CustomDataset(
    split='train',data_path=args.data_path,split_name=args.split_name)
val_dataset=CustomDataset(split='val',data_path=args.data_path,split_name=args.split_name)
test_dataset=CustomDataset(split='test',data_path=args.data_path,split_name=args.split_name)
# Create the data loaders
    
train_loader = DataLoader(train_dataset, 
                          batch_size=args.configs['train']['batch_size'],
                          shuffle=True, num_workers=args.configs['num_works'],drop_last=True)
val_loader = DataLoader(val_dataset,
                        batch_size=args.configs['train']['batch_size'],
                        shuffle=False, num_workers=args.configs['num_works'])
test_loader=  DataLoader(test_dataset,
                        batch_size=args.configs['train']['batch_size'],
                        shuffle=False, num_workers=args.configs['num_works'])
if args.configs["smoothing"]> 0.:
    from timm.loss import LabelSmoothingCrossEntropy
    criterion = LabelSmoothingCrossEntropy()
else:
    from torch.nn import CrossEntropyLoss
    criterion = CrossEntropyLoss()
    
# init metic
print("There is {} batch size".format(args.configs["train"]['batch_size']))
print(f"Train: {len(train_loader)}, Val: {len(val_loader)}")

early_stop_counter = 0
best_val_loss = float('inf')
best_auc=0
best_avgrecall=0
total_epoches=args.configs['train']['end_epoch']
save_model_name=args.split_name+args.configs['save_name']
save_epoch=0
# Training and validation loop

for epoch in range(last_epoch,total_epoches):
    train_loss = train_epoch(model, optimizer, train_loader, criterion, device,lr_scheduler,epoch)
    val_loss,  acc,auc= val_epoch(model, val_loader, criterion, device)
    print(f"Epoch {epoch + 1}/{total_epoches}, "
      f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, "
      f"Lr: {optimizer.state_dict()['param_groups'][0]['lr']:.6f}, "
      f"Acc:{acc:.4f}, Auc: {auc:.4f}"
      )
    if auc >best_auc:
        save_epoch=epoch
        best_auc= auc
        early_stop_counter = 0
        torch.save(model.state_dict(),
                   os.path.join(args.save_dir,save_model_name))
        print("[Save Model In Epoch {}] Model saved as {}".format(str(epoch),os.path.join(args.save_dir,save_model_name)))
    else:
        early_stop_counter += 1
        if early_stop_counter >= args.configs['train']['early_stop']:
            print("Early stopping triggered")
            break 

with open(os.path.join(args.data_path,'split',f'{args.split_name}.json'), 'r') as f:
    ori_split_list=json.load(f)['test']
with open(os.path.join(args.data_path,'annotations.json'),'r') as f:
    data_dict=json.load(f)
metric={}
for image_name in ori_split_list:
    metric[image_name]=0

model.load_state_dict(
    torch.load(os.path.join(args.save_dir,save_model_name))
)   
model.eval() 
for imgs,labels,image_names in test_loader:
    outputs = model(imgs.cuda())
    probs= torch.softmax(outputs,dim=-1).detach().cpu().numpy()
    predictions = np.argmax(probs, axis=1)
    for image_name,preds in zip(image_names,predictions):
        metric[image_name]= max(preds,metric[image_name])
        
confu_matrix=np.zeros((3,3))
for image_name in metric:
    if data_dict[image_name]['stage'] ==0:
        continue
    confu_matrix[metric[image_name],data_dict[image_name]['stage']-1]+=1
print(confu_matrix)
# Total number of predictions
total_predictions = confu_matrix.sum()

# Total correct predictions
correct_predictions = np.trace(confu_matrix)

# Accuracy
accuracy = correct_predictions / total_predictions

# Recall for each class
recall_1 = confu_matrix[0, 0] / confu_matrix[0, :].sum()
recall_2 = confu_matrix[1, 1] / confu_matrix[1, :].sum()
recall_3 = confu_matrix[2, 2] / confu_matrix[2, :].sum()

print(f"Accuracy: {accuracy:.2f}")
print(f"Recall for Class 1: {recall_1:.2f}")
print(f"Recall for Class 2: {recall_2:.2f}")
print(f"Recall for Class 3: {recall_3:.2f}")
# print(f"Positive Recall: {positive_recall:.2f}")