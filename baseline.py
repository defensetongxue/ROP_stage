import torch
from torch.utils.data import DataLoader
from util.dataset import BaselineDataset as CustomDataset
from  models import build_model
import os,json
from torch.nn import CrossEntropyLoss
import numpy as np
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
args.configs['model']['name']=args.model_name

os.makedirs(args.save_dir,exist_ok=True)
print("Saveing the model in {}".format(args.save_dir))
# Create the model and criterion
model= build_model(args.configs["model"])# as we are loading the exite

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
    split='train',data_path=args.data_path,split_name=args.split_name,img_resize=args.resize)
val_dataset=CustomDataset(split='val',data_path=args.data_path,split_name=args.split_name,img_resize=args.resize)
test_dataset=CustomDataset(split='test',data_path=args.data_path,split_name=args.split_name,img_resize=args.resize)
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
    criterion = LabelSmoothingCrossEntropy(0.2)
else:
    
    criterion = CrossEntropyLoss()
val_loss_function = CrossEntropyLoss()
# init metic
print("There is {} batch size".format(args.configs["train"]['batch_size']))
print(f"Train batch size numeber: {len(train_loader)}, data number:{len(train_dataset)}")
print(f"Val batch size numeber: {len(val_loader)}, data number:{len(val_dataset)}")

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
    val_loss,  acc,auc= val_epoch(model, val_loader, val_loss_function, device)
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
        
model.load_state_dict(
    torch.load(os.path.join(args.save_dir,save_model_name)))
print(f"test batch size: {args.configs['train']['batch_size']}, test batch number: {len(test_loader)}, test data number: {len(test_dataset)}")
_,test_acc,test_auc= val_epoch(model,test_loader,val_loss_function,device)

print("test in patch format ",f"Acc:{test_acc:.4f}, Auc: {test_auc:.4f}")