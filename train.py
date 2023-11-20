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

os.makedirs(args.save_dir,exist_ok=True)
print("Saveing the model in {}".format(args.save_dir))
# Create the model and criterion
model= build_model(num_classes=args.configs["num_classes"],num_patches=args.word_size)# as we are loading the exite
model.load_pretrained(pretrained_path=args.configs["pretrained_path"])


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
args.configs["smoothing"]=0
if args.configs["smoothing"]> 0.:
    from models.losses import LabelSmoothingCrossEntropy
    criterion = LabelSmoothingCrossEntropy(train_dataset+val_dataset,smoothing=args.configs["smoothing"])
else:
    from models.losses import AdaptiveCrossEntropyLoss
    criterion = AdaptiveCrossEntropyLoss(train_dataset+val_dataset,device)
    
# init metic
metirc= Metrics(val_dataset,"Main")
metirc_aux=Metrics(val_dataset,"Aux")
print("There is {} batch size".format(args.configs["train"]['batch_size']))
print(f"Train: {len(train_loader)}, Val: {len(val_loader)}")

early_stop_counter = 0
best_val_loss = float('inf')
best_auc=0
total_epoches=args.configs['train']['end_epoch']
save_model_name=args.split_name+args.configs['save_name']
# Training and validation loop
for epoch in range(last_epoch,total_epoches):

    train_loss = train_epoch(model, optimizer, train_loader, criterion, device,lr_scheduler,epoch)
    val_loss,  metirc,metirc_aux= val_epoch(model, val_loader, criterion, device,metirc,metirc_aux)
    print(f"Epoch {epoch + 1}/{total_epoches}, "
      f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
      f"Lr: {optimizer.state_dict()['param_groups'][0]['lr']:.4f}"
      )
    print(metirc)
    print(metirc_aux)
    # Update the learning rate if using ReduceLROnPlateau or CosineAnnealingLR
    # Early stopping
    if val_loss <best_val_loss:
        best_val_loss=val_loss
        early_stop_counter = 0
        torch.save(model.state_dict(),
                   os.path.join(args.save_dir,save_model_name))
        print("Model saved as {}".format(os.path.join(args.save_dir,save_model_name)))
    else:
        early_stop_counter += 1
        if early_stop_counter >= args.configs['train']['early_stop']:
            print("Early stopping triggered")
            break


# Load the best model and evaluate
metirc=Metrics(test_dataset,"Main")
metirc_aux=Metrics(test_dataset,"Aux")
model.load_state_dict(
        torch.load(os.path.join(args.save_dir, save_model_name)))
val_loss, metirc,metirc_aux=val_epoch(model, test_loader, criterion, device,metirc,metirc_aux)
print(f"Best Epoch ")
print(metirc)
print(metirc_aux)

# record_name = f"{args.split_name}"

# # Load existing records
# with open('record.json', 'r') as f:
#     record = json.load(f)

# # Update the record
# record[record_name] = {
#     "Last_epoch": {
#         "Auc": round(auc_last, 4),
#         "Acc": round(accuracy_last, 4),
#         "Recall": round(recall_last, 4)
#     },
#     "Best_epoch": {
#         "Auc": round(auc, 4),
#         "Acc": round(accuracy, 4),
#         "Recall": round(recall, 4)
#     }
# }
# # Save the updated record
# with open("record.json", 'w') as f:
#     json.dump(record, f, indent=4)