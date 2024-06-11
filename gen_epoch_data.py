import torch
from torch.utils.data import DataLoader
from util.dataset import CustomDataset
from  models import build_model
import os,json
from torch.nn import CrossEntropyLoss
import numpy as np
from util.functions import train_epoch,val_epoch,get_optimizer,lr_sche
from configs import get_config
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score, roc_auc_score
class get_test_result:
    def __init__(self,split_list_path,data_dict_path):
        with open(split_list_path) as f:
            split_list=json.load(f)['test']
        with open(data_dict_path) as f:
            data_dict=json.load(f)
        self.label_dict={}
        self.pred_dict={}
        for image_name in split_list:
            self.label_dict[image_name]=data_dict[image_name]['stage']-1
            self.pred_dict[image_name]=0
    def reset(self):
        for image_name in self.pred_dict:
            self.pred_dict[image_name]=0
        
    def get_res(self):
        label_list=[]
        pred_list=[]
        for image_name in self.label_dict: #do not change this
            label_list.append(self.label_dict[image_name])
            pred_list.append(self.pred_dict[image_name])
        acc = accuracy_score(label_list, pred_list)

        # One-hot encode labels and predictions for AUC calculation
        classes = np.unique(label_list)
        label_one_hot = label_binarize(label_list, classes=classes)
        pred_one_hot = label_binarize(pred_list, classes=classes)

        # Calculate AUC; handle cases where there is only one class present in the dataset
        try:
            if len(classes) == 1:
                auc = float('nan')  # Not enough classes to compute AUC
            else:
                auc = roc_auc_score(label_one_hot, pred_one_hot, multi_class="ovr")
        except ValueError as e:
            print(f"Error computing AUC: {e}")
            auc = float('nan')

        return acc, auc
    def test_epoch(self,test_loader,model,device):
        self.reset()
        with torch.no_grad():
            for img,images_names in test_loader:
            
                outputs = model(img.to(device))
                probs = torch.softmax(outputs.cpu(), axis=1)
                pred_labels = torch.argmax(probs, dim=1)

                for pred_label,image_name in zip(pred_labels,images_names):
                    self.pred_dict[image_name]=max(self.pred_dict[image_name],int(pred_label))
        return self.get_res()

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
    split='train',data_path=args.data_path,split_name='stage_1',img_resize=args.resize)
val_dataset=CustomDataset(split='val',data_path=args.data_path,split_name='stage_1',img_resize=args.resize)
# Create the data loaders
    
train_loader = DataLoader(train_dataset, 
                          batch_size=args.configs['train']['batch_size'],
                          shuffle=True, num_workers=args.configs['num_works'],drop_last=True)
val_loader = DataLoader(val_dataset,
                        batch_size=args.configs['train']['batch_size'],
                        shuffle=False, num_workers=args.configs['num_works'])
if args.configs["smoothing"]> 0.:
    from timm.loss import LabelSmoothingCrossEntropy
    criterion = LabelSmoothingCrossEntropy(0.2)
else:
    criterion = CrossEntropyLoss()
if args.model_name=='inceptionv3':
    from util.losses import InceptionV3Loss
    criterion=InceptionV3Loss(args.configs["smoothing"])
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

from util.dataset import TestPatchDataset
test_dataset=TestPatchDataset(os.path.join(args.data_path,'patch_clr_1','annotations.json'))
test_loader=DataLoader(test_dataset,batch_size=args.configs['train']['batch_size'],
                          shuffle=True, num_workers=args.configs['num_works'],drop_last=True)
test_handler=get_test_result(os.path.join(args.data_path,'split','stage_1.json'),os.path.join(args.data_path,'annotations.json'))
record={}
for epoch in range(last_epoch,total_epoches):
    train_loss = train_epoch(model, optimizer, train_loader, criterion, device,lr_scheduler,epoch)
    acc,auc=test_handler.test_epoch(test_loader,model,device)
    
    record[epoch]={'acc':f"{acc:.4f}",'auc':f"{auc:.4f}"}
    print(acc)
with open('./experiments/0507.json','w') as f:
    json.dump(record,f)