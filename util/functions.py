import torch,math
from torch import optim
import numpy as np
from .metric import Metrics
from torch import nn
from sklearn.metrics import accuracy_score, roc_auc_score


def to_device(x, device):
    if isinstance(x, tuple):
        return tuple(to_device(xi, device) for xi in x)
    elif isinstance(x,list):
        return [to_device(xi,device) for xi in x]
    else:
        return x.to(device)

def train_epoch(model, optimizer, train_loader, loss_function, device,lr_scheduler,epoch):
    model.train()
    running_loss = 0.0
    batch_length=len(train_loader)
    for data_iter_step,(inputs, targets, meta) in enumerate(train_loader):
        # Moving inputs and targets to the correct device
        lr_scheduler.adjust_learning_rate(optimizer,epoch+(data_iter_step/batch_length))
        inputs = to_device(inputs, device)
        targets = to_device(targets, device)

        optimizer.zero_grad()

        # Assuming your model returns a tuple of outputs
        outputs = model(inputs)

        # Assuming your loss function can handle tuples of outputs and targets
        loss = loss_function(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    return running_loss / len(train_loader)


def val_epoch(model, val_loader, loss_function, device):
    loss_function=nn.CrossEntropyLoss()
    model.eval()
    running_loss = 0.0
    all_predictions = []
    all_targets = []
    all_probs = []
    with torch.no_grad():
        for inputs, targets, _ in val_loader:
            inputs = to_device(inputs,device)
            targets = to_device(targets,device)
            outputs = model(inputs)
            loss = loss_function(outputs, targets)
            running_loss += loss.item()
            probs = torch.softmax(outputs.cpu(), dim=1).numpy()
            predictions = np.argmax(probs, axis=1)
            all_predictions.extend(predictions)
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs)
            
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    all_probs = np.vstack(all_probs)
    accuracy = accuracy_score(all_targets, all_predictions)
    auc = roc_auc_score(all_targets,all_probs, multi_class='ovr')
    return running_loss / len(val_loader), accuracy,auc

def get_instance(module, class_name, *args, **kwargs):
    cls = getattr(module, class_name)
    instance = cls(*args, **kwargs)
    return instance

def get_optimizer(cfg, model):
    if  cfg['train']['layer_decay']<=1 and False:
        param_groups = param_groups_lrd(model, cfg['train']['wd'],
        no_weight_decay_list=model.no_weight_decay(),
        layer_decay=cfg['train']['layer_decay'],
        )
        optimizer = torch.optim.AdamW(param_groups, cfg['train']['lr'])
    else:
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg['train']['lr'], weight_decay=cfg['train']['wd']
        )
    
    return optimizer

class lr_sche():
    def __init__(self,config):
        self.warmup_epochs=config["warmup_epochs"]
        self.lr=config["lr"]
        self.min_lr=config["min_lr"]
        self.epochs=config['epochs']
    def adjust_learning_rate(self,optimizer, epoch):
        """Decay the learning rate with half-cycle cosine after warmup"""
        if epoch < self.warmup_epochs:
            lr = self.lr * epoch / self.warmup_epochs
        else:
            lr = self.min_lr + (self.lr  - self.min_lr) * 0.5 * \
                (1. + math.cos(math.pi * (epoch - self.warmup_epochs) / (self.epochs - self.warmup_epochs)))
        for param_group in optimizer.param_groups:
            if "lr_scale" in param_group:
                param_group["lr"] = lr * param_group["lr_scale"]
            else:
                param_group["lr"] = lr
        return lr
def param_groups_lrd(model, weight_decay=0.05, no_weight_decay_list=[], layer_decay=.75):
    """
    Parameter groups for layer-wise lr decay
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58
    """
    param_group_names = {}
    param_groups = {}

    num_layers = len(model.blocks) + 1
    
    layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # no decay: all 1D parameters and model specific ones
        if p.ndim == 1 or n in no_weight_decay_list:
            g_decay = "no_decay"
            this_decay = 0.
        else:
            g_decay = "decay"
            this_decay = weight_decay
            
        layer_id = get_layer_id_for_vit(n, num_layers)
        group_name = "layer_%d_%s" % (layer_id, g_decay)

        if group_name not in param_group_names:
            if n.startswith('patch_embed'):
                this_scale =1.
            else:
                this_scale = layer_scales[layer_id]

            param_group_names[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }
            param_groups[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }

        param_group_names[group_name]["params"].append(n)
        param_groups[group_name]["params"].append(p)
    # import json
    # print("parameter groups: \n%s" % json.dumps(param_group_names, indent=2))
    # raise
    return list(param_groups.values())


def get_layer_id_for_vit(name, num_layers):
    """
    Assign a parameter with its layer id
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
    """
    if name in ['cls_token', 'pos_embed']:
        return 0
    elif name.startswith('patch_embed'):
        return 0
    elif name.startswith('blocks'):
        return int(name.split('.')[1]) + 1
    else:
        return num_layers