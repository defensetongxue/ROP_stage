import torch,math
from torch import optim
import numpy as np
from .metric import Metrics
from torch import nn
from collections import Counter


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


def val_epoch(model, val_loader, loss_function, device,metirc:Metrics,metirc_aux:Metrics):
    loss_function=nn.CrossEntropyLoss()
    model.eval()
    running_loss = 0.0
    all_predictions = []
    all_predicction_aux=[]
    all_targets = []
    all_probs_aux = []
    all_probs = []
    with torch.no_grad():
        for inputs, targets, _ in val_loader:
            inputs = to_device(inputs,device)
            targets = to_device(targets[0],device)
            outputs = model(inputs)
            loss = loss_function(outputs[0], targets)
            running_loss += loss.item()
            probs = torch.softmax(outputs[0].cpu(), dim=1).numpy()
            predictions = np.argmax(probs, axis=1)
            probs_aux=torch.softmax(outputs[1].cpu(), dim=-1).numpy()
            predictions_aux_word = np.argmax(probs_aux, axis=2)
            predictions_aux=np.max(predictions_aux_word ,axis=1)
            for i in range(len(targets)):
                # Identify words contributing to image-level prediction
                contributing_words = (predictions_aux_word[i] == int(predictions_aux[i]))

                # Select probabilities for contributing words
                selected_probs = probs_aux[i][contributing_words]

                # Calculate the average probability for the contributing words
                if len(selected_probs) > 0:
                    avg_probs = np.mean(selected_probs, axis=0)
                else:
                    avg_probs = np.zeros(probs_aux.shape[-1])

                all_probs_aux.append(avg_probs)

            all_predictions.extend(predictions)
            all_predicction_aux.extend(predictions_aux)
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs)
            
    all_predicction_aux=np.array(all_predicction_aux)
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    all_probs = np.vstack(all_probs)
    all_probs_aux = np.array(all_probs_aux)
    # print(all_predictions.shape,all_probs.shape,)
    metirc.update(all_predictions,all_probs,all_targets)
    metirc_aux.update(all_predicction_aux,all_probs_aux,all_targets)
    return running_loss / len(val_loader), metirc,metirc_aux
def get_instance(module, class_name, *args, **kwargs):
    cls = getattr(module, class_name)
    instance = cls(*args, **kwargs)
    return instance

def get_optimizer(cfg, model):
    optimizer = None
    if cfg['train']['optimizer'] == 'sgd':
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg['train']['lr'],
            momentum=cfg['train']['momentum'],
            weight_decay=cfg['train']['wd'],
            nesterov=cfg['train']['nesterov']
        )
    elif cfg['train']['optimizer'] == 'adam':
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg['train']['lr']
        )
    elif cfg['train']['optimizer'] == 'rmsprop':
        optimizer = optim.RMSprop(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg['train']['lr'],
            momentum=cfg['train']['momentum'],
            weight_decay=cfg['train']['wd'],
            alpha=cfg['train']['rmsprop_alpha'],
            centered=cfg['train']['rmsprop_centered']
        )
    else:
        raise
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
    
# def calculate_recall(labels, preds):
#     """
#     Calculate recall for class 1 in a binary classification task.
    
#     Args:
#     labels (np.array): Array of true labels.
#     preds (np.array): Array of predicted labels.
    
#     Returns:
#     float: Recall for class 1.
#     """
#     # Ensure labels and predictions are numpy arrays
#     labels = np.array(labels)
#     preds = np.array(preds)
#     labels[labels>0]=1
#     preds[preds>0]=1
#     # Calculate True Positives and False Negatives
#     true_positives = np.sum((labels == 1) & (preds == 1))
#     false_negatives = np.sum((labels == 1) & (preds == 0))

#     # Calculate recall
#     recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
#     return recall