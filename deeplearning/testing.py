import os
import glob
import cv2
from PIL import Image
import numpy as np
import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import  Dataset
from torch.optim.lr_scheduler import StepLR

import pandas as pd
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

import utils
from utils.Averagemeter import AverageMeter

def save_model(file_path, file_name, model, optimizer=None):
    """
    In this function, a model is saved.Usually save model after training in each epoch.
    ------------------------------------------------
    Args:
        - model (torch.nn.Module)
        - optimizer (torch.optim)
        - file_path (str): Path(Folder) for saving the model
        - file_name (str): name of the model checkpoint to save
    """
    state_dict = dict()
    state_dict["model"] = model.state_dict()

    if optimizer is not None:
        state_dict["optimizer"] = optimizer.state_dict()
    torch.save(state_dict, os.path.join(file_path, file_name))


def load_model(ckpt_path, model, optimizer=None):
    """
    Loading a saved model and optimizer (from checkpoint)
    """
    checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["model"])
    if (optimizer != None) & ("optimizer" in checkpoint.keys()):
        optimizer.load_state_dict(checkpoint["optimizer"])
    return model, optimizer

def test(
    farst_test_loader,
    model,
    model_name,
    epochs,
    learning_rate,
    gamma,
    step_size,
    device,
    load_saved_model,
    ckpt_save_freq,
    ckpt_save_path,
    ckpt_path,
    report_path,
):
    torch.backends.cudnn.benchmark = True 
    model = model.to(device, non_blocking=True)

    # loss function
    criterion = nn.CrossEntropyLoss()

    # optimzier
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if load_saved_model:
        model, optimizer = load_model(
            ckpt_path=ckpt_path, model=model, optimizer=optimizer
        )
        s = ckpt_path.find('.ckpt')
        extra_epoch = int(ckpt_path[s-1:s])
    else:
        extra_epoch = 0

    cf = []
    labels_pred_cf = np.array([])
    labels_cf = np.array([])

    for epoch in tqdm(range(1, epochs + 1)):
        top1_acc_test = AverageMeter()
        loss_avg_test = AverageMeter()


        model.eval()
        mode = "test"
        with torch.no_grad():
            loop_val = tqdm(
                enumerate(farst_test_loader, 1),
                total=len(farst_test_loader),
                desc="test",
                position=0,
                leave=True,
            )
            for batch_idx, (k, labels) in loop_val:
                optimizer.zero_grad()
                input_ids=k['input_ids'][:,0]
                attention_mask = k['attention_mask'][:,0]
                # attention_mask = None
                token_type_ids = k['token_type_ids'][:,0]
                # token_type_ids = None


                input_ids = input_ids.to(device, non_blocking=True)
                attention_mask = attention_mask.to(device, non_blocking=True)
                token_type_ids = token_type_ids.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                labels_pred = model(input_ids = input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids)

                labels_pred_cf = np.append(labels_pred_cf, labels_pred.cpu().numpy().argmax())
                labels_cf = np.append(labels_cf, labels.cpu().numpy())

                loss = criterion(labels_pred, labels)
                acc1 = utils.accuracy(labels_pred, labels)
                top1_acc_test.update(acc1[0], k['input_ids'].size(0))
                loss_avg_test.update(loss.item(), k['input_ids'].size(0))

                
                loop_val.set_description(f"test - iteration : {epoch}")
                loop_val.set_postfix(
                    loss_batch="{:.4f}".format(loss.detach().item()),
                    avg_test_loss_till_current_batch="{:.4f}".format(loss_avg_test.avg),
                    top1_accuracy_test="{:.4f}".format(top1_acc_test.avg),
                    refresh=True,
                )

    # Compute confusion matrix
    cr = classification_report(labels_cf, labels_pred_cf, labels=[0, 1, 2])
    print(cr)
    cm = confusion_matrix(labels_cf, labels_pred_cf, labels=[0, 1, 2])

    # Plot confusion matrix as heatmap
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",  xticklabels=["0", "1", "2"], yticklabels=["0", "1", "2"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    
    # Save the figure as a PNG file
    plt.savefig('figure.png')

    # Download the file
    files.download('figure.png')
    plt.show()
    return model, optimizer
