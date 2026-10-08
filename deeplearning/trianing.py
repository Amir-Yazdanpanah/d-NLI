import os
import glob
import cv2
from PIL import Image

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

def train(
    batch_size,
    farst_train_loader,
    farst_val_loader,
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

    lr_scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    report = pd.DataFrame(
        columns=[
            "model_name",
            "mode",
            "image_type",
            "epoch",
            "learning_rate",
            "batch_size",
            "batch_index",
            "loss_batch",
            "avg_train_loss_till_current_batch",
            "avg_train_top1_acc_till_current_batch",
            "avg_val_loss_till_current_batch",
            "avg_val_top1_acc_till_current_batch",
            "avg_test_loss_till_current_batch",
            "avg_test_top1_acc_till_current_batch"])

    for epoch in tqdm(range(1, epochs + 1)):
        top1_acc_train = AverageMeter()
        loss_avg_train = AverageMeter()
        top1_acc_val = AverageMeter()
        loss_avg_val = AverageMeter()
        top1_acc_test = AverageMeter()
        loss_avg_test = AverageMeter()

        model.train()
        mode = "train"
        
        loop_train = tqdm(
            enumerate(farst_train_loader, 1),
            total=len(farst_train_loader),
            desc="train",
            position=0,
            leave=True)
        for batch_idx, (k, labels) in loop_train:
            input_ids=k['input_ids'][:,0]
            # attention_mask = k['attention_mask'][:,0]
            attention_mask = None
            token_type_ids = k['token_type_ids'][:,0]
            # token_type_ids = None
            # label=k[1]
            input_ids = input_ids.to(device, non_blocking=True,)
            # attention_mask = attention_mask.to(device, non_blocking=True,)
            token_type_ids = token_type_ids.to(device, non_blocking=True,)
            labels = labels.to(device, non_blocking=True)

            labels_pred = model(input_ids = input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids)

            loss = criterion(labels_pred, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            acc1 = utils.accuracy(labels_pred, labels)
            top1_acc_train.update(acc1[0], k['input_ids'].size(0))
            loss_avg_train.update(loss.item(), k['input_ids'].size(0))

            new_row = pd.DataFrame(
                {"model_name": model_name,
                "mode": mode,
                "image_type":"original",
                "epoch": epoch + extra_epoch,
                "learning_rate":optimizer.param_groups[0]["lr"],
                "batch_size": k["input_ids"].size(0),
                "batch_index": batch_idx,
                "loss_batch": loss.detach().item(),
                "avg_train_loss_till_current_batch":loss_avg_train.avg,
                "avg_train_top1_acc_till_current_batch":top1_acc_train.avg,
                "avg_val_loss_till_current_batch":None,
                "avg_val_top1_acc_till_current_batch":None,
                "avg_test_loss_till_current_batch":None,
                "avg_test_top1_acc_till_current_batch":None},index=[0])

            
            report.loc[len(report)] = new_row.values[0]


            
            loop_train.set_description(f"Train - iteration : {epoch}")
            loop_train.set_postfix(
                loss_batch="{:.4f}".format(loss.detach().item()),
                avg_train_loss_till_current_batch="{:.4f}".format(loss_avg_train.avg),
                top1_accuracy_train="{:.4f}".format(top1_acc_train.avg),
                max_len=2,
                refresh=True,
            )
            
        lr_scheduler.step()
        if epoch % ckpt_save_freq == 0:
            utils.save_model(
                file_path=ckpt_save_path,
                file_name=f"ckpt_{model_name}_epoch{epoch + extra_epoch}.ckpt",
                model=model,
                optimizer=optimizer,
            )

        model.eval()
        mode = "val"
        with torch.no_grad():
            loop_val = tqdm(
                enumerate(farst_val_loader, 1),
                total=len(farst_val_loader),
                desc="val",
                position=0,
                leave=True,
            )
            for batch_idx, (k, labels) in loop_val:
                input_ids=k['input_ids'][:,0]
                # attention_mask = k['attention_mask'][:,0]
                attention_mask = None
                token_type_ids = k['token_type_ids'][:,0]
                # token_type_ids = None

                optimizer.zero_grad()

                input_ids = input_ids.to(device, non_blocking=True)
                # attention_mask = attention_mask.to(device, non_blocking=True)
                token_type_ids = token_type_ids.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                labels_pred = model(input_ids = input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids)

                loss = criterion(labels_pred, labels)
                acc1 = utils.accuracy(labels_pred, labels)
                top1_acc_val.update(acc1[0], k['input_ids'].size(0))
                loss_avg_val.update(loss.item(), k['input_ids'].size(0))
                new_row = pd.DataFrame(
                    {"model_name": model_name,
                    "mode": mode,
                    "image_type":"original",
                    "epoch": epoch + extra_epoch,
                    "learning_rate":optimizer.param_groups[0]["lr"],
                    "batch_size": k["input_ids"].size(0),
                    "batch_index": batch_idx,
                    "loss_batch": loss.detach().item(),
                    "avg_train_loss_till_current_batch":None,
                    "avg_train_top1_acc_till_current_batch":None,
                    "avg_val_loss_till_current_batch":loss_avg_val.avg,
                    "avg_val_top1_acc_till_current_batch":top1_acc_val.avg,
                    "avg_test_loss_till_current_batch":None,
                    "avg_test_top1_acc_till_current_batch":None},index=[0],)
                
                report.loc[len(report)] = new_row.values[0]
                loop_val.set_description(f"val - iteration : {epoch}")
                loop_val.set_postfix(
                    loss_batch="{:.4f}".format(loss.detach().item()),
                    avg_val_loss_till_current_batch="{:.4f}".format(loss_avg_val.avg),
                    top1_accuracy_val="{:.4f}".format(top1_acc_val.avg),
                    refresh=True,
                )

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
                # attention_mask = k['attention_mask'][:,0]
                attention_mask = None
                token_type_ids = k['token_type_ids'][:,0]
                # token_type_ids = None


                input_ids = input_ids.to(device, non_blocking=True)
                # attention_mask = attention_mask.to(device, non_blocking=True)
                token_type_ids = token_type_ids.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                labels_pred = model(input_ids = input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids)

                loss = criterion(labels_pred, labels)
                acc1 = utils.accuracy(labels_pred, labels)
                top1_acc_test.update(acc1[0], k['input_ids'].size(0))
                loss_avg_test.update(loss.item(), k['input_ids'].size(0))


                new_row = pd.DataFrame(
                    {"model_name": model_name,
                    "mode": mode,
                    "image_type":"original",
                    "epoch": epoch + extra_epoch,
                    "learning_rate":optimizer.param_groups[0]["lr"],
                    "batch_size": k['input_ids'].size(0),
                    "batch_index": batch_idx,
                    "loss_batch": loss.detach().item(),
                    "avg_train_loss_till_current_batch":None,
                    "avg_train_top1_acc_till_current_batch":None,
                    "avg_val_loss_till_current_batch":None,
                    "avg_val_top1_acc_till_current_batch":None,
                    "avg_test_loss_till_current_batch":loss_avg_test.avg,
                    "avg_test_top1_acc_till_current_batch":top1_acc_test.avg},index=[0],)
                
                report.loc[len(report)] = new_row.values[0]
                loop_val.set_description(f"test - iteration : {epoch}")
                loop_val.set_postfix(
                    loss_batch="{:.4f}".format(loss.detach().item()),
                    avg_test_loss_till_current_batch="{:.4f}".format(loss_avg_test.avg),
                    top1_accuracy_test="{:.4f}".format(top1_acc_test.avg),
                    refresh=True,
                )
    if(load_saved_model):
      report.to_csv(f"{report_path}/{model_name}_report.csv", mode = 'a')
    else:
      report.to_csv(f"{report_path}/{model_name}_report.csv")
    return model, optimizer, report
