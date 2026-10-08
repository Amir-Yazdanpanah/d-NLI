import dataloader
import deeplearning
import utils
from nets.mBERT import B_model
# import nets
import torch
import pandas as pd
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

yaml = utils.load()
custom_model = B_model()

train_dataset = dataloader.farstailData(type = 'train')
val_dataset = dataloader.farstailData(type = 'val')
test_dataset = dataloader.farstailData(type = 'test')

batch_size = yaml['batch_size']

farst_train_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size = batch_size,
                                                    shuffle = True, pin_memory = True, num_workers = 2)

farst_test_loader = torch.utils.data.DataLoader(test_dataset,
                                                    batch_size = batch_size,
                                                    shuffle = False, pin_memory = True, num_workers = 2)

farst_val_loader = torch.utils.data.DataLoader(val_dataset,
                                                    batch_size = batch_size,
                                                    shuffle = False, pin_memory = True, num_workers = 2)


trainer = deeplearning.train(
    batch_size = yaml['batch_size'],
    farst_train_loader = farst_train_loader,
    farst_val_loader = farst_val_loader,
    farst_test_loader = farst_test_loader,
    model = custom_model,
    model_name=yaml['name_model'],
    epochs=yaml['epochs'],
    learning_rate=yaml['learning_rate'],
    gamma = yaml['gamma'],
    step_size = yaml['step_size'],
    device=device,
    load_saved_model=yaml['load_saved_model'],
    ckpt_save_freq=yaml['ckpt_save_freq'],
    ckpt_save_path="/content/drive/MyDrive/HW4/",
    ckpt_path="/content/drive/MyDrive/HW4/ckpt_parsBERT_epoch2.ckpt",
    report_path="/content/drive/MyDrive/HW4/",
)
