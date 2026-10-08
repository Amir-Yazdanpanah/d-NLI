import torch
from torch.utils.data import  Dataset
import pandas as pd
# import dataloader
from transformers import AutoTokenizer
 
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
# tokenizer = AutoTokenizer.from_pretrained("HooshvareLab/bert-base-parsbert-uncased")
# tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
def tokenize_function(examples):
    return tokenizer(examples["premise"], examples["hypothesis"], max_length=512,padding="max_length", truncation=True,return_tensors='pt')

# def farstail_load(batch_size):
class farstailData(Dataset):
    def __init__(self, type):
        super(farstailData, self).__init__
        if type == 'train':
            self.data = pd.read_csv('./drive/MyDrive/HW4/dataset/Train-word.csv', sep='\t')
            self.classes = {self.data['label'][i] : i for i in range(3)}
        elif type == 'test':
            self.data = pd.read_csv('./drive/MyDrive/HW4/dataset/FarsTail/data/Val-word.csv', sep='\t')
            self.classes = {self.data['label'][i] : i for i in range(3)}
        if type == 'val':
            self.data = pd.read_csv('./drive/MyDrive/HW4/dataset/FarsTail/data/Test-word.csv', sep='\t')
            self.classes = {self.data['label'][i] : i for i in range(3)}

    def __getitem__(self, index):
        p_h = self.data.iloc[index][:-1]
        p_h_token = tokenize_function(p_h)
        label = self.classes[self.data['label'][index]]
        # print(self.data['label'].values[index])
        # print(p_h_token.shape)
        # print(label.shape)
        return p_h_token, label
    
    def __len__(self):
        return self.data.shape[0]

train_dataset = farstailData(type = 'train')
val_dataset = farstailData(type = 'val')
test_dataset = farstailData(type = 'test')

batch_size=4

farst_train_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size = batch_size,
                                                    shuffle = False, pin_memory = True, num_workers = 2)

farst_test_loader = torch.utils.data.DataLoader(test_dataset,
                                                    batch_size = batch_size,
                                                    shuffle = False, pin_memory = True, num_workers = 2)

farst_val_loader = torch.utils.data.DataLoader(val_dataset,
                                                    batch_size = batch_size,
                                                    shuffle = False, pin_memory = True, num_workers = 2)


# print(next(iter(farst_train_loader))[0])

# def main():
#     farst_train_loader, farst_val_loader,farst_test_loader = farstail_load(1)
# if __name__ == "__main__":main()