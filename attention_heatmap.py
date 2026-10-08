import torch
import numpy as np

from transformers import AutoModel
from transformers import BertTokenizer, BertModel
from transformers import AutoTokenizer
import seaborn as sns
import matplotlib.pyplot as plt

tokenizer = AutoTokenizer.from_pretrained("HooshvareLab/bert-base-parsbert-uncased")
parsi_bert = AutoModel.from_pretrained("HooshvareLab/bert-base-parsbert-uncased", output_attentions=True )

def attention_heatmap():
    input_text = "با ظهور هیتلریسم موسیقی عامیانه و ملی مجدداً مورد توجه قرار گرفت."
    inputs = tokenizer.encode(input_text, return_tensors='pt')
    outputs = parsi_bert(inputs)
    tokens = tokenizer.convert_ids_to_tokens(inputs[0])
    fig, ax = plt.subplots(figsize=(30, 30))
    sns.heatmap(outputs[-1][0][0][-3].detach().numpy(), xticklabels=tokens, yticklabels=tokens, cmap='Reds')
    
    plt.savefig('figure.png')
    plt.show()

    fig, ax = plt.subplots(figsize=(30, 30))
    sns.heatmap(outputs[-1][0][0][-3][2].detach().numpy().reshape(1,-1), xticklabels=tokens, yticklabels=tokens, cmap='Reds', annot = True)
    plt.savefig('figure2.png')
    plt.show()
attention_heatmap()