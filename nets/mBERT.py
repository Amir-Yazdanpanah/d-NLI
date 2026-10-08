import torch
from transformers import AutoModel
from transformers import BertTokenizer, BertModel
from transformers import AutoModelForSequenceClassification
import torch.nn as nn
import dataloader

class B_model(nn.Module):
    def __init__(self):
        super(B_model, self).__init__()
        # model_name = "HooshvareLab/bert-base-parsbert-uncased"
        # model_name = "bert-base-multilingual-cased"
        self.bertt = BertModel.from_pretrained("HooshvareLab/bert-base-parsbert-uncased", output_hidden_states = True)

        self.linear = nn.Sequential(
            nn.Tanh(),
            nn.Linear(768, 3076, bias = True),
            nn.BatchNorm1d(3076),
            nn.Dropout(0.2),
            nn.Tanh(),
            nn.Linear(3076, 3, bias = True),
            nn.Dropout(0.1),
            # nn.BatchNorm1d(3),
            nn.Tanh()
        )
    def forward(self, input_ids, attention_mask, token_type_ids):
        # x = self.bertt(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)['pooler_output']
        x = self.bertt(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)['hidden_states']
        x = x[-1]
        x = x[:,0]
        x = self.linear(x)
        return x