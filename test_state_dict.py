import torch
from transformers import AutoModel

# create a BERT model
model = AutoModel.from_pretrained('google/bert_uncased_L-2_H-128_A-2')

# get the state_dict of the model
state_dict = model.state_dict()

# print the keys of the state_dict
print(state_dict.keys())