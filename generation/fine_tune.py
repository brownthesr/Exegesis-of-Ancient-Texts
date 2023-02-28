"""
This file fine tuned the model on similarities between
the KJV and the World English versions
"""
from sentence_transformers import SentenceTransformer 
from transformers import AutoModel
from sentence_transformers import InputExample
import torch
import json
from sentence_transformers import losses

from torch.utils.data import DataLoader


model = SentenceTransformer('sentence-transformers/all-roberta-large-v1',device="cuda")

asv = {}
with open("datasets/World_english.json",'r') as f:
    asv = json.loads(f.read())
print("loaded world_english")

kjv = {}
with open("datasets/bible.json",'r') as f:
    kjv = json.loads(f.read())
print("loaded kjv")

train_examples = []
for a,b in zip(asv,kjv):
    train_examples.append(InputExample(texts=[a["text"], b["text"]]))
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
print("created dataloader")

train_loss = losses.MegaBatchMarginLoss(model=model)
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=10) 
print("fit model")

torch.save(model.state_dict(), "models/modern_speech_roberta.json")