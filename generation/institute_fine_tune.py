"""
This file will fine tune the data on the institute manual.
It still needs some work for right now, but the basic idea
is to use MNBRL
"""

from sentence_transformers import SentenceTransformer 
from transformers import AutoModel
from sentence_transformers import InputExample
import torch
import json
from sentence_transformers import losses

from torch.utils.data import DataLoader
james = {}
with open("datasets/bible.json",'r') as f:
    file = json.loads(f.read())
for verse in file:
    if verse["book"] == "James":
        james[verse["reference"]] = verse["text"]
        # print(verse["reference"])
model = SentenceTransformer('sentence-transformers/all-roberta-large-v1',device="cuda")

with open("datasets/institute_man.json",'r') as f:
    file = json.loads(f.read())
train_examples = []
for comparison in file:
    for verse in comparison["verses"]:
        verses = []
        verses.append(james[verse])
        print(verses)
        for text in comparison["text"]:
            verses.append(text)
            train_examples.append(InputExample(texts=verses))
            verses = verses[:-1]
    # print(verses)
print(len(train_examples))
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=100)
print("created_dataloader")

train_loss = losses.MultipleNegativesRankingLoss(model=model)
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=100) 
print("fitted_model")
torch.save(model.state_dict(), "models/institute_fine_tuned.json")