"""
This file processes the verses in a sliding window so that each verse attends to 
both the verses in front of it and behind it. Simply specify how large the window
needs to be. If it is zero, it will be equivalant to running process vectors.

These vectors are then stored in contextual_vectors(size_of_window).csv
"""

from sentence_transformers import SentenceTransformer 
import numpy as np
from transformers import AutoTokenizer, AutoModel
import json
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

def get_sentences():
    """
    This loads in our datasets
    """
    json_files = ["bible.json","bom.json","dc.json","pogp.json"]
    all_sentences = []
    books = []
    chapters = []
    references = []
    i = 0
    for file_name in json_files:
        file = []
        with open("datasets/" + file_name,'r') as f:
            file = json.loads(f.read())
        for p in file:
            all_sentences.append(p["text"])
            books.append( p["book"])
            references.append( p["reference"])
            chapters.append( p["chapter"])
            i += 1
        print(f"processed {file_name}")
    return all_sentences,books,chapters,references

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def generate_context_vectors(window):
    sentences,books,chapters,references = get_sentences()
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-roberta-large-v1')
    model = AutoModel.from_pretrained('sentence-transformers/all-roberta-large-v1').cuda()
    X = []
    for i,verse in enumerate(tqdm(sentences)):
        sentence_window = []
        t0 = 0
        t1 = 0
        t0_mask = 0
        t1_mask = 0
        end = False
        continue_on = True
        for j in range(-window,window+1):
            if i+j >= len(sentences):
                # This makes 
                continue_on = False
            if (continue_on and chapters[i+j] == chapters[i]):
                end = False
                sentence_window.append(sentences[i+j])
                tokens = tokenizer(sentences[i+j], padding=True, truncation=True, return_tensors='pt')
                sentence_len = tokens["input_ids"].shape[1]
                t0 = t1
                if t0 == 0:
                    t1 = t0+sentence_len -1
                else:
                    t1 = t0+sentence_len-2
                if(i+j == i):
                    t0_mask = t0
                    t1_mask = t1
                    end = True
        if(end):
            t1_mask += 1

        # Joins all of the sentences together and processes them
        sentence = "".join(sentence_window)
        emb = tokenizer(sentence, padding=True, truncation=True, return_tensors='pt')
        emb["input_ids"] = emb["input_ids"].cuda()
        emb["attention_mask"] = emb["attention_mask"].cuda()
        with torch.no_grad():
            model_output = model(**emb)
        
        # Creates a special mask to only pool over the desired tokens
        mask = torch.zeros_like(emb['attention_mask']).cuda()#FIXME
        mask[0,t0_mask:t1_mask] = 1

        # This applies our special attention mask to the pooling operation
        sentence_embeddings = mean_pooling(model_output, emb['attention_mask'].cuda()* mask).cpu().squeeze()
        X.append(sentence_embeddings.tolist())
        
    df = pd.DataFrame(X)
    print(df.head())
    df["Book"] = books
    df["Chapters"] = chapters
    df["References"] = references
    df.to_csv(f"generated_vectors/contextual_vectors({window}).csv")

window = 2 # This is how wide our context window is
new_vec, sentence = generate_vectors(window)