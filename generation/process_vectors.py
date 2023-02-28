"""
This file encodes each verse using just a simple roberta model
and then stores them in sentence_vectors.csv
"""

from sentence_transformers import SentenceTransformer 
import numpy as np
from transformers import AutoTokenizer,AutoModel
import json
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import pandas as pd
import torch.nn as nn

def get_sentences():
    json_files = ["bible.json","bom.json","dc.json","pogp.json"]
    all_sentences = []
    books = []
    chapters = []
    references = []
    i = 0
    for file_name in json_files:
        file = []
        with open("datasets/"+file_name,'r') as f:
            file = json.loads(f.read())
        for p in file:
            all_sentences.append(p["text"])
            books.append( p["book"])
            references.append( p["reference"])
            chapters.append( p["chapter"])
            i += 1
        print(f"processed {file_name}")
    return all_sentences,books,chapters,references

sentences,books,chapters,ids = get_sentences()

print(f"Importing model using {torch.cuda.device_count()}")
model = SentenceTransformer('sentence-transformers/all-roberta-large-v1',device="cuda")

print("Encoding sentences")
X = []
for s in tqdm( sentences ):
   tmp = model.encode( s )
   X.append( tmp )

X = np.array(X)
df = pd.DataFrame(X)
df["Book"] = books
df["Chapters"] = chapters
df["References"] = ids
df.to_csv("sentence_vectors.csv")