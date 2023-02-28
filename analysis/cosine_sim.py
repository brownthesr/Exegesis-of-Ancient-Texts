"""
This script finds the scripture most related to the 
given sentence.
"""

import pandas as pd
import numpy as np
from umap import UMAP
from sentence_transformers import SentenceTransformer 
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler,normalize
import seaborn as sns
import matplotlib.pyplot as plt
import json

df = pd.read_csv("processed_vectors/contextual_vectors(2).csv",index_col=0)
df_x = df.iloc[:,:-3]# This excludes the information columns
X = df_x.to_numpy()
X = normalize(X,axis=1,norm="l2")

source = "Your mom is very angry with me"
model = SentenceTransformer('sentence-transformers/all-roberta-large-v1',device="cpu")

sim = np.array(model.encode(source))
sim = sim/np.linalg.norm(sim)
df_sim =  pd.DataFrame(X@sim,columns=["sim"])
df_sim["books"] = df["Book"]
df_sim["reference"] = df["References"]
df_sim["chapter"] = df["Chapters"]
with open("datasets/all_references.json","r") as file:
    scriptures = json.loads(file.read())
df_sim["text"] = df_sim["reference"].map(scriptures)

most_similar = df_sim.sort_values(by=["sim"],ascending=False)

print(f"{most_similar.iloc[0]['reference']} score: {most_similar.iloc[0]['sim']}")
print(most_similar.iloc[0]["text"])
