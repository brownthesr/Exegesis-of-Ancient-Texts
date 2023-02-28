"""
This file processes subsets of the data using UMAP and PCA.
This file should be used to visualize how the different verses
relate to each other in high dimensional space.
"""

import pandas as pd
import numpy as np
from umap import UMAP
from sentence_transformers import SentenceTransformer 
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


import seaborn as sns
import matplotlib.pyplot as plt

book_subset = [ "Moroni","Mormon"]
df = pd.read_csv("processed_vectors/contextual_vectors(2).csv",index_col=0)
df = df[df["Book"].isin(book_subset)]
df = df.reset_index(drop=True)
df_x = df.iloc[:,:-3]# This excludes the information columns
X = df_x.to_numpy()

sns.set_theme()

scaler = StandardScaler()
pca = PCA(n_components=.9)
umap = UMAP(
    n_components=2,
    # not f for n_neighbors
    n_neighbors=15,

    min_dist=0.0001,
    spread=1,

    # metric='cosine',
    metric='cosine',
    init='spectral',
    # # init='random',
    # random_state=0
)

print("rescaling data")
scaler.fit(X)
X_scaled = scaler.transform(X)

print("Reducing Data")
pca.fit(X_scaled)

print(f"reducing compnents via PCA to {pca.n_components_} components")
X_reduced = pca.transform(X_scaled)

print("UMAP")
clusters = umap.fit_transform(X_reduced)

df_plot = pd.DataFrame(clusters,columns=["x","y"])
df_plot["books"] = df["Book"]
df_plot["references"] = df["References"]
df_plot["chapter"] = df["Chapters"]

sns.set_palette(sns.color_palette()[2:])
sns.relplot(data=df_plot,x="x",y="y",hue="books",kind="scatter")
mng = plt.get_current_fig_manager()
mng.full_screen_toggle()
plt.savefig(f"plots/plot{35}.png")