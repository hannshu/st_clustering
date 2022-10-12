import squidpy as sq
import scanpy as sc
from CCST import train_DGI, PCA_process
import utilities
import pandas as pd
import os
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder 
from sklearn.cluster import KMeans
from torch_geometric.data import DataLoader, DataListLoader
import torch


def train(adata, radius, epochs=3000, seed=2022, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), build_graph=True, lamb=0.8, embedding_size=256):
    data = utilities.build_graph(adata, radius=radius, lamb=lamb, build_graph=build_graph)
    data_loader = DataLoader([data], batch_size=1)
    adata.obsm['CCST'] = train_DGI(data_loader, len(data.x[0]), embedding_size, epochs, 1e-6, seed, device)
    adata.obsm['CCST_pca'] = PCA_process(adata.obsm['CCST'], 50)

def evaluate(adata, n_clusters, name='cluster', seed=2022, spot_size=None, show=True):
    adata.obs['kmeans'] = [str(x) for x in KMeans(n_clusters=n_clusters, random_state=seed).fit(adata.obsm['CCST_pca']).predict(adata.obsm['CCST_pca'])]
    
    obs_df = adata.obs.dropna()
    label = LabelEncoder().fit_transform(obs_df[name])
    pred = LabelEncoder().fit_transform(obs_df['kmeans'])
    
    print('pred:', metrics.adjusted_rand_score(label, pred)) 

    if (show):
        if (spot_size):
            sc.pl.spatial(adata, color=[name, 'kmeans'], spot_size=spot_size)
        else:
            sc.pl.spatial(adata, color=[name, 'kmeans'])
