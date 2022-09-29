import squidpy as sq
import scanpy as sc
import numpy as np
import pandas as pd
import os
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder 
from sklearn.cluster import KMeans

def train(adata, n_cluster, n_neighbors=6, lr=1e-2, seed=2022):
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    pre_processed_adata = adata[:, adata.var['highly_variable']]
    
    sc.pp.neighbors(pre_processed_adata, n_neighbors=n_neighbors, random_state=seed)
    resolution = 1.0
    latest_lr = lr
    cnt = 0
    while (True):
        cnt += 1
        sc.tl.louvain(pre_processed_adata, random_state=seed, resolution=resolution)

        if (len(set(pre_processed_adata.obs['louvain'])) == n_cluster):
            print('>>> resolution: {}, pre_cluster: {}, cluster: {}'.format(resolution, len(set(pre_processed_adata.obs['louvain'])), n_cluster))
            break
        elif (len(set(pre_processed_adata.obs['louvain'])) > n_cluster):
            resolution -= lr
            print('>>> cluster: {}, modify resolution: {}, lr: {}'.format(len(set(pre_processed_adata.obs['louvain'])), resolution, lr))
        else:
            resolution += lr
            print('>>> cluster: {}, modify resolution: {}, lr: {}'.format(len(set(pre_processed_adata.obs['louvain'])), resolution, lr))

        if (0 == cnt % 5):
            lr = latest_lr
        else:
            lr *= np.random.random()

    adata.obs['louvain'] = pre_processed_adata.obs['louvain']

def evaluate(adata, spot_size=None):
    label = LabelEncoder().fit_transform(adata.obs['cluster'])
    pred_louvain = LabelEncoder().fit_transform(adata.obs['louvain'])
    print('>>> pred:', metrics.adjusted_rand_score(label, pred_louvain))

    if (spot_size):
        sc.pl.spatial(adata, color=['cluster', 'louvain'], spot_size=spot_size)
    else:
        sc.pl.spatial(adata, color=['cluster', 'louvain'])
