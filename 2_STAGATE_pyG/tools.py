from Train_STAGATE import train_STAGATE
from utils import mclust_R, Stats_Spatial_Net, Cal_Spatial_Net, build_Spatial_Net
import squidpy as sq
import scanpy as sc
import numpy as np
import pandas as pd
import os
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder 
from sklearn.cluster import KMeans
import torch

def train(adata, epochs, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), k_cutoff=None, rad_cutoff=None, seed=2022, build_graph=True):
    if (build_graph):
        if (k_cutoff):
            Cal_Spatial_Net(adata, k_cutoff=6, model='KNN') # 生成网络，保存在.uns['Spatial_Net']中 
        else:
            Cal_Spatial_Net(adata, rad_cutoff=rad_cutoff) # 生成网络，保存在.uns['Spatial_Net']中 
    else:
        build_Spatial_Net(adata)
    return train_STAGATE(adata, n_epochs=epochs, random_seed=seed, device=device)

def evaluate(adata, n_clusters, seed=2022, spot_size=None, show=True, name='cluster'):
    adata = mclust_R(adata, n_clusters) # 对embedding进行聚类，聚类结果保存在.obs['mclust']中
    adata.obs['kmeans'] = [str(x) for x in KMeans(n_clusters=n_clusters, random_state=seed).fit(adata.obsm['STAGATE']).predict(adata.obsm['STAGATE'])]

    label = LabelEncoder().fit_transform(adata.obs[name])
    pred = LabelEncoder().fit_transform(adata.obs['mclust'])
    pred_kmeans = LabelEncoder().fit_transform(adata.obs['kmeans'])
    print('pred:', metrics.adjusted_rand_score(label, pred))    # 计算预测值与真实值之间的兰德系数
    print('pred_kmeans:', metrics.adjusted_rand_score(label, pred_kmeans))    # 计算预测值与真实值之间的兰德系数

    if (show):
        if (spot_size):
            sc.pl.spatial(adata, color=[name, 'mclust', 'kmeans'], spot_size=spot_size)
        else:
            sc.pl.spatial(adata, color=[name, 'mclust', 'kmeans'])
