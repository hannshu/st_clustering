import scanpy as sc
import pandas as pd
import numpy as np
import os
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder 
from sklearn.cluster import KMeans
import squidpy as sq

from calculate_adj import *
from util import *
from SpaGCN import *


def train(adata, n_clusters, seed, l_preset=None):
    
    # 生成邻接矩阵(这里只计算距离)
    pos = adata.obsm['spatial'].T
    x_array = pos[0]
    y_array = pos[1]

    adj = calculate_adj_matrix(x_array, y_array, histology=False)
    
    # 对features进行初始化
    prefilter_genes(adata,min_cells=3) # avoiding all genes are zeros
    prefilter_specialgenes(adata)
    #Normalize and take log for UMI
    sc.pp.normalize_per_cell(adata)
    sc.pp.log1p(adata)
    
    # 计算用于归一化邻接矩阵的超参数l
    if (l_preset):
        l = l_preset
    else:
        p=0.5 
        l=search_l(p, adj, start=0.01, end=1000, tol=0.01, max_run=100)
    
    # 确定louvain算法需要的分辨率
    r_seed=t_seed=n_seed=seed
    #Seaech for suitable resolution
    res=search_res(adata, adj, l, n_clusters, start=0.7, step=0.1, tol=5e-3, lr=0.05, max_epochs=20, r_seed=r_seed, t_seed=t_seed, n_seed=n_seed)
    
    # GCN
    clf=SpaGCN()
    clf.set_l(l)
    #Set seed
    random.seed(r_seed)
    torch.manual_seed(t_seed)
    np.random.seed(n_seed)
    #Run
    clf.train(adata,adj,init_spa=True,init="louvain",res=res, tol=5e-3, lr=0.05, max_epochs=200)
    y_pred, _, embed=clf.predict()
    adata.obs["pred"]= y_pred
    adata.obs["pred"]=adata.obs["pred"].astype('category')
    #Do cluster refinement(optional)
    #shape="hexagon" for Visium data, "square" for ST data.
    adj_2d=calculate_adj_matrix(x=x_array,y=y_array, histology=False)
    refined_pred=refine(sample_id=adata.obs.index.tolist(), pred=adata.obs["pred"].tolist(), dis=adj_2d, shape="hexagon")
    adata.obs["refined_pred"]=refined_pred
    adata.obs["refined_pred"]=adata.obs["refined_pred"].astype('category')
    adata.obsm['SpaGCN'] = embed.cpu().detach().numpy()

def evaluate(adata, n_clusters, seed=2022, spot_size=None):
    adata = mclust_R(adata, n_clusters, random_seed=seed)
    adata.obs['pred_kmeans'] = [str(x) for x in KMeans(n_clusters=n_clusters, random_state=seed).fit(adata.obsm['SpaGCN']).predict(adata.obsm['SpaGCN'])]
    
    obs_df = adata.obs.dropna()

    label = LabelEncoder().fit_transform(obs_df['cluster'])
    pred_mclust_embed = LabelEncoder().fit_transform(obs_df['mclust'])
    pred = LabelEncoder().fit_transform(obs_df['pred'])
    pred_kmeans = LabelEncoder().fit_transform(obs_df['pred_kmeans'])
    pred_refine = LabelEncoder().fit_transform(obs_df["refined_pred"])

    print('pred_mclust:', metrics.adjusted_rand_score(label, pred_mclust_embed))
    print('pred:', metrics.adjusted_rand_score(label, pred))
    print('pred_kmeans:', metrics.adjusted_rand_score(label, pred_kmeans))
    print('pred_refine:', metrics.adjusted_rand_score(label, pred_refine))
    
    if (spot_size):
        sc.pl.spatial(adata, color=['cluster', 'mclust', 'pred', 'pred_kmeans', 'refined_pred'], spot_size=spot_size)
    else:
        sc.pl.spatial(adata, color=['cluster', 'mclust', 'pred', 'pred_kmeans', 'refined_pred'])

def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='SpaGCN', random_seed=2022):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """
    
    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']

    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
    mclust_res = np.array(res[-2])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    return adata