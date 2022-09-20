from Train_STAGATE import train_STAGATE
from utils import mclust_R, Stats_Spatial_Net, Cal_Spatial_Net
import squidpy as sq
import scanpy as sc
import numpy as np
import pandas as pd
import os
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder 
from sklearn.cluster import KMeans

def train(adata, epochs, k_cutoff=None, rad_cutoff=None, seed=2022):
    if (k_cutoff):
        Cal_Spatial_Net(adata, k_cutoff=6, model='KNN') # 生成网络，保存在.uns['Spatial_Net']中 
    else:
        Cal_Spatial_Net(adata, rad_cutoff=rad_cutoff) # 生成网络，保存在.uns['Spatial_Net']中 
    return train_STAGATE(adata, n_epochs=epochs, random_seed=seed)

def get_dlpfc_data(id):
    section_list = ['151507', '151508', '151509', '151510', '151669', '151670', 
                    '151671', '151672', '151673', '151674', '151675', '151676']
    section_id = section_list[id]

    adata = sc.read_visium(path=os.path.join('..', 'dataset', 'DLPFC', section_id))
    adata.var_names_make_unique()

    Ann_df = pd.read_csv(os.path.join('..', 'dataset', 'DLPFC', section_id, 
                                      'ground_truth.txt'), sep='\t', header=None, index_col=0)
    Ann_df.columns = ['Ground Truth']
    adata.obs['cluster'] = Ann_df.loc[adata.obs_names, 'Ground Truth']
    
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    cluster_num = len(set(adata.obs['cluster'])) - 1
    print('>>> dataset id: {}, size: {}, cluster: {}.'.format(section_id, adata.X.shape, cluster_num))
    return adata, cluster_num

def evaluate(adata, n_clusters, seed=2022):
    adata = mclust_R(adata, n_clusters) # 对embedding进行聚类，聚类结果保存在.obs['mclust']中
    adata.obs['kmeans'] = [str(x) for x in KMeans(n_clusters=n_clusters, random_state=seed).fit(adata.obsm['STAGATE']).predict(adata.obsm['STAGATE'])]

    label = LabelEncoder().fit_transform(adata.obs['cluster'])
    pred = LabelEncoder().fit_transform(adata.obs['mclust'])
    pred_kmeans = LabelEncoder().fit_transform(adata.obs['kmeans'])
    print('pred:', metrics.adjusted_rand_score(label, pred))    # 计算预测值与真实值之间的兰德系数
    print('pred_kmeans:', metrics.adjusted_rand_score(label, pred_kmeans))    # 计算预测值与真实值之间的兰德系数
    sc.pl.spatial(adata, color=['cluster', 'mclust', 'kmeans'])