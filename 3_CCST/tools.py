import squidpy as sq
import scanpy as sc
from CCST import train_DGI, PCA_process, Umap
from utilities import build_graph
import numpy as np
import pandas as pd
import os
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder 
from sklearn.cluster import KMeans
from torch_geometric.data import DataLoader


def train(adata, seed=2022):
    data = build_graph(adata, radius=250, lamb=0.8)
    data_loader = DataLoader([data], batch_size=1)
    adata.obsm['CCST'] = train_DGI(data_loader, len(data.x[0]), 128, 3000, 1e-6, seed)
    adata.obsm['CCST_pca'] = PCA_process(adata.obsm['CCST'], 50)

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

def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='CCST_pca', random_seed=2022):
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

def evaluate(adata, n_clusters, seed=2022):
    adata.obs['kmeans'] = [str(x) for x in KMeans(n_clusters=n_clusters, random_state=seed).fit(adata.obsm['CCST_pca']).predict(adata.obsm['CCST_pca'])]
    mclust_R(adata, num_cluster=n_clusters)
    
    label = LabelEncoder().fit_transform(adata.obs['cluster'])
    pred = LabelEncoder().fit_transform(adata.obs['kmeans'])
    pred_mclust = LabelEncoder().fit_transform(adata.obs['mclust'])
    
    print('pred:', metrics.adjusted_rand_score(label, pred)) 
    print('pred_clust:', metrics.adjusted_rand_score(label, pred_mclust)) 
    sc.pl.spatial(adata, color=['cluster', 'kmeans', 'mclust'])