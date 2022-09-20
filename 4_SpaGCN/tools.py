import scanpy as sc
import pandas as pd
import numpy as np
import os
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder 
from sklearn.cluster import KMeans

from calculate_adj import *
from util import *
from SpaGCN import *


def train(adata, n_clusters, seed):
    
    # 生成邻接矩阵(这里只计算距离)
    pos = adata.obsm['spatial'].T
    x_array = pos[0]
    y_array = pos[1]

    adj = calculate_adj_matrix(pos[0], pos[1], histology=False)
    
    # 对features进行初始化
    prefilter_genes(adata,min_cells=3) # avoiding all genes are zeros
    prefilter_specialgenes(adata)
    #Normalize and take log for UMI
    sc.pp.normalize_per_cell(adata)
    sc.pp.log1p(adata)
    
    # 计算用于归一化邻接矩阵的超参数l
    p=0.5 
    l=search_l(p, adj, start=0.01, end=1000, tol=0.01, max_run=100)
    
    # 确定louvain算法需要的分辨率
    r_seed=t_seed=n_seed=seed
    #Seaech for suitable resolution
    res=search_res(adata, adj, l, n_clusters, start=0.9, step=0.1, tol=5e-3, lr=0.05, max_epochs=20, r_seed=r_seed, t_seed=t_seed, n_seed=n_seed)
    
    # GCN
    clf=SpaGCN()
    clf.set_l(l)
    #Set seed
    random.seed(r_seed)
    torch.manual_seed(t_seed)
    np.random.seed(n_seed)
    #Run
    clf.train(adata,adj,init_spa=True,init="louvain",res=res, tol=5e-3, lr=0.05, max_epochs=1000)
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
    adata = mclust_R(adata, n_clusters, random_seed=seed)
    adata.obs['pred_kmeans'] = [str(x) for x in KMeans(n_clusters=n_clusters, random_state=seed).fit(adata.obsm['SpaGCN']).predict(adata.obsm['SpaGCN'])]
    
    label = LabelEncoder().fit_transform(adata.obs['cluster'])
    pred_mclust_embed = LabelEncoder().fit_transform(adata.obs['mclust'])
    pred = LabelEncoder().fit_transform(adata.obs['pred'])
    pred_kmeans = LabelEncoder().fit_transform(adata.obs['pred_kmeans'])
    
    print('pred_mclust:', metrics.adjusted_rand_score(label, pred_mclust_embed))
    print('pred:', metrics.adjusted_rand_score(label, pred))
    print('pred_kmeans:', metrics.adjusted_rand_score(label, pred_kmeans))
    
    sc.pl.spatial(adata, color=['cluster', 'mclust', 'pred', 'pred_kmeans'])

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