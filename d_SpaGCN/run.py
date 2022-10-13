import scanpy as sc
import numpy as np

from .calculate_adj import *
from .util import *
from .SpaGCN import *

import SpaGCN as spg

def train(adata, n_clusters, seed=2022, l_preset=None):
    
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
        l=search_l(0.5, adj, start=0.01, end=1000, tol=0.01, max_run=100)
    
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


def train_spg(adata, n_clusters):
    x_pixel=adata.obsm["spatial"].T[0].tolist()
    y_pixel=adata.obsm["spatial"].T[1].tolist()
    x_array=adata.obsm["spatial"].T[0].tolist()
    y_array=adata.obsm["spatial"].T[1].tolist()

    s=1
    b=49
    adj=spg.calculate_adj_matrix(x=x_pixel,y=y_pixel, x_pixel=x_pixel, y_pixel=y_pixel, beta=b, alpha=s, histology=False)

    adata.var_names_make_unique()
    spg.prefilter_genes(adata,min_cells=3) # avoiding all genes are zeros
    spg.prefilter_specialgenes(adata)
    #Normalize and take log for UMI
    sc.pp.normalize_per_cell(adata)
    sc.pp.log1p(adata)

    p=0.5 
    #Find the l value given p
    l=spg.search_l(p, adj, start=0.01, end=1000, tol=0.01, max_run=100)

    #If the number of clusters known, we can use the spg.search_res() fnction to search for suitable resolution(optional)
    #For this toy data, we set the number of clusters=7 since this tissue has 7 layers
    n_clusters=n_clusters
    #Set seed
    r_seed=t_seed=n_seed=100
    #Seaech for suitable resolution
    res=spg.search_res(adata, adj, l, n_clusters, start=0.7, step=0.1, tol=5e-3, lr=0.05, max_epochs=20, r_seed=r_seed, t_seed=t_seed, n_seed=n_seed)

    clf=spg.SpaGCN()
    clf.set_l(l)
    #Set seed
    random.seed(r_seed)
    torch.manual_seed(t_seed)
    np.random.seed(n_seed)
    #Run
    clf.train(adata,adj,init_spa=True,init="louvain",res=res, tol=5e-3, lr=0.05, max_epochs=200)
    y_pred, prob=clf.predict()
    adata.obs["pred"]= y_pred
    adata.obs["pred"]=adata.obs["pred"].astype('category')
    #Do cluster refinement(optional)
    #shape="hexagon" for Visium data, "square" for ST data.
    adj_2d=spg.calculate_adj_matrix(x=x_array,y=y_array, histology=False)
    refined_pred=spg.refine(sample_id=adata.obs.index.tolist(), pred=adata.obs["pred"].tolist(), dis=adj_2d, shape="hexagon")
    adata.obs["refined_pred"]=refined_pred
    adata.obs["refined_pred"]=adata.obs["refined_pred"].astype('category')
