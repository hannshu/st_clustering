import scanpy as sc
import numpy as np

def train(adata, n_cluster, lr=1e-2, seed=2022, comp=30, resolution=1.0):
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    pre_processed_adata = adata[:, adata.var['highly_variable']]
    
    sc.tl.pca(pre_processed_adata, n_comps=comp, random_state=seed)
    sc.pp.neighbors(pre_processed_adata, random_state=seed, use_rep='X_pca')
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
