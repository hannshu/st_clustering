import scanpy as sc
import squidpy as sq
import pandas as pd
import os

def get_dlpfc_data(id):
    section_list = ['151507', '151508', '151509', '151510', '151669', '151670', 
                    '151671', '151672', '151673', '151674', '151675', '151676']
    if (isinstance(id, int) and id in range(12)):
        section_id = section_list[id]
    elif (isinstance(id, str) and id in section_list):
        section_id = id

    adata = sc.read_visium(path=os.path.join('dataset', 'DLPFC', section_id))
    adata.var_names_make_unique()

    Ann_df = pd.read_csv(os.path.join('dataset', 'DLPFC', section_id, 
                                      'ground_truth.txt'), sep='\t', header=None, index_col=0)
    Ann_df.columns = ['Ground Truth']
    adata.obs['cluster'] = Ann_df.loc[adata.obs_names, 'Ground Truth']
    
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    cluster_num = len(set(adata.obs['cluster'])) - 1
    print('>>> dataset name: dlpfc, slice id: {}, size: {}, cluster: {}.'.format(section_id, adata.X.shape, cluster_num))
    return adata, cluster_num

def get_visium_hne_data():
    adata = sq.datasets.visium_hne_adata(path=os.path.join('dataset', 'visium_hne.h5ad'))

    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    cluster_num = len(set(adata.obs['cluster']))
    print('>>> dataset name: visium_hne, size: {}, cluster: {}.'.format(adata.X.shape, cluster_num))
    return adata, cluster_num

def get_visium_fluo_data():
    adata = sq.datasets.visium_fluo_adata(path=os.path.join('dataset', 'visium_fluo.h5ad'))

    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    cluster_num = len(set(adata.obs['cluster']))
    print('>>> dataset name: visium_fluo, size: {}, cluster: {}.'.format(adata.X.shape, cluster_num))
    return adata, cluster_num

def get_human_breast_cancer_data():
    adata = sc.read_visium(path=os.path.join('dataset', 'Human_Breast_Cancer'))
    adata.var_names_make_unique()
    Ann_df = pd.read_csv(os.path.join('dataset', 'Human_Breast_Cancer', 
                                      'metadata.csv'), sep=',', header=None, index_col=0)
    Ann_df.columns = ['Ground Truth']
    adata.obs['cluster'] = Ann_df.loc[adata.obs_names, 'Ground Truth']

    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    cluster_num = len(set(adata.obs['cluster']))
    print('>>> dataset name: human_breast_cancer, size: {}, cluster: {}.'.format(adata.X.shape, cluster_num))
    return adata, cluster_num