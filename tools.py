import scanpy as sc
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder 
from sklearn.cluster import KMeans


def mclust_R(adata, num_cluster, used_obsm, result_name, modelNames='EEE', random_seed=2022):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.

    from STAGATE source code
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

    adata.obs[result_name] = mclust_res
    adata.obs[result_name] = adata.obs[result_name].astype('int')
    adata.obs[result_name] = adata.obs[result_name].astype('category')
    return adata


def louvain(adata, n_neighbors, use_rep, result_name, resolution, seed=2022):
    pre_processed_adata = adata.copy()
    sc.pp.neighbors(pre_processed_adata, n_neighbors=n_neighbors, random_state=seed, use_rep=use_rep)
    sc.tl.louvain(pre_processed_adata, random_state=seed, resolution=resolution)

    adata.obs[result_name] = pre_processed_adata.obs['louvain']


def evaluate(adata, n_clusters, name_list=None, label_name='cluster', test_item=[True, True, True], pred_list=None, louvain_neighbors=8, louvain_resolution=0.8, seed=2022):
    pred_name_list = []

    if (name_list):
        # cluster
        for item in name_list:
            assert(item in adata.obsm)

            if (test_item[0]):
                adata = mclust_R(adata, n_clusters, used_obsm=item, result_name='mclust_pred_'+item, random_seed=seed)   # mclust
                pred_name_list.append('mclust_pred_'+item)
            if (test_item[1]):
                adata.obs['kmeans_pred_' + item] = [str(x) for x in KMeans(n_clusters=n_clusters, random_state=seed).fit(adata.obsm[item]).predict(adata.obsm[item])]   # kmeans
                pred_name_list.append('kmeans_pred_' + item)
            if (test_item[2]):
                louvain(adata, n_neighbors=louvain_neighbors, use_rep=item, resolution=louvain_resolution, result_name='louvain_pred_' + item) # louvain
                pred_name_list.append('louvain_pred_' + item)

    if (pred_list):
        pred_name_list += pred_list

    obs_df = adata.obs.dropna() # drop NA spot

    # count ARI
    label = LabelEncoder().fit_transform(obs_df[label_name])
    for item in pred_name_list:
        pred = LabelEncoder().fit_transform(obs_df[item])
        print('>>> {}: {}'.format(item, metrics.adjusted_rand_score(label, pred)))
    
    # display
    sc.pl.spatial(adata, color=['cluster']+pred_name_list)
