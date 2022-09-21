import scanpy as sc
import os
import torch
from tqdm import tqdm

from STAGATE import STAGATE
from utilities import *

from sklearn import metrics
from sklearn.preprocessing import LabelEncoder 
from sklearn.cluster import KMeans

def train(adata, radius=None, knears=None, components_spatial=3000, 
          walk_length=45, walk_times=10, n_neighbors=6,
          components_features=256, embedding_dim=60, lr=1e-3,
          epochs_spatial=500, epochs_features=500, seed=2022,
          hidden_layer_dim_spatial=512, hidden_layer_dim_features=128,
          device = 'cuda' if torch.cuda.is_available() else 'cpu'):
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    data_spatial = build_spatial_graph(adata, components_spatial, radius, knears)
    data_features = build_feature_graph(adata, components_features, data_spatial, 
                                        walk_length, walk_times, n_neighbors, seed)

    # 建立模型
    model_spatial = STAGATE(hidden_dims=[components_spatial, 
                            hidden_layer_dim_spatial,
                            embedding_dim // 2]).to(device)
    model_features = STAGATE(hidden_dims=[components_features, 
                            hidden_layer_dim_features,
                            embedding_dim // 2]).to(device)
    loss_spatial_func = torch.nn.MSELoss().to(device)
    loss_features_func = torch.nn.MSELoss().to(device)
    optim_spatial = torch.optim.Adam(model_spatial.parameters(), lr=lr, weight_decay=1e-4)
    optim_features = torch.optim.Adam(model_features.parameters(), lr=lr, weight_decay=1e-4)

    data_spatial = data_spatial.to(device)
    data_features = data_features.to(device)
    print('>>> spatial model data(features) size: ({}, {})'.format(data_spatial.x.shape[0], data_spatial.x.shape[1]))
    print('>>> feature model data(deepwalk) size: ({}, {})'.format(data_features.x.shape[0], data_features.x.shape[1]))

    model_spatial.train()
    for _ in tqdm(range(epochs_spatial), desc='>>> training spatial model'):
        optim_spatial.zero_grad()
        _, output_spatial = model_spatial(data_spatial)
        loss_spatial = loss_spatial_func(data_spatial.x, output_spatial)
        loss_spatial.backward()
        torch.nn.utils.clip_grad_norm_(model_spatial.parameters(), 5)
        optim_spatial.step()
        
    model_features.train()
    for _ in tqdm(range(epochs_features), desc='>>> training features model'):
        optim_features.zero_grad()
        _, output_features = model_features(data_features)
        loss_features = loss_features_func(data_features.x, output_features)
        loss_features.backward()
        torch.nn.utils.clip_grad_norm_(model_features.parameters(), 5)
        optim_features.step()

    model_spatial.eval()
    model_features.eval()
    with torch.no_grad():
        embedding_spatial, _ = model_spatial(data_spatial.to(device))
        embedding_features, _ = model_features(data_features.to(device))
    return (torch.cat([embedding_spatial, embedding_features], dim=1).cpu().detach().numpy(), 
            embedding_spatial.cpu().detach().numpy(), 
            embedding_features.cpu().detach().numpy())
    
def evaluate(adata, embed, spatial, features, pca_components, n_clusters, name='cluster', seed=2022):
    pca = PCA(n_components=pca_components).fit(embed).transform(embed)
    
    adata = mclust_R(adata, n_clusters, embed, random_seed=seed, name='mclust_embed')
    adata = mclust_R(adata, n_clusters, spatial, random_seed=seed, name='mclust_spatial')
    adata.obs['embed'] = [str(x) for x in KMeans(n_clusters=n_clusters, random_state=seed).fit(embed).predict(embed)]
    adata.obs['pred'] = [str(x) for x in KMeans(n_clusters=n_clusters, random_state=seed).fit(pca).predict(pca)]
    adata.obs['pred_spatial'] = [str(x) for x in KMeans(n_clusters=n_clusters, random_state=seed).fit(spatial).predict(spatial)]
    adata.obs['pred_features'] = [str(x) for x in KMeans(n_clusters=n_clusters, random_state=seed).fit(features).predict(features)]
    
    label = LabelEncoder().fit_transform(adata.obs[name])
    pred_mclust_embed = LabelEncoder().fit_transform(adata.obs['mclust_embed'])
    pred_mclust_spatial = LabelEncoder().fit_transform(adata.obs['mclust_spatial'])
    pred_embed = LabelEncoder().fit_transform(adata.obs['embed'])
    pred = LabelEncoder().fit_transform(adata.obs['pred'])
    pred_spatial = LabelEncoder().fit_transform(adata.obs['pred_spatial'])
    pred_features = LabelEncoder().fit_transform(adata.obs['pred_features'])
    
    print('pred_mclust_embed:', metrics.adjusted_rand_score(label, pred_mclust_embed))
    print('pred_mclust_spatial:', metrics.adjusted_rand_score(label, pred_mclust_spatial))
    print('pred_embed:', metrics.adjusted_rand_score(label, pred_embed))
    print('pred_pca:', metrics.adjusted_rand_score(label, pred))
    print('pred_spatial:', metrics.adjusted_rand_score(label, pred_spatial))
    print('pred_features:', metrics.adjusted_rand_score(label, pred_features))
    
    sc.pl.spatial(adata, color=[name, 'mclust_embed', 'mclust_spatial', 'embed', 'pred', 'pred_spatial', 'pred_features'])
    
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

def mclust_R(adata, num_cluster, data, name, modelNames='EEE', random_seed=2020):
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

    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(data), num_cluster, modelNames)
    mclust_res = np.array(res[-2])

    adata.obs[name] = mclust_res
    adata.obs[name] = adata.obs[name].astype('int')
    adata.obs[name] = adata.obs[name].astype('category')
    return adata


def train_1(adata, radius=None, knears=None, components_spatial=3000, 
          walk_length=45, walk_times=10, n_neighbors=6,
          components_features=256, embedding_dim=60, lr=1e-3,
          epochs_spatial=500, epochs_features=500, seed=2022,
          hidden_layer_dim_spatial=512, hidden_layer_dim_features=128,
          device = 'cuda' if torch.cuda.is_available() else 'cpu'):
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    data_spatial = build_spatial_graph(adata, components_spatial, radius, knears)
    data_features = build_feature_graph(adata, components_features, data_spatial, 
                                        walk_length, walk_times, n_neighbors, seed)

    # 建立模型
    model_spatial = STAGATE(hidden_dims=[components_spatial, 
                            hidden_layer_dim_spatial,
                            embedding_dim // 2]).to(device)
    model_features = STAGATE(hidden_dims=[components_features, 
                            hidden_layer_dim_features,
                            embedding_dim // 2]).to(device)
    loss_spatial_func = torch.nn.MSELoss().to(device)
    loss_features_func = torch.nn.MSELoss().to(device)
    optim_spatial = torch.optim.Adam(model_spatial.parameters(), lr=lr, weight_decay=1e-4)
    optim_features = torch.optim.Adam(model_features.parameters(), lr=lr, weight_decay=1e-4)

    data_spatial = data_spatial.to(device)
    data_features = data_features.to(device)
    print('>>> spatial model data(features) size: ({}, {})'.format(data_spatial.x.shape[0], data_spatial.x.shape[1]))
    print('>>> feature model data(deepwalk) size: ({}, {})'.format(data_features.x.shape[0], data_features.x.shape[1]))

    model_spatial.train()
    for _ in tqdm(range(epochs_spatial), desc='>>> training spatial model'):
        optim_spatial.zero_grad()
        _, output_spatial = model_spatial(data_spatial)
        loss_spatial = loss_spatial_func(data_spatial.x, output_spatial)
        loss_spatial.backward()
        torch.nn.utils.clip_grad_norm_(model_spatial.parameters(), 5)
        optim_spatial.step()
        
    # model_features.train()
    # for _ in tqdm(range(epochs_features), desc='>>> training features model'):
    #     optim_features.zero_grad()
    #     _, output_features = model_features(data_features)
    #     loss_features = loss_features_func(data_features.x, output_features)
    #     loss_features.backward()
    #     torch.nn.utils.clip_grad_norm_(model_features.parameters(), 5)
    #     optim_features.step()

    model_spatial.eval()
    # model_features.eval()
    with torch.no_grad():
        embedding_spatial, _ = model_spatial(data_spatial.to(device))
        # embedding_features, _ = model_features(data_features.to(device))
        
    embedding_features = data_features.x
    return (torch.cat([embedding_spatial, embedding_features], dim=1).cpu().detach().numpy(), 
            embedding_spatial.cpu().detach().numpy(), 
            embedding_features.cpu().detach().numpy())