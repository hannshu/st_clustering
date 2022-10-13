import torch
from tqdm import tqdm
from sklearn.decomposition import PCA

from .STAGATE import STAGATE
from .utilities import *


def train(adata, radius=None, knears=None, components_spatial=3000, 
          walk_length=45, walk_times=10, n_neighbors=6,
          node2vec_p=1, node2vec_q=1, louvain_resolution=None,
          components_features=256, embedding_dim=60, lr=1e-3,
          epochs_spatial=500, epochs_features=500, seed=2022,
          hidden_layer_dim_spatial=512, hidden_layer_dim_features=64,
          pca_comps=30,
          device = 'cuda' if torch.cuda.is_available() else 'cpu'):
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    adata = adata[:, adata.var['highly_variable']]
    data_spatial = build_spatial_graph(adata, components_spatial, radius, knears)
    if (louvain_resolution):
        print('>>> use louvain to modify spatial graph')
        modified_spatial_index = louvain_modify(adata, louvain_resolution, data_spatial.edge_index, seed)
    else:
        modified_spatial_index = data_spatial.edge_index
    data_features = build_feature_graph(adata, components_features, modified_spatial_index, 
                                        walk_length, walk_times, n_neighbors, 
                                        node2vec_p, node2vec_q, seed=seed, device='cuda')

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
    print('>>> feature model data(node2vec) size: ({}, {})'.format(data_features.x.shape[0], data_features.x.shape[1]))

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

    adata.obsm['embedding'] = torch.cat([embedding_spatial, embedding_features], dim=1).cpu().detach().numpy()
    adata.obsm['embedding+pca'] = PCA(n_components=pca_comps).fit_transform(adata.obsm['embedding'])
    adata.obsm['feature_embedding'] = embedding_spatial.cpu().detach().numpy()
    adata.obsm['spatial_embedding'] = embedding_features.cpu().detach().numpy()
    return adata


# def evaluate(adata, embed, spatial, features, pca_components, n_clusters, name='cluster', seed=2022, spot_size=None):
#     pca = PCA(n_components=pca_components).fit(embed).transform(embed)
    
#     adata.obs['embed'] = [str(x) for x in KMeans(n_clusters=n_clusters, random_state=seed).fit(embed).predict(embed)]
#     adata.obs['pred'] = [str(x) for x in KMeans(n_clusters=n_clusters, random_state=seed).fit(pca).predict(pca)]
#     adata.obs['pred_spatial'] = [str(x) for x in KMeans(n_clusters=n_clusters, random_state=seed).fit(spatial).predict(spatial)]
#     adata.obs['pred_features'] = [str(x) for x in KMeans(n_clusters=n_clusters, random_state=seed).fit(features).predict(features)]
    
#     label = LabelEncoder().fit_transform(adata.obs[name])
#     pred_embed = LabelEncoder().fit_transform(adata.obs['embed'])
#     pred = LabelEncoder().fit_transform(adata.obs['pred'])
#     pred_spatial = LabelEncoder().fit_transform(adata.obs['pred_spatial'])
#     pred_features = LabelEncoder().fit_transform(adata.obs['pred_features'])
    
#     print('spatial+features:', metrics.adjusted_rand_score(label, pred_embed))
#     print('spatial+features+pca:', metrics.adjusted_rand_score(label, pred))
#     print('spatial:', metrics.adjusted_rand_score(label, pred_spatial))
#     print('features:', metrics.adjusted_rand_score(label, pred_features))
    
#     if (spot_size):
#         sc.pl.spatial(adata, color=[name, 'embed', 'pred', 'pred_spatial', 'pred_features'], spot_size=spot_size)
#     else:
#         sc.pl.spatial(adata, color=[name, 'embed', 'pred', 'pred_spatial', 'pred_features'])
