import torch
from tqdm import tqdm
from sklearn.decomposition import PCA

from .STAGATE import STAGATE as STA
from .utilities import *


def train(adata, radius=None, knears=None, components_spatial=3000, 
          walk_length=45, walk_times=10, n_neighbors=6,
          node2vec_p=1, node2vec_q=1, louvain_resolution=None,
          components_features=256, embedding_dim=60, lr=1e-3,
          epochs_spatial=500, epochs_features=200, seed=2022,
          hidden_layer_dim_spatial=512, hidden_layer_dim_features=64,
          pca_comps=30,
          device = 'cuda' if torch.cuda.is_available() else 'cpu'):
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    data_spatial = build_spatial_graph(adata, components_spatial, radius, knears)
    if (louvain_resolution):
        print('>>> use louvain to modify spatial graph')
        modified_spatial_index = louvain_modify(adata, louvain_resolution, data_spatial.edge_index, seed)
    else:
        modified_spatial_index = data_spatial.edge_index
    data_features = build_feature_graph(components_features, modified_spatial_index, 
                                        walk_length, walk_times, 
                                        node2vec_p, node2vec_q, seed=seed, 
                                        epochs=epochs_features, device='cuda')

    # 建立模型
    model_spatial = STA(hidden_dims=[components_spatial, 
                            hidden_layer_dim_spatial,
                            embedding_dim // 2]).to(device)
    loss_spatial_func = torch.nn.MSELoss().to(device)
    optim_spatial = torch.optim.Adam(model_spatial.parameters(), lr=lr, weight_decay=1e-4)

    data_spatial = data_spatial.to(device)
    print('>>> spatial model data(features) size: ({}, {})'.format(data_spatial.x.shape[0], data_spatial.x.shape[1]))

    model_spatial.train()
    for _ in tqdm(range(epochs_spatial), desc='>>> training spatial model'):
        optim_spatial.zero_grad()
        _, output_spatial = model_spatial(data_spatial)
        loss_spatial = loss_spatial_func(data_spatial.x, output_spatial)
        loss_spatial.backward()
        torch.nn.utils.clip_grad_norm_(model_spatial.parameters(), 5)
        optim_spatial.step()

    model_spatial.eval()
    with torch.no_grad():
        embedding_spatial, _ = model_spatial(data_spatial.to(device))

    adata.obsm['embedding'] = torch.cat([embedding_spatial.cpu(), data_features.x.cpu()], dim=1).cpu().detach().numpy()
    adata.obsm['embedding+pca'] = PCA(n_components=pca_comps).fit_transform(adata.obsm['embedding'])
    adata.obsm['STAGATE'] = embedding_spatial.cpu().detach().numpy()
    adata.obsm['spatial_embedding'] = data_features.x.cpu().detach().numpy()