import scanpy as sc
import torch
from STAGATE import STAGATE
from utilities import *

def train(adata, radius=300, components_spatial=3000, 
          components_features=200, embedding_dim=128, lr=1e-3, epochs=100):

    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    data_spatial = build_spatial_graph(adata, radius, components_spatial)
    data_features = build_feature_graph(adata, components_features, data_spatial)

    # 建立模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_spatial = STAGATE(hidden_dims=[components_spatial, 
                            (components_spatial + embedding_dim // 2) // 2,
                            embedding_dim // 2]).to(device)
    model_features = STAGATE(hidden_dims=[components_features, 
                            (components_features + embedding_dim // 2) // 2,
                            embedding_dim // 2]).to(device)
    loss_spatial_func = torch.nn.MSELoss().to(device)
    loss_features_func = torch.nn.MSELoss().to(device)
    optim_spatial = torch.optim.Adam(model_spatial.parameters(), lr=lr, weight_decay=0.005)
    optim_features = torch.optim.Adam(model_features.parameters(), lr=lr, weight_decay=0.005)

    model_spatial.train()
    model_features.train()
    for _ in tqdm(range(epochs), desc='training model'):
        optim_spatial.zero_grad()
        optim_features.zero_grad()
        _, output_spatial = model_spatial(data_spatial)
        _, output_features = model_features(data_features)
        loss_spatial = loss_spatial_func(output_spatial, data_spatial.x)
        loss_features = loss_features_func(output_features, data_features.x)
        loss_spatial.backward()
        loss_features.backward()
        optim_spatial.step()
        optim_features.step()

    embedding_spatial, _ = model_spatial(data_spatial)
    embedding_features, _ = model_features(data_features)
    return torch.cat([embedding_spatial, embedding_features], dim=1).cpu().detach().numpy()