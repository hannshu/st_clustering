import pandas as pd
import scanpy as sc
import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data
from torch_geometric.nn import Node2Vec
from tqdm import tqdm


def build_spatial_graph(adata, radius, components):
    adata = adata[:, adata.var['highly_variable']]
    coor = pd.DataFrame(adata.obsm['spatial'])  
    coor.index = adata.obs.index
    coor.columns = ['imagerow', 'imagecol']

    nbrs = NearestNeighbors(radius=radius).fit(coor)
    _, indices = nbrs.radius_neighbors(coor, return_distance=True)

    edge_list = [[], []]
    for i in range(len(indices)):
        for j in range(len(indices[i])):
            edge_list[0].append(i)
            edge_list[1].append(indices[i][j])

    pca = PCA(n_components=components).fit(adata.X.todense()).transform(adata.X.todense())
    data = Data(edge_index=torch.LongTensor(np.array([edge_list[0], edge_list[1]])), 
                x=torch.FloatTensor(pca))

    print('>>> Building spatial graph success!')
    return data


def build_feature_graph(adata, features, spatial_data):
    adata = adata[:, adata.var['highly_variable']]
    sc.pp.neighbors(adata)
    edge_list = np.nonzero(adata.obsp['distances'].todense())

    # node2vec
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Node2Vec(spatial_data.edge_index, embedding_dim=features, walk_length=20,
                     context_size=10, walks_per_node=10, num_negative_samples=1).to(device)
    loader = model.loader(batch_size=128, shuffle=True, num_workers=4)
    optim = torch.optim.Adam(model.parameters(), lr=0.01)

    model.train()
    for _ in tqdm(range(200), desc='training node embeddings'):
        for pos_rw, neg_rw in loader:
            optim.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optim.step()

    model.eval()
    x = model()
    data = Data(edge_index=torch.LongTensor(np.array([edge_list[0], edge_list[1]])),
                x=torch.FloatTensor(x))

    print('>>> Building features graph success!')
    return data