from .CCST import train_DGI, PCA_process
from torch_geometric.data import DataLoader
import torch
import pandas as pd
import sklearn 
import numpy as np
from torch_geometric.data import Data
import scipy as sp


def build_graph(adata, radius, lamb):
    # 读取各个节点的位置信息
    coor = pd.DataFrame(adata.obsm['spatial'])  
    coor.index = adata.obs.index
    coor.columns = ['imagerow', 'imagecol']

    nbrs = sklearn.neighbors.NearestNeighbors(radius=radius).fit(coor)
    _, indices = nbrs.radius_neighbors(coor, return_distance=True)

    edge_list = [[], []]
    for i in range(len(indices)):
        for j in range(len(indices[i])):
            edge_list[0].append(i)
            edge_list[1].append(indices[i][j])

    edge_weight = []
    for edge_id in range(len(edge_list[0])):
        edge_weight.append(lamb if edge_list[0][edge_id] == edge_list[1][edge_id] else 1 - lamb)

    data = Data(edge_index=torch.LongTensor(np.array([edge_list[0], edge_list[1]])), 
                x=torch.FloatTensor(adata[:, adata.var['highly_variable']].X.todense()), 
                edge_attr=torch.FloatTensor(edge_weight))

    print('>>> graph contains {} edges, average {} edges per node'.format(len(edge_list[0]), len(edge_list[0]) / adata.X.shape[0]))
    print(data.x.shape)

    return data


def train(adata, radius, epochs=5000, seed=2022, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), lamb=0.8, embedding_size=256):
    data = build_graph(adata, radius=radius, lamb=lamb)
    data_loader = DataLoader([data], batch_size=1)
    adata.obsm['CCST'] = train_DGI(data_loader, len(data.x[0]), embedding_size, epochs, 1e-6, seed, device)
    adata.obsm['CCST_pca'] = PCA_process(adata.obsm['CCST'], 50)

