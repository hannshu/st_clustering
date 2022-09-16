import pandas as pd
import scanpy as sc
import torch
import sklearn 
import numpy as np
from sklearn.decomposition import PCA
from torch_geometric.data import Data

def build_graph(adata, radius, components):
    adata = adata[:, adata.var['highly_variable']]
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

    pca = PCA(n_components=components).fit(adata.X.todense()).transform(adata.X.todense())
    data = Data(edge_index=torch.LongTensor(np.array([edge_list[0], edge_list[1]])), 
                x=torch.FloatTensor(pca))

    return data


def build_feature_graph(adata):
    adata = adata[:, adata.var['highly_variable']]
    sc.pp.neighbors(adata)

    edgeList = np.nonzero(adata.obsp['distances'].todense())
    data = Data(edge_index=torch.LongTensor(np.array([edgeList[0], edgeList[1]])))

    return data