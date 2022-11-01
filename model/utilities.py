import os
import numpy as np
import pandas as pd
import scanpy as sc

from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder 

import torch
from torch_geometric.data import Data
from torch_geometric.nn import Node2Vec

from tqdm import tqdm


def build_spatial_graph(adata, components, radius=None, knears=None):
    assert (None == radius and knears) or (radius and None == knears)
    
    adata = adata[:, adata.var['highly_variable']]
    coor = pd.DataFrame(adata.obsm['spatial'])  
    coor.index = adata.obs.index
    coor.columns = ['row', 'col']

    if (radius):
        nbrs = NearestNeighbors(radius=radius).fit(coor)
        _, indices = nbrs.radius_neighbors(coor, return_distance=True)
    else:
        nbrs = NearestNeighbors(n_neighbors=knears+1).fit(coor)
        _, indices = nbrs.kneighbors(coor)

    edge_list = [[], []]
    for i in range(len(indices)):
        for j in range(len(indices[i])):
            edge_list[0].append(i)
            edge_list[1].append(indices[i][j])

    if (components != adata.X.todense().shape[1]):
        pca = PCA(n_components=components).fit(adata.X.todense()).transform(adata.X.todense())
    else:
        pca = adata.X.todense()

    print('>>> The graph contains {} edges, average {} edges per node.'.format(len(edge_list[0]), len(edge_list[0]) / adata.X.shape[0]))
    
    data = Data(edge_index=torch.LongTensor(np.array(edge_list)), 
                x=torch.FloatTensor(pca))

    print('>>> Building spatial graph success!')
    return data


def build_feature_graph(features, spatial_edge_index, 
                        walk_length, walk_times,  
                        node2vec_p, node2vec_q,
                        seed=2022, epochs=200, lr=1e-3,
                        device='cuda' if torch.cuda.is_available() else 'cpu'):

    model = Node2Vec(spatial_edge_index, embedding_dim=features, walk_length=walk_length,
                     context_size=5, walks_per_node=walk_times,
                     num_negative_samples=2, p=node2vec_p, q=node2vec_q).to(device)

    loader = model.loader(batch_size=128)
    optimizer = torch.optim.Adam(list(model.parameters()), lr=lr)

    model.train()
    for _ in tqdm(range(epochs), desc='>>> node2vec'):
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
    
    model.eval()
    with torch.no_grad():    
        x = model().cpu().detach().numpy()
    
    data = Data(x=torch.FloatTensor(x))

    return data


def louvain_modify(adata, resolution, edges, seed=2022, show=False):
    adata = adata[:, adata.var['highly_variable']]
    sc.pp.neighbors(adata, random_state=seed)
    sc.tl.louvain(adata, resolution=resolution, key_added='pre_trained_louvain_label', random_state=seed)
    adata.obs['pre_trained_louvain_label'] = [str(x) for x in LabelEncoder().fit_transform(adata.obs['pre_trained_louvain_label'])]
    print('>>> finish louvain, begin to prune graph')

    if (show):
        sc.pl.spatial(adata, color=['pre_trained_louvain_label'])

    label = dict(zip(range(adata.X.shape[0]), adata.obs['pre_trained_louvain_label']))
    df = pd.DataFrame(edges.T)
    df.columns = ['node1', 'node2']

    df['node1_label'] = df['node1'].map(label)
    df['node2_label'] = df['node2'].map(label)

    new_df = df[df['node1_label'] == df['node2_label']]
    edge_index = [np.array(new_df['node1']), np.array(new_df['node2'])]
    
    print('>>> pruned graph contains {} edges, average {} edges per node.'.format(len(edge_index[0]), len(edge_index[0]) / adata.X.shape[0]))

    return torch.LongTensor(np.array(edge_index))


