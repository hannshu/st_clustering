import pandas as pd
import scanpy as sc
import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data
from torch_geometric.nn import Node2Vec
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder 

def build_spatial_graph(adata, components, radius=None, knears=None):
    assert (None == radius and knears) or (radius and None == knears)
    
    coor = pd.DataFrame(adata.obsm['spatial'])  
    coor.index = adata.obs.index
    coor.columns = ['row', 'ecol']

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

def build_feature_graph(adata, features, spatial_edge_index, 
                        walk_length, walk_times, n_neighbors, 
                        node2vec_p, node2vec_q,
                        seed=2022, epochs=200,
                        device='cuda' if torch.cuda.is_available() else 'cpu'):
    adata = adata[:, adata.var['highly_variable']]
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, random_state=seed)
    edge_list = np.nonzero(adata.obsp['distances'].todense())

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    model = Node2Vec(spatial_edge_index, embedding_dim=features, walk_length=walk_length,
                     context_size=5, walks_per_node=walk_times,
                     num_negative_samples=2, p=node2vec_p, q=node2vec_q).to(device)

    loader = model.loader(batch_size=128, shuffle=True)
    optimizer = torch.optim.Adam(list(model.parameters()), lr=1e-3)

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
    
    data = Data(edge_index=torch.LongTensor(np.array(edge_list)),
                x=torch.FloatTensor(x))

    return data


def louvain_modify(adata, resolution, edges, seed):
    sc.pp.neighbors(adata, random_state=seed)
    sc.tl.louvain(adata, resolution=resolution, key_added='pre_trained_louvain_label', random_state=seed)
    adata.obs['pre_trained_louvain_label'] = LabelEncoder().fit_transform(adata.obs['pre_trained_louvain_label'])
    print('>>> finish louvain, begin to purne graph')

    edge_index = [[], []]
    for item in tqdm(edges.T, desc='>>> purning graph'):
        if (adata[int(item[0])].obs['pre_trained_louvain_label'].item() == adata[int(item[1])].obs['pre_trained_louvain_label'].item()):
            edge_index[0].append(int(item[0]))
            edge_index[1].append(int(item[1]))
    
    print('>>> Modified graph contains {} edges, average {} edges per node.'.format(len(edge_index[0]), len(edge_index[0]) / adata.X.shape[0]))

    return torch.LongTensor(np.array(edge_index))


