import pandas as pd
import scanpy as sc
import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data
from torch_geometric.nn import Node2Vec
from tqdm import tqdm

from gensim.models import word2vec
import networkx as nx
import random
import os


def build_spatial_graph(adata, components, radius=None, knears=None):
    
    assert (None == radius and knears) or (radius and None == knears)
    
    adata = adata[:, adata.var['highly_variable']]
    coor = pd.DataFrame(adata.obsm['spatial'])  
    coor.index = adata.obs.index
    coor.columns = ['imagerow', 'imagecol']

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

    print('>>> The graph contains {} edges.'.format(len(edge_list[0])))
    
    data = Data(edge_index=torch.LongTensor(np.array([edge_list[0], edge_list[1]])), 
                x=torch.FloatTensor(pca))

    print('>>> Building spatial graph success!')
    return data

def random_walk(graph, walk_length, walk_times, seed):
    random.seed(seed)
    sentences = []
    
    for node in graph.nodes:
        for _ in range(walk_times):
            sentence = [node]
            cur_node = node
            for _ in range(walk_length - 1):
                neighbors = list(graph.neighbors(cur_node))
                if (0 == len(neighbors)):
                    break
                cur_node = neighbors[random.randint(0, len(neighbors) - 1)]
                sentence.append(cur_node)
            sentences.append(sentence)
            
    return sentences


def build_feature_graph(adata, features, spatial_data, walk_length, walk_times, seed):
    adata = adata[:, adata.var['highly_variable']]
    sc.pp.neighbors(adata)
    edge_list = np.nonzero(adata.obsp['distances'].todense())

    # node2vec
    # model = Node2Vec(spatial_data.edge_index, embedding_dim=features, walk_length=45,
    #                  context_size=10, walks_per_node=10, num_negative_samples=5).to(device)
    # loader = model.loader(batch_size=128, shuffle=True, num_workers=4)
    # optim = torch.optim.Adam(model.parameters(), lr=0.01)

    # model.train()
    # for _ in tqdm(range(epochs), desc='training node embeddings'):
    #     for pos_rw, neg_rw in loader:
    #         optim.zero_grad()
    #         loss = model.loss(pos_rw.to(device), neg_rw.to(device))
    #         loss.backward()
    #         optim.step()

    # model.eval()
    # x = model()
    
    graph = nx.Graph()
    for i in range(len(spatial_data.edge_index[0])):
        graph.add_edge(int(spatial_data.edge_index[0][i]), 
                       int(spatial_data.edge_index[1][i]))
    sentences = random_walk(graph, walk_length, walk_times, seed)
    print('>>> Random walk finished!')
    
    word2vec_model = word2vec.Word2Vec(sentences, vector_size=features, sg=1, hs=0, 
                                       negative=5, seed=seed, epochs=500, workers=12)
    x = word2vec_model.wv.vectors
    
    data = Data(edge_index=torch.LongTensor(np.array([edge_list[0], edge_list[1]])),
                x=torch.FloatTensor(x))
    
    print('>>> Building features graph success!')
    return data

if __name__ == '__main__':
    section_list = ['151507', '151508', '151509', '151510', '151669', '151670', '151671', '151672', '151673', '151674', '151675', '151676']
    section_id = section_list[0]

    # 读取section_id这个slice的数据
    adata = sc.read_visium(path=os.path.join('dataset', 'DLPFC', section_id))
    adata.var_names_make_unique()
    
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # 为数据添加ground truth
    Ann_df = pd.read_csv(os.path.join('dataset', 'DLPFC', section_id, 'ground_truth.txt'), sep='\t', header=None, index_col=0)
    Ann_df.columns = ['Ground Truth']
    adata.obs['Cluster'] = Ann_df.loc[adata.obs_names, 'Ground Truth']
    
    data_spatial = build_spatial_graph(adata, 300, 3000)    
    data_features = build_feature_graph(adata, 500, data_spatial, 45, 10, 2022)