import pandas as pd
import scanpy as sc
import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data

from gensim.models import word2vec
import networkx as nx
import random
import time

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
    
    graph = nx.Graph()
    for i in range(len(spatial_data.edge_index[0])):
        graph.add_edge(int(spatial_data.edge_index[0][i]), 
                       int(spatial_data.edge_index[1][i]))
    sentences = random_walk(graph, walk_length, walk_times, seed)
    print('>>> Random walk finished!')
    
    start_time = time.time()
    word2vec_model = word2vec.Word2Vec(sentences, vector_size=features, sg=1, hs=0, 
                                       negative=5, seed=seed, epochs=500, workers=12)
    x = word2vec_model.wv.vectors
    end_time = time.time()
    
    data = Data(edge_index=torch.LongTensor(np.array([edge_list[0], edge_list[1]])),
                x=torch.FloatTensor(x))
    
    print('>>> Building features graph success! (word2vec time: {}min{}s)'.
          format(int((end_time - start_time) // 60), int((end_time - start_time) % 60)))
    return data
