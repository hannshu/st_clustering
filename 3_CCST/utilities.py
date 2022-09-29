import pandas as pd
import torch
import sklearn 
import numpy as np
from torch_geometric.data import Data

def build_graph(adata, radius, lamb, build_graph):
    if (build_graph):
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
    else:
        edge_list = np.nonzero(adata.obsp['connectivities'].todense())

    edge_weight = []
    for edge_id in range(len(edge_list[0])):
        edge_weight.append(lamb if edge_list[0][edge_id] == edge_list[1][edge_id] else 1 - lamb)

    data = Data(edge_index=torch.LongTensor(np.array([edge_list[0], edge_list[1]])), 
                x=torch.FloatTensor(adata[:, adata.var['highly_variable']].X.todense()), 
                edge_attr=torch.FloatTensor(edge_weight))

    print('>>> graph contains {} edges'.format(len(edge_list[0])))
    print(data.x.shape)

    return data
