import pandas as pd
import scipy.sparse as sp
import torch
import sklearn 
import numpy as np
from torch_geometric.data import Data

def build_graph(adata, radius, lamb):
    # 读取各个节点的位置信息
    coor = pd.DataFrame(adata.obsm['spatial'])  
    coor.index = adata.obs.index
    coor.columns = ['imagerow', 'imagecol']

    nbrs = sklearn.neighbors.NearestNeighbors(radius=radius).fit(coor)
    distances, indices = nbrs.radius_neighbors(coor, return_distance=True)
    # node_list = []
    # for it in range(indices.shape[0]):
    #     node_list.append(pd.DataFrame(zip([it]*indices[it].shape[0], indices[it], distances[it])))

    # node_df = pd.concat(node_list)
    # node_df.columns = ['Cell1', 'Cell2', 'Distance']

    # node_df = node_df.loc[node_df['Distance']>0,]               # 删除所有指向节点本身的记录
    # id_cell_trans = dict(zip(range(coor.shape[0]), np.array(coor.index), )) # 建立节点名称和编号的索引
    # # 将原来记录中的编号修改为节点原有的名称
    # node_df['Cell1'] = node_df['Cell1'].map(id_cell_trans) 
    # node_df['Cell2'] = node_df['Cell2'].map(id_cell_trans)

    # cells = np.array(adata.obs_names)
    # cells_id_tran = dict(zip(cells, range(cells.shape[0])))
    # node_df['Cell1'] = node_df['Cell1'].map(cells_id_tran)
    # node_df['Cell2'] = node_df['Cell2'].map(cells_id_tran)

    # # 生成邻接矩阵
    # G = sp.coo_matrix((np.ones(node_df.shape[0]), (node_df['Cell1'], node_df['Cell2'])), shape=(adata.n_obs, adata.n_obs))
    # G = G + sp.eye(G.shape[0])

    edge_list = [[], []]
    for i in range(len(indices)):
        for j in range(len(indices[i])):
            edge_list[0].append(i)
            edge_list[1].append(indices[i][j])

    # edgeList = np.nonzero(G)    # 记录所有边的集合
    edge_weight = []
    for edge_id in range(len(edge_list[0])):
        edge_weight.append(lamb if edge_list[0][edge_id] == edge_list[1][edge_id] else 1 - lamb)

    data = Data(edge_index=torch.LongTensor(np.array([edge_list[0], edge_list[1]])), 
                x=torch.FloatTensor(adata[:, adata.var['highly_variable']].X.todense()), 
                edge_attr=torch.FloatTensor(edge_weight))

    print(data.x.shape)

    return data
