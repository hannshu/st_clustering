##exocrine GCNG with normalized graph matrix 
import os
import sys
import matplotlib
# matplotlib.use('Agg')
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


from sklearn import metrics
from scipy import sparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, ChebConv, GATConv, DeepGraphInfomax, global_mean_pool, global_max_pool
from torch_geometric.data import Data, DataLoader


from tqdm import tqdm


def get_graph(adj, X):
    # create sparse matrix
    row_col = []
    edge_weight = []
    rows, cols = adj.nonzero()
    edge_nums = adj.getnnz() 
    for i in range(edge_nums):
        row_col.append([rows[i], cols[i]])
        edge_weight.append(adj.data[i])
    edge_index = torch.tensor(np.array(row_col), dtype=torch.long).T
    edge_attr = torch.tensor(np.array(edge_weight), dtype=torch.float)

    graph_bags = []
    graph = Data(x=torch.tensor(X.copy(), dtype=torch.float), edge_index=edge_index, edge_attr=edge_attr)  
    graph_bags.append(graph)
    return graph_bags


class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(Encoder, self).__init__()
        self.conv = GCNConv(in_channels, hidden_channels)
        self.conv_2 = GCNConv(hidden_channels, hidden_channels)
        self.conv_3 = GCNConv(hidden_channels, hidden_channels)
        self.conv_4 = GCNConv(hidden_channels, hidden_channels)
        
        self.prelu = nn.PReLU(hidden_channels)
        
    def forward(self, data):
        x, edge_index, edge_weight = data.x.clone(), data.edge_index, data.edge_attr
        x = self.conv(x, edge_index, edge_weight=edge_weight)
        x = self.conv_2(x, edge_index, edge_weight=edge_weight)
        x = self.conv_3(x, edge_index, edge_weight=edge_weight)
        x = self.conv_4(x, edge_index, edge_weight=edge_weight)
        x = self.prelu(x)

        return x

class my_data():
    def __init__(self, x, edge_index, edge_attr):
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr

def corruption(data):
    x = data.x[torch.randperm(data.x.size(0))]
    return my_data(x, data.edge_index, data.edge_attr)

def train_DGI(data_loader, in_channels, hidden, epochs, lr, seed, device):
    
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    DGI_model = DeepGraphInfomax(
        hidden_channels=hidden,
        encoder=Encoder(in_channels=in_channels, hidden_channels=hidden),
        summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
        corruption=corruption).to(device)
    DGI_optimizer = torch.optim.Adam(DGI_model.parameters(), lr=lr)

    import datetime
    start_time = datetime.datetime.now()

    for _ in tqdm(range(epochs)):
        DGI_model.train()
        DGI_optimizer.zero_grad()
        
        for data in data_loader:
            data = data.to(device)
            pos_z, neg_z, summary = DGI_model(data=data)
            DGI_loss = DGI_model.loss(pos_z, neg_z, summary)
            DGI_loss.backward()

            DGI_optimizer.step()

    end_time = datetime.datetime.now()
    print('Training time in seconds: ', (end_time-start_time).seconds)

    result, _, _ = DGI_model(data)
    result = result.to('cpu').detach().numpy()
    return result


def PCA_process(X, nps):
    from sklearn.decomposition import PCA
    print('Shape of data to PCA:', X.shape)
    pca = PCA(n_components=nps)
    X_PC = pca.fit_transform(X)     #等价于pca.fit(X) pca.transform(X)
    print('Shape of data output by PCA:', X_PC.shape)
    print('PCA recover:', pca.explained_variance_ratio_.sum())
    return X_PC


from sklearn.cluster import KMeans, DBSCAN, AffinityPropagation
def Kmeans_cluster(X_embedding, n_clusters, merge=False):
    cluster_model = KMeans(n_clusters=n_clusters, init='k-means++', n_init=100, max_iter=1000, tol=1e-6)
    cluster_labels = cluster_model.fit_predict(X_embedding)

    # merge clusters with less than 3 cells
    if merge:
        cluster_labels = merge_cluser(X_embedding, cluster_labels)

    score = metrics.silhouette_score(X_embedding, cluster_labels, metric='euclidean')
    
    return cluster_labels, score

