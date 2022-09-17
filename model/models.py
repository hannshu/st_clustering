from torch_geometric.nn import GCNConv, Node2Vec
import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import KMeans

class my_model(nn.Module):

    def __init__(self, in_features, out_features,
                # graph_node2vec, embedding_size, walk_length, context_size, walks_per_node,
                # node2vec_p, node2vec_q
    ) -> None:
        super().__init__()

        # self.gcn1 = GraphConvolution(in_features=in_features, out_features=in_features)
        # self.gcn2 = GraphConvolution(in_features=in_features, out_features=out_features)
        self.gcn1 = GCNConv(in_channels=in_features, out_channels=out_features)
        # self.gcn1 = GCNConv(in_channels=in_features, out_channels=in_features)
        # self.gcn2 = GCNConv(in_channels=in_features, out_channels=out_features)
        # self.node2vec = Node2Vec(
        #     edge_index=graph_node2vec.edge_index, embedding_dim=embedding_size, walk_length=walk_length,
        #     context_size=context_size, walks_per_node=walks_per_node,
        #     p=node2vec_p, q=node2vec_q
        # )
        # self.dataloader = self.node2vec.loader()

        # self.fc = nn.Linear(in_features=out_features * 2, out_features=out_features)
        # self.relu = nn.ReLU()

    def forward(self, data):
        # X, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        # x_features = self.gcn(X, edge_index, edge_weight)
        # for pos_rw, neg_rw in self.dataloader:
        #     loss = self.node2vec.loss(pos_rw, neg_rw)
        #     loss.backward()
        X, edge_index = data.x, data.edge_index
        x_features = self.gcn1(X, edge_index)
        # x_features = self.gcn2(x_features, edge_index)

        # embed = torch.cat((x_features, self.node2vec()), dim=1)
        # embed = x_features
        # return self.relu(self.fc(embed))
        return nn.functional.elu(x_features)

class my_loss(nn.Module):

    def __init__(self, cluster, seed, device) -> None:
        super().__init__()

        self.cluster = cluster
        self.seed = seed
        self.device = device

    def forward(self, matrix, label=None):
        if (not isinstance(label, np.ndarray)):
            label = KMeans(n_clusters=self.cluster, random_state=self.seed).fit(matrix.cpu().detach().numpy()).predict(matrix.cpu().detach().numpy())

        cluster_centers = torch.zeros(self.cluster, matrix.shape[1]).to(self.device)
        for i, item in enumerate(matrix):
            cluster_centers[label[i]] += item
        label = list(label)
        for num in label:
            cluster_centers[label[i]] /= label.count(num)
        dis = nn.functional.pdist(cluster_centers)
        return -dis.mean()
