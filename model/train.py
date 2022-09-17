from torch_geometric.nn import Node2Vec
from models import my_model, my_loss
import torch
from tqdm import tqdm
from utilities import build_feature_graph, build_graph


def train(adata, in_features, label=None, epochs=50, lr=1e-3, embedding_size=128,
            walk_length=20, context_size=5, walks_per_node=10, node2vec_p=1, node2vec_q=1,
            q_cluster=25
    ):
    seed = 2022
    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 构造特征所需的图
    graph_gcn = build_graph(adata, 200, in_features).to(device)
    graph_node2vec = build_feature_graph(adata).to(device)

    # 构造模型
    model = my_model(in_features, embedding_size, 
        # graph_node2vec, embedding_size, walk_length, context_size, walks_per_node,
        # node2vec_p, node2vec_q
    ).to(device)
    loss_function = my_loss(q_cluster, seed, device).to(device)
    optimizer_model = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    model.train()
    for _ in tqdm(range(epochs)):
        optimizer_model.zero_grad()
        output = model(graph_gcn)
        loss = loss_function(output, label)
        loss.backward()
        optimizer_model.step()

    model.eval()
    with torch.no_grad():
        output = model(graph_gcn)

    return output
